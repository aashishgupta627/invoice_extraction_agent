import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Literal, Sequence, TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from shared.db import save_invoice
from shared.logging_config import get_logger, with_thread
from shared.validate import validate_invoice

logger = get_logger("validation_agent")

load_dotenv()

CHECK_NAMES = ("check_gstin", "check_state_tax_rule", "check_arithmetic")
MAX_MODEL_TURNS = 10  # hard safety cap independent of graph recursion_limit


# --- State ---
class ValidationState(TypedDict):
    record: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]
    checks_run: Annotated[List[str], operator.add]
    turn_count: int


# --- Validation implementation (authoritative — always runs against
# state["record"], never against anything the model supplies) ---
def _run_full_validation(record: dict) -> tuple[dict, list]:
    validated = validate_invoice(record)
    return validated, validated.get("validation_flags", [])


def _filter_flags(flags: list, keywords: tuple[str, ...]) -> list:
    lowered = [(f, f.lower()) for f in flags]
    return [f for f, fl in lowered if any(k.lower() in fl for k in keywords)]


# --- Tools exposed to the LLM ---
# Deliberately take NO arguments. Earlier versions accepted a record_json
# argument that the model was expected to fill in — but the model is never
# shown the actual record, so it had to hallucinate that payload. The
# hallucinated JSON (not the real extracted record) was what got validated
# and reported back, even though the correct record was separately (and
# silently) persisted to SQLite. Making these tools zero-argument removes
# the hallucination surface entirely: the model can only choose *which*
# check to run and *when*, never supply the data it runs against. The
# actual record always comes from graph state inside the node below.
@tool
def check_gstin() -> str:
    """Run the GSTIN format and state-code consistency check on the
    current invoice record."""
    raise NotImplementedError("Executed by validate_tool_node using graph state.")


@tool
def check_state_tax_rule() -> str:
    """Run the CGST/SGST vs IGST tax-type rule check on the current
    invoice record."""
    raise NotImplementedError("Executed by validate_tool_node using graph state.")


@tool
def check_arithmetic() -> str:
    """Run the line-item / taxable-value / total-amount arithmetic
    check on the current invoice record."""
    raise NotImplementedError("Executed by validate_tool_node using graph state.")


@tool
def commit_to_approved() -> str:
    """Submit the current invoice record for human approval. Only call
    this once all checks have passed with no outstanding flags."""
    raise NotImplementedError("Executed by commit_node using graph state.")


VALIDATION_TOOLS = [check_gstin, check_state_tax_rule, check_arithmetic, commit_to_approved]


def _thread_id_from_record(record: dict) -> str:
    return record.get("source_file", "unknown")


# --- Nodes ---
def call_model(state: ValidationState):
    record = state.get("record", {})
    thread_id = _thread_id_from_record(record)
    tlog = with_thread(logger, thread_id)
    tlog.info("call_model entered")

    turn_count = state.get("turn_count", 0) + 1
    if turn_count > MAX_MODEL_TURNS:
        tlog.error(f"Exceeded MAX_MODEL_TURNS={MAX_MODEL_TURNS}; forcing stop.")
        # No tool call -> route_after_model will terminate this safely.
        return {"messages": [], "turn_count": turn_count}

    tlog.info(f"Validation agent bound tools: {[t.name for t in VALIDATION_TOOLS]}")

    model = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.0,
        timeout=60.0,
    ).bind_tools(VALIDATION_TOOLS, parallel_tool_calls=False)

    messages = state.get("messages", [])
    start = time.monotonic()
    try:
        if not messages:
            system = (
                "You are a validation agent for GST invoices. You cannot see the "
                "invoice data directly — call check_gstin, check_state_tax_rule, "
                "and check_arithmetic (each takes no arguments) to run each check "
                "against the record already loaded by the system. Call each of "
                "the three checks exactly once. "
                "If, after all three checks, there are no outstanding flags, call "
                "commit_to_approved to request human approval. "
                "If there are flags, state them plainly and do NOT call "
                "commit_to_approved."
            )
            system_msg = SystemMessage(content=system)
            tlog.debug(f"System prompt: {system}")
            response = model.invoke([system_msg])
            new_messages = [system_msg, response]
        else:
            response = model.invoke(messages)
            new_messages = [response]
    except Exception:
        tlog.error("Model invocation failed", exc_info=True)
        raise
    finally:
        tlog.debug(f"Model call took {time.monotonic() - start:.2f}s")

    tlog.debug(f"Model response tool_calls: {getattr(response, 'tool_calls', None)}")
    return {"messages": new_messages, "turn_count": turn_count}


def validate_tool_node(state: ValidationState):
    record = state.get("record", {})
    thread_id = _thread_id_from_record(record)
    tlog = with_thread(logger, thread_id)
    tlog.info("validate_tool_node entered")

    last_msg = state["messages"][-1]
    tool_call = last_msg.tool_calls[0]
    tool_name = tool_call["name"]
    if tool_call.get("args"):
        tlog.debug(
            f"Ignoring model-supplied args for {tool_name} "
            f"(record always sourced from graph state): {tool_call['args']}"
        )

    start = time.monotonic()
    try:
        validated, flags = _run_full_validation(record)
    except Exception:
        tlog.error(f"Validation failed while running {tool_name}", exc_info=True)
        raise
    tlog.debug(f"Full validation took {time.monotonic() - start:.2f}s; flags={flags}")

    if tool_name == "check_gstin":
        relevant = _filter_flags(flags, ("gstin", "state"))
        result = f"GSTIN flags: {', '.join(relevant)}" if relevant else "GSTIN checks passed."
    elif tool_name == "check_state_tax_rule":
        relevant = _filter_flags(flags, ("cgst", "sgst", "igst", "inter-state"))
        result = f"Tax rule flags: {', '.join(relevant)}" if relevant else "Tax rule checks passed."
    elif tool_name == "check_arithmetic":
        relevant = _filter_flags(flags, ("taxable", "total", "arithmetic"))
        result = f"Arithmetic flags: {', '.join(relevant)}" if relevant else "Arithmetic checks passed."
    else:
        tlog.warning(f"Unexpected tool name routed to validate_tool_node: {tool_name}")
        result = "Unknown tool"

    tlog.info(f"{tool_name} -> {result}")

    try:
        save_invoice(validated)
    except Exception:
        tlog.error("Failed to persist validated record to DB", exc_info=True)
        raise

    return {
        "messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])],
        "record": validated,
        "checks_run": [tool_name],
    }


def finalize_no_commit_node(state: ValidationState):
    """Terminal node for the 'checks ran, flags remain (or model stalled),
    commit was never called' path. Without this, that case previously fell
    through to auto_validate -> call_model -> auto_validate in a loop that
    only stopped when the graph's recursion_limit raised an error, rather
    than ending in a clean, reportable status."""
    record = state.get("record", {})
    thread_id = _thread_id_from_record(record)
    tlog = with_thread(logger, thread_id)

    flags = record.get("validation_flags", [])
    status = "flagged"
    tlog.warning(
        f"Ending validation without commit. status={status} flags={flags} "
        f"checks_run={state.get('checks_run', [])}"
    )
    finalized = {**record, "status": status}
    try:
        save_invoice(finalized)
    except Exception:
        tlog.error("Failed to persist finalized (no-commit) record to DB", exc_info=True)
        raise
    return {"record": finalized}


def commit_node(state: ValidationState):
    record = state.get("record", {})
    thread_id = _thread_id_from_record(record)
    tlog = with_thread(logger, thread_id)
    tlog.info("commit_node entered")

    last_msg = state["messages"][-1]
    tool_call = last_msg.tool_calls[0]

    flags = record.get("validation_flags", [])
    if flags:
        tlog.warning(
            f"Model requested commit despite {len(flags)} outstanding flag(s): {flags}"
        )

    tlog.debug(f"Pausing for human approval. Record: {json.dumps(record)[:2000]}")
    try:
        resume = interrupt({"record": record, "flags": flags})
    except GraphInterrupt:
        tlog.info("GraphInterrupt raised; supervisor will handle resume")
        raise

    if resume == "approve":
        try:
            approved_dir = Path("output/approved/")
            approved_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(record.get("source_file", "unknown")).stem
            filename = approved_dir / f"{base_name}_{timestamp}.json"
            approved_record = {**record, "status": "approved"}
            with open(filename, "w") as f:
                json.dump(approved_record, f, indent=2)
            save_invoice(approved_record)
            result = f"Approved and saved to {filename}"
            tlog.info(result)
        except Exception:
            tlog.error("Failed to write/persist approved record", exc_info=True)
            raise
    elif resume == "reject":
        rejected_record = {**record, "status": "rejected"}
        save_invoice(rejected_record)
        result = "Invoice rejected."
        tlog.info(result)
    elif isinstance(resume, dict) and resume.get("action") == "edit":
        edited_record = resume["record"]
        validated = validate_invoice(edited_record)
        save_invoice(validated)
        result = "EDIT_REQUESTED"
        tlog.info(f"Edit requested; re-validated. New flags: {validated.get('validation_flags', [])}")
    else:
        tlog.warning(f"Unrecognized resume value {resume!r}; treating as reject.")
        rejected_record = {**record, "status": "rejected"}
        save_invoice(rejected_record)
        result = "Unknown decision - rejected."

    return {"messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])]}


def route_after_model(
    state: ValidationState,
) -> Literal["validate_tool", "commit_node", "finalize_no_commit"]:
    last_msg = state["messages"][-1]
    thread_id = _thread_id_from_record(state.get("record", {}))
    tlog = with_thread(logger, thread_id)

    tool_calls = getattr(last_msg, "tool_calls", None)
    if tool_calls:
        tool_name = tool_calls[0]["name"]
        tlog.info(f"Routing to tool: {tool_name}")
        if tool_name in CHECK_NAMES:
            return "validate_tool"
        if tool_name == "commit_to_approved":
            return "commit_node"
        tlog.warning(f"Unrecognized tool_call name {tool_name!r}; ending without commit.")
        return "finalize_no_commit"

    # No tool call. If the model still hasn't run all three checks, that's
    # a model failure to follow instructions, not a legitimate "done" state
    # — but we do NOT loop indefinitely (turn_count / MAX_MODEL_TURNS in
    # call_model already caps that); we just end cleanly either way.
    checks_run = set(state.get("checks_run", []))
    if checks_run < set(CHECK_NAMES):
        tlog.warning(
            f"Model stopped without calling all checks (ran={checks_run}); ending without commit."
        )
    else:
        tlog.info("All checks ran; model chose not to commit. Ending without commit.")
    return "finalize_no_commit"


# --- Build Graph ---
def build_validation_graph(checkpointer: SqliteSaver):
    graph = StateGraph(ValidationState)
    graph.add_node("call_model", call_model)
    graph.add_node("validate_tool", validate_tool_node)
    graph.add_node("finalize_no_commit", finalize_no_commit_node)
    graph.add_node("commit_node", commit_node)

    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "validate_tool": "validate_tool",
            "commit_node": "commit_node",
            "finalize_no_commit": "finalize_no_commit",
        },
    )
    graph.add_edge("validate_tool", "call_model")
    graph.add_edge("finalize_no_commit", END)
    graph.add_edge("commit_node", END)

    return graph.compile(checkpointer=checkpointer)
