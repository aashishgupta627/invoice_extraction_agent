import json
import os
import time
from typing import Annotated, Literal, Sequence, TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from shared.db import save_invoice
from shared.extractors.openrouter_vision import OpenRouterExtractor
from shared.logging_config import get_logger, with_thread

logger = get_logger("extraction_agent")

load_dotenv()

EXTRACTION_TOOLS_DESC = ["extract_invoice"]


# --- State ---
class ExtractionState(TypedDict):
    file_path: str
    record: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]


# --- Tool ---
# NOTE: this tool is bound to the LLM so the model can decide *when* to
# extract, but the file_path it's actually run against always comes from
# graph state (see extract_tool_node), never from the model's arguments.
# This avoids a hallucinated/garbled path ever reaching the extractor.
@tool
def extract_invoice(file_path: str) -> str:
    """Extract structured invoice data from the given file path."""
    extractor = OpenRouterExtractor()
    data = extractor.extract(file_path)
    data["source_file"] = file_path
    data["extraction_backend"] = "openrouter_free_vision"
    data["status"] = "extracted"
    return json.dumps(data, indent=2)


# --- Nodes ---
def call_model(state: ExtractionState):
    file_path = state["file_path"]
    tlog = with_thread(logger, file_path)
    tlog.info("call_model entered")

    tools = [extract_invoice]
    tlog.info(f"Extraction agent bound tools: {[t.name for t in tools]}")

    model = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.0,
        timeout=60.0,
    ).bind_tools(tools, parallel_tool_calls=False)

    messages = state.get("messages", [])
    start = time.monotonic()
    try:
        if not messages:
            system = (
                f"Extract the invoice data from {file_path}. "
                "Call extract_invoice with the exact file path. Do not do anything else."
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

    tool_calls = getattr(response, "tool_calls", None)
    tlog.debug(f"Model response tool_calls: {tool_calls}")
    return {"messages": new_messages}


def extract_tool_node(state: ExtractionState):
    file_path = state["file_path"]  # authoritative — never trust model args
    tlog = with_thread(logger, file_path)
    tlog.info("extract_tool_node entered")

    last_msg = state["messages"][-1]
    tool_call = last_msg.tool_calls[0]
    model_supplied_path = tool_call.get("args", {}).get("file_path")
    if model_supplied_path and model_supplied_path != file_path:
        tlog.warning(
            f"Model supplied file_path={model_supplied_path!r} which differs "
            f"from authoritative state file_path={file_path!r}; using state value."
        )

    start = time.monotonic()
    try:
        result = extract_invoice.invoke({"file_path": file_path})
        record = json.loads(result)
        record.setdefault("validation_flags", [])
        record["status"] = "extracted"
        tlog.info(f"Extraction succeeded in {time.monotonic() - start:.2f}s")
        tlog.debug(f"Extracted record: {json.dumps(record)[:2000]}")
    except Exception as e:
        tlog.error(f"Extraction failed after {time.monotonic() - start:.2f}s", exc_info=True)
        result = f"Error extracting: {e}"
        record = {
            "source_file": file_path,
            "status": "error_extraction",
            "validation_flags": [f"Extraction failed: {e}"],
            "seller": {},
            "buyer": {},
            "invoice_number": "",
            "invoice_date": "",
            "line_items": [],
            "tax_breakup": {},
            "total_amount": 0.0,
        }

    try:
        save_invoice(record)
    except Exception:
        tlog.error("Failed to persist extraction record to DB", exc_info=True)
        raise

    return {
        "messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])],
        "record": record,
    }


def route_after_model(state: ExtractionState) -> Literal["extract_tool", "__end__"]:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "extract_tool"
    logger.warning(
        "Extraction model returned no tool call — ending without extraction. "
        f"file_path={state.get('file_path')}"
    )
    return "__end__"


# --- Build Graph ---
def build_extraction_graph(checkpointer: SqliteSaver):
    graph = StateGraph(ExtractionState)
    graph.add_node("call_model", call_model)
    graph.add_node("extract_tool", extract_tool_node)

    graph.set_entry_point("call_model")
    graph.add_conditional_edges(
        "call_model",
        route_after_model,
        {"extract_tool": "extract_tool", "__end__": END},
    )
    graph.add_edge("extract_tool", END)

    return graph.compile(checkpointer=checkpointer)


# Helper to run the agent (used by supervisor)
def run_extraction(file_path: str, checkpointer: SqliteSaver, thread_id: str) -> dict:
    tlog = with_thread(logger, thread_id)
    tlog.info(f"run_extraction starting for {file_path}")
    app = build_extraction_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    initial_state = {"file_path": file_path, "messages": []}
    try:
        final_state = app.invoke(initial_state, config)
    except Exception:
        tlog.error("Extraction graph invocation failed", exc_info=True)
        raise
    record = final_state.get("record", {})
    tlog.info(f"run_extraction finished with status={record.get('status')}")
    return record
