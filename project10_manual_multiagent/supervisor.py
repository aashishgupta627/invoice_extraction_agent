import json
import re
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from project10_manual_multiagent.extraction_agent import build_extraction_graph
from project10_manual_multiagent.validation_agent import build_validation_graph
from shared.db import save_invoice
from shared.logging_config import get_logger, with_thread
from shared.validate import validate_invoice

logger = get_logger("supervisor")

# Statuses that mean extraction already succeeded and produced a usable
# record — safe to skip straight to validation instead of re-extracting.
EXTRACTION_COMPLETE_STATUSES = {"extracted"}
# Statuses that mean extraction failed and should be retried from scratch.
EXTRACTION_RETRY_STATUSES = {"error_extraction", "error", None}


def _default_thread_id(file_path: str) -> str:
    """Deterministic thread_id derived from the file path.

    Previously this was f"{stem}_{uuid.uuid4()}" — a fresh random ID on
    every call. That silently broke LangGraph checkpointing across process
    restarts: the SqliteSaver can only resume a thread it has seen before,
    and a random ID guarantees it never has. Deriving the ID from the file
    path itself means the same invoice always maps to the same thread, so
    a rerun after a crash / Ctrl+C / interrupted approval can actually
    resume where it left off instead of starting over.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(file_path))


def get_user_decision(record: dict, flags: list):
    print("\n" + "=" * 60)
    print("APPROVAL REQUIRED")
    print(json.dumps(record, indent=2))
    print("Validation Flags:", ", ".join(flags) if flags else "None")
    print("=" * 60)
    while True:
        choice = input("Enter decision (approve/edit/reject): ").strip().lower()
        if choice == "approve":
            return "approve"
        if choice == "reject":
            return "reject"
        if choice == "edit":
            print("Paste updated invoice JSON, blank line to finish, 'cancel' to abort:")
            lines = []
            while True:
                line = input()
                if line.strip().lower() == "cancel":
                    return "reject"
                if line == "":
                    break
                lines.append(line)
            try:
                edited = json.loads("\n".join(lines))
                validated = validate_invoice(edited)
                return {"action": "edit", "record": validated}
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}")
                continue
        print("Please enter approve, edit, or reject.")


def run_pipeline(
    file_path: str,
    checkpointer: SqliteSaver,
    thread_id: Optional[str] = None,
    existing_record: Optional[dict] = None,
) -> dict:
    """Run extraction (unless a usable existing_record is supplied) then
    validation for a single invoice.

    existing_record: pass the DB row for this file when its status is
    already "extracted" (i.e. a prior run completed extraction but never
    reached validation — most commonly because that run was interrupted).
    This skips the extraction stage entirely rather than burning another
    model call to redo work that already succeeded.
    """
    if thread_id is None:
        thread_id = _default_thread_id(file_path)
    tlog = with_thread(logger, thread_id)

    can_reuse = (
        existing_record is not None
        and existing_record.get("status") in EXTRACTION_COMPLETE_STATUSES
    )

    if can_reuse:
        tlog.info(
            "Reusing previously extracted record; skipping extraction stage "
            f"(status={existing_record.get('status')})."
        )
        record = existing_record
    else:
        extract_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
        extract_app = build_extraction_graph(checkpointer)
        extract_state = {"file_path": file_path, "messages": []}
        tlog.info("Starting extraction stage")
        try:
            extract_result = extract_app.invoke(extract_state, extract_config)
        except Exception as e:
            tlog.error("Extraction stage raised an exception", exc_info=True)
            error_record = {
                "source_file": file_path,
                "status": "error_extraction",
                "validation_flags": [f"Extraction error: {e}"],
            }
            save_invoice(error_record)
            return error_record

        record = extract_result.get("record", {})
        status = record.get("status")

        if status in EXTRACTION_RETRY_STATUSES:
            tlog.error(f"Extraction did not produce a usable record (status={status})")
            if status is None:
                # Defensive: the extraction graph ended without ever
                # reaching extract_tool_node (e.g. the model returned no
                # tool call). Without this, the record has no status at
                # all and would silently fall through to validation with
                # an empty/garbage record instead of being flagged.
                record = {
                    "source_file": file_path,
                    "status": "error_extraction",
                    "validation_flags": ["Extraction graph ended without calling extract_invoice"],
                }
                save_invoice(record)
            return record

        tlog.info(f"Extraction stage completed, status={status}")

    # --- Validation ---
    validation_thread_id = f"{thread_id}_validation"
    validation_config = {"configurable": {"thread_id": validation_thread_id}, "recursion_limit": 15}
    validate_app = build_validation_graph(checkpointer)
    validation_started = False

    tlog.info("Starting validation stage")
    while True:
        try:
            if not validation_started:
                initial_state = {"record": record, "messages": [], "checks_run": [], "turn_count": 0}
                final_state = validate_app.invoke(initial_state, validation_config)
                validation_started = True
            else:
                final_state = validate_app.invoke(None, validation_config)
            final_record = final_state.get("record", {})
            tlog.info(f"Validation stage completed, status={final_record.get('status')}")
            return final_record
        except GraphInterrupt:
            snapshot = validate_app.get_state(validation_config)
            state_vals = snapshot.values
            record = state_vals.get("record", {})
            flags = record.get("validation_flags", [])
            tlog.info(f"Paused for human approval; {len(flags)} flag(s) present")
            decision = get_user_decision(record, flags)
            validate_app.invoke(Command(resume=decision), validation_config)
            validation_started = True
        except Exception as e:
            tlog.error("Validation stage raised an exception", exc_info=True)
            error_record = {
                "source_file": file_path,
                "status": "error",
                "validation_flags": [f"Validation error: {e}"],
            }
            save_invoice(error_record)
            return error_record
