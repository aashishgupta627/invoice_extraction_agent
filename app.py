import streamlit as st
import json
import sqlite3
import sys
import time
import tempfile
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to sys.path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.db import init_db, get_invoice, save_invoice, get_all_invoices
from shared.logging_config import get_logger
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

# Import graphs from Project 10
from project10_manual_multiagent.extraction_agent import build_extraction_graph
from project10_manual_multiagent.validation_agent import build_validation_graph
from project10_manual_multiagent.supervisor import _default_thread_id

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Invoice Pipeline (LangGraph)", layout="wide")
st.title("📄 Invoice Processing Pipeline — LangGraph")

# ----------------------------------------------------------------------
# Initialise database
# ----------------------------------------------------------------------
init_db()

# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------
if "phase" not in st.session_state:
    st.session_state.phase = "idle"          # idle | extracting | validating | needs_approval | resuming | done
if "files" not in st.session_state:
    st.session_state.files = []
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "records" not in st.session_state:
    st.session_state.records = []            # final records for each file
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []

# For approval
if "pending_record" not in st.session_state:
    st.session_state.pending_record = None
if "pending_flags" not in st.session_state:
    st.session_state.pending_flags = []
if "validation_config" not in st.session_state:
    st.session_state.validation_config = None
if "decision" not in st.session_state:
    st.session_state.decision = None
if "show_edit" not in st.session_state:
    st.session_state.show_edit = False

# For upload temp directory
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None

# Graphs and checkpointer (cached resource)
@st.cache_resource
def get_checkpointer():
    # Single connection for local single-user usage.
    # For multi-user, consider a connection pool or per-session connections.
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    return SqliteSaver(conn)

@st.cache_resource
def get_extraction_app():
    return build_extraction_graph(get_checkpointer())

@st.cache_resource
def get_validation_app():
    return build_validation_graph(get_checkpointer())

checkpointer = get_checkpointer()
extract_app = get_extraction_app()
validate_app = get_validation_app()

# ----------------------------------------------------------------------
# Helper: add a log message
# ----------------------------------------------------------------------
def add_log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"{timestamp} – {msg}")

# ----------------------------------------------------------------------
# Sidebar: input and controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Input")

    input_mode = st.radio("Input mode", ["Local folder path", "Upload files"])

    if input_mode == "Local folder path":
        folder_path = st.text_input("Folder path", value="invoices_folder")
        uploaded_files = None
    else:
        folder_path = None
        uploaded_files = st.file_uploader(
            "Upload invoice files (PDF, PNG, JPG, JPEG)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

    if st.button("🚀 Start batch", disabled=(st.session_state.phase not in ("idle", "done"))):
        # Build list of file paths
        file_paths = []
        if input_mode == "Local folder path":
            folder = Path(folder_path)
            if not folder.exists():
                st.error(f"Folder not found: {folder_path}")
            else:
                extensions = [".pdf", ".png", ".jpg", ".jpeg"]
                file_paths = [str(f) for f in folder.iterdir() if f.suffix.lower() in extensions]
        else:
            if uploaded_files:
                # Create a session-specific temp directory
                if st.session_state.temp_dir is None:
                    st.session_state.temp_dir = tempfile.mkdtemp(prefix="streamlit_uploads_")
                temp_path = Path(st.session_state.temp_dir)
                for uploaded_file in uploaded_files:
                    # Use original name, but avoid collisions by adding a short hash
                    file_bytes = uploaded_file.read()
                    # Derive a stable ID for checkpointing: original name + size
                    file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
                    safe_name = f"{uploaded_file.name}_{file_hash}"
                    dest = temp_path / safe_name
                    with open(dest, "wb") as f:
                        f.write(file_bytes)
                    file_paths.append(str(dest))
            else:
                st.error("No files uploaded.")

        if not file_paths:
            st.error("No invoice files found.")
        else:
            st.session_state.files = file_paths
            st.session_state.idx = 0
            st.session_state.records = []
            st.session_state.log_messages = []
            st.session_state.phase = "extracting"
            st.success(f"Found {len(file_paths)} files. Starting...")
            st.rerun()

    if st.button("🔄 Reset session"):
        # Clean up temp dir if exists
        if st.session_state.temp_dir:
            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ----------------------------------------------------------------------
# Log area (always visible)
# ----------------------------------------------------------------------
if st.session_state.log_messages:
    with st.expander("📋 Live Logs", expanded=True):
        for msg in st.session_state.log_messages[-30:]:
            st.text(msg)

# ----------------------------------------------------------------------
# State machine
# ----------------------------------------------------------------------

# ---- IDLE ----
if st.session_state.phase == "idle":
    st.info("Select input and click 'Start batch'.")

# ---- EXTRACTING ----
elif st.session_state.phase == "extracting":
    files = st.session_state.files
    idx = st.session_state.idx

    # Check if all files done
    if idx >= len(files):
        st.session_state.phase = "done"
        st.rerun()

    file_path = files[idx]
    add_log(f"Extracting: {Path(file_path).name}")

    # Check DB for terminal status (skip if already done)
    existing = get_invoice(file_path)
    status = existing.get("status") if existing else None
    if status in ("approved", "rejected", "flagged"):
        add_log(f"Skipping {Path(file_path).name} – already {status}")
        st.session_state.idx += 1
        st.rerun()

    # Run extraction graph
    thread_id = _default_thread_id(file_path)
    extract_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}

    try:
        extract_state = {"file_path": file_path, "messages": []}
        extract_result = extract_app.invoke(extract_state, extract_config)
        record = extract_result.get("record", {})
        rec_status = record.get("status")
        if rec_status in ("error_extraction", "error") or not record:
            add_log(f"Extraction failed for {Path(file_path).name} (status={rec_status})")
            st.session_state.idx += 1
            st.rerun()
        # Store record for validation
        st.session_state.current_record = record
        st.session_state.validation_thread_id = f"{thread_id}_validation"
        st.session_state.validation_config = {
            "configurable": {"thread_id": st.session_state.validation_thread_id},
            "recursion_limit": 15,
        }
        st.session_state.phase = "validating"
        add_log(f"Extraction OK. Starting validation for {Path(file_path).name}")
        st.rerun()
    except Exception as e:
        add_log(f"Extraction error: {e}")
        save_invoice({"source_file": file_path, "status": "error_extraction", "validation_flags": [str(e)]})
        st.session_state.idx += 1
        st.rerun()

# ---- VALIDATING ----
elif st.session_state.phase == "validating":
    file_path = st.session_state.files[st.session_state.idx]
    config = st.session_state.validation_config

    try:
        # Start validation (or continue if already started – but we always start fresh)
        initial_state = {
            "record": st.session_state.current_record,
            "messages": [],
            "checks_run": [],
            "turn_count": 0,
        }
        final_state = validate_app.invoke(initial_state, config)
        # If we get here, validation finished without interruption
        record = final_state.get("record", {})
        add_log(f"Validation finished for {Path(file_path).name} – status={record.get('status')}")
        st.session_state.records.append(record)
        st.session_state.idx += 1
        st.session_state.phase = "extracting"
        st.rerun()

    except GraphInterrupt:
        # Pause for human approval
        snapshot = validate_app.get_state(config)
        state_vals = snapshot.values
        record = state_vals.get("record", {})
        flags = record.get("validation_flags", [])
        st.session_state.pending_record = record
        st.session_state.pending_flags = flags
        st.session_state.phase = "needs_approval"
        add_log(f"⏸️ Approval needed for {Path(file_path).name}")
        st.rerun()

    except Exception as e:
        add_log(f"Validation error: {e}")
        error_record = {
            "source_file": file_path,
            "status": "error",
            "validation_flags": [f"Validation error: {e}"],
        }
        save_invoice(error_record)
        st.session_state.records.append(error_record)
        st.session_state.idx += 1
        st.session_state.phase = "extracting"
        st.rerun()

# ---- NEEDS_APPROVAL ----
elif st.session_state.phase == "needs_approval":
    st.subheader("✋ Approval Required")
    record = st.session_state.pending_record
    flags = st.session_state.pending_flags

    col1, col2 = st.columns([2, 1])
    with col1:
        st.json(record)
    with col2:
        if flags:
            st.warning(f"Flags: {', '.join(flags)}")
        else:
            st.success("No validation flags")

    st.markdown("---")
    decision = None

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("✅ Approve", use_container_width=True):
            decision = "approve"
    with col_b:
        if st.button("❌ Reject", use_container_width=True):
            decision = "reject"
    with col_c:
        if st.button("✏️ Edit", use_container_width=True):
            st.session_state.show_edit = True
            st.rerun()

    # Edit area (toggle)
    if st.session_state.get("show_edit", False):
        st.subheader("Edit Record")
        current_json = json.dumps(record, indent=2)
        edited_json = st.text_area("Edit JSON (must be valid)", current_json, height=300)
        col_edit1, col_edit2 = st.columns(2)
        with col_edit1:
            if st.button("Submit edited version"):
                try:
                    edited = json.loads(edited_json)
                    # Re-validate
                    from shared.validate import validate_invoice
                    validated = validate_invoice(edited)
                    decision = {"action": "edit", "record": validated}
                    st.session_state.show_edit = False
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
        with col_edit2:
            if st.button("Cancel edit"):
                st.session_state.show_edit = False
                st.rerun()

    if decision is not None:
        st.session_state.decision = decision
        st.session_state.phase = "resuming"
        st.rerun()

# ---- RESUMING ----
elif st.session_state.phase == "resuming":
    file_path = st.session_state.files[st.session_state.idx]
    config = st.session_state.validation_config
    decision = st.session_state.decision

    add_log(f"Resuming validation for {Path(file_path).name} with decision")

    try:
        final_state = validate_app.invoke(Command(resume=decision), config)
        record = final_state.get("record", {})
        add_log(f"Validation resumed and finished – status={record.get('status')}")
        st.session_state.records.append(record)
        # Clear temporary approval state
        st.session_state.decision = None
        st.session_state.pending_record = None
        st.session_state.pending_flags = []
        st.session_state.show_edit = False
        st.session_state.idx += 1
        st.session_state.phase = "extracting"
        st.rerun()
    except Exception as e:
        add_log(f"Error during resume: {e}")
        error_record = {
            "source_file": file_path,
            "status": "error",
            "validation_flags": [f"Resume error: {e}"],
        }
        save_invoice(error_record)
        st.session_state.records.append(error_record)
        st.session_state.decision = None
        st.session_state.pending_record = None
        st.session_state.pending_flags = []
        st.session_state.idx += 1
        st.session_state.phase = "extracting"
        st.rerun()

# ---- DONE ----
elif st.session_state.phase == "done":
    st.success("✅ Batch processing complete!")
    records = st.session_state.records

    if records:
        df = pd.DataFrame(records)
        st.dataframe(df)

        # Review queue – records with flags
        flagged = [r for r in records if r.get("validation_flags")]
        if flagged:
            st.subheader("📋 Review Queue (flagged)")
            for rec in flagged:
                st.json(rec)

        # Export
        if st.button("📥 Export summary to Excel"):
            excel_path = "batch_summary.xlsx"
            df.to_excel(excel_path, index=False)
            with open(excel_path, "rb") as f:
                st.download_button("Download Excel", f, "batch_summary.xlsx")

    # Clean up temp directory if it exists
    if st.session_state.temp_dir:
        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        st.session_state.temp_dir = None

    if st.button("Process another batch"):
        st.session_state.phase = "idle"
        st.session_state.files = []
        st.session_state.idx = 0
        st.session_state.records = []
        st.session_state.log_messages = []
        st.rerun()
