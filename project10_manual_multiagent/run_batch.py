import argparse
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.insert(0, str(Path(__file__).parent.parent))

from project10_manual_multiagent.supervisor import run_pipeline
from shared.db import get_invoice, init_db, save_invoice
from shared.logging_config import get_logger, with_thread

load_dotenv()

logger = get_logger("run_batch")

# Fully done — never touch again.
TERMINAL_STATUSES = {"flagged", "approved", "rejected"}
# Extraction already succeeded — resume at validation, don't re-extract.
RESUMABLE_STATUSES = {"extracted"}
# Everything else (error_extraction, error, missing/None) gets a full retry.


def run_batch(folder_path: str):
    init_db()
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} not found.")
        return

    extensions = [".pdf", ".png", ".jpg", ".jpeg"]
    files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
    print(f"Found {len(files)} invoice files.")
    logger.info(f"Batch run starting over {len(files)} file(s) in {folder_path}")

    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    try:
        for file_path in files:
            key = str(file_path)
            tlog = with_thread(logger, key)
            existing = get_invoice(key)
            status = existing.get("status") if existing else None

            if status in TERMINAL_STATUSES:
                print(f"Skipping {file_path.name} - already processed (status: {status}).")
                tlog.info(f"Skipped: terminal status={status}")
                continue

            try:
                if status in RESUMABLE_STATUSES:
                    print(f"Resuming {file_path.name} from validation (status: {status})...")
                    tlog.info(f"Resuming from validation; existing status={status}")
                    record = run_pipeline(key, checkpointer, existing_record=existing)
                else:
                    print(f"Processing {file_path.name}...")
                    tlog.info(f"Running full pipeline; prior status={status}")
                    record = run_pipeline(key, checkpointer)

                flags = record.get("validation_flags", [])
                if flags:
                    print(f"  Flags: {', '.join(flags)}")
                else:
                    print("  No flags.")
                print(f"  Final status: {record.get('status')}")
            except KeyboardInterrupt:
                print(f"\nInterrupted while processing {file_path.name}. State saved - resume later.")
                tlog.warning("Interrupted by user (KeyboardInterrupt)")
                raise
            except Exception as e:
                print(f"  Error: {e}")
                tlog.error("Unhandled error while processing file", exc_info=True)
                save_invoice({
                    "source_file": key,
                    "status": "error",
                    "validation_flags": [f"Processing error: {str(e)}"],
                })
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to folder containing invoice files")
    args = parser.parse_args()
    run_batch(args.folder)
