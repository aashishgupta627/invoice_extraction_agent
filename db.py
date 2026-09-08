import sqlite3
import json
from pathlib import Path

DB_PATH = Path("extraction.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS invoices (
            source_file TEXT PRIMARY KEY,
            extraction_backend TEXT,
            seller TEXT,
            buyer TEXT,
            invoice_number TEXT,
            invoice_date TEXT,
            line_items TEXT,      -- JSON
            tax_breakup TEXT,     -- JSON
            total_amount REAL,
            extraction_confidence TEXT, -- JSON
            validation_flags TEXT,      -- JSON array
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_invoice(record: dict):
    """Save or update an invoice record."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Convert nested dicts to JSON strings
    row = (
        record.get("source_file"),
        record.get("extraction_backend"),
        json.dumps(record.get("seller", {})),
        json.dumps(record.get("buyer", {})),
        record.get("invoice_number"),
        record.get("invoice_date"),
        json.dumps(record.get("line_items", [])),
        json.dumps(record.get("tax_breakup", {})),
        record.get("total_amount", 0.0),
        json.dumps(record.get("extraction_confidence", {})),
        json.dumps(record.get("validation_flags", [])),
        record.get("status", "pending_review")
    )
    c.execute('''
        INSERT OR REPLACE INTO invoices (
            source_file, extraction_backend, seller, buyer, invoice_number,
            invoice_date, line_items, tax_breakup, total_amount,
            extraction_confidence, validation_flags, status
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    ''', row)
    conn.commit()
    conn.close()

def get_invoice(source_file: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM invoices WHERE source_file=?', (source_file,))
    row = c.fetchone()
    conn.close()
    if row:
        columns = [d[0] for d in c.description]
        record = dict(zip(columns, row))
        # Parse JSON fields
        for key in ["seller", "buyer", "line_items", "tax_breakup", "extraction_confidence", "validation_flags"]:
            if key in record and record[key]:
                record[key] = json.loads(record[key])
        return record
    return None