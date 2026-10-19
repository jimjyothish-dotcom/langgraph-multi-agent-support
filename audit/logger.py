"""
Module:
    logger.py
Description:
    Audit logging for the Galaxy Telecom multi-agent support system.
    Creates a local SQLite database and writes a structured log entry
    for every customer request processed by the agent graph — whether
    resolved by a worker agent or escalated to a human agent.
    The audit log provides business-level visibility into agent decisions,
    separate from LangFuse which handles LLM-level observability."""

import sqlite3
import json
from datetime import datetime

# Path to the local SQLite audit database file
DB_PATH = "audit/galaxy_telecom_audit.db"


def initialise_db():
    """Create the audit log table if it does not already exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            customer_message TEXT NOT NULL,
            intent          TEXT,
            worker_used     TEXT,
            worker_result   TEXT,
            final_response  TEXT,
            escalated       INTEGER NOT NULL DEFAULT 0,
            escalation_reason TEXT
        )
    """)
    conn.commit()
    conn.close()


def write_log(log_entry: dict):
    """
    Write a single audit log entry to the SQLite database.
    Called by the responder node and escalation handler at the end
    of every graph run before returning final state.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_log (
            timestamp,
            customer_message,
            intent,
            worker_used,
            worker_result,
            final_response,
            escalated,
            escalation_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        log_entry.get("customer_message", ""),
        log_entry.get("intent", ""),
        log_entry.get("worker_used", ""),
        log_entry.get("worker_result", ""),
        log_entry.get("final_response", ""),
        1 if log_entry.get("escalated") else 0,
        log_entry.get("escalation_reason", "")
    ))
    conn.commit()
    conn.close()


def read_logs():
    """
    Retrieve all audit log entries as a list of dictionaries.
    Useful for building an admin dashboard or running reports.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_log ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    # Map each row to a named dictionary for easy consumption
    columns = [
        "id", "timestamp", "customer_message", "intent",
        "worker_used", "worker_result", "final_response",
        "escalated", "escalation_reason"
    ]
    return [dict(zip(columns, row)) for row in rows]


# Initialise the database when this module is first imported
initialise_db()