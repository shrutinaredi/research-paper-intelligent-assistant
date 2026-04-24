"""SQLite telemetry — logs every run and user feedback to a local DB.

Two tables:
  runs     — one row per question. Holds query, retrieved chunks, answer,
             critic verdict, latency.
  feedback — thumbs up / thumbs down from the UI, linked to runs by run_id.

Used by:
  - the Streamlit UI (to persist feedback)
  - eval/run_eval.py (to diff production vs eval behavior)
  - manual triage (sqlite3 research_assistant.db)
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("research_assistant.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    ts              DATETIME DEFAULT CURRENT_TIMESTAMP,
    question        TEXT NOT NULL,
    question_type   TEXT,
    retrieval_query TEXT,
    top_k           INTEGER,
    attempts        INTEGER,
    chunks_json     TEXT,
    answer          TEXT,
    critic_json     TEXT,
    latency_ms      INTEGER
);

CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    ts        DATETIME DEFAULT CURRENT_TIMESTAMP,
    rating    TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment   TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_run(
    *,
    run_id: str,
    question: str,
    question_type: str | None,
    retrieval_query: str | None,
    top_k: int | None,
    attempts: int | None,
    chunks: list[dict],
    answer: str,
    critic: dict,
    latency_ms: int,
) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO runs (run_id, question, question_type, retrieval_query,
                                 top_k, attempts, chunks_json, answer, critic_json,
                                 latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                question,
                question_type,
                retrieval_query,
                top_k,
                attempts,
                json.dumps(chunks),
                answer,
                json.dumps(critic),
                latency_ms,
            ),
        )


def log_feedback(run_id: str, rating: str, comment: str | None = None) -> None:
    assert rating in ("up", "down")
    with _conn() as conn:
        conn.execute(
            "INSERT INTO feedback (run_id, rating, comment) VALUES (?, ?, ?)",
            (run_id, rating, comment),
        )
