"""SQLite persistence: audit log (WorkflowTraceRecord), feedback (FeedbackSignalRecord), prompt versions, user profiles, batches."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from typing import Any

from .llm.prompts import DEFAULT_PROMPTS
from .schemas import FeedbackSignalRecord, UserProfile, WorkflowTraceRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_trace_records (
  trace_id TEXT PRIMARY KEY, created_at TEXT, user_id TEXT, domain TEXT, intent TEXT, level TEXT,
  model_variant TEXT, prompt_version TEXT, llm TEXT, status TEXT, verdict TEXT, attempts INTEGER, record TEXT);
CREATE INDEX IF NOT EXISTS idx_trace_created ON workflow_trace_records(created_at);
CREATE TABLE IF NOT EXISTS feedback_signal_records (
  feedback_id TEXT PRIMARY KEY, trace_id TEXT, created_at TEXT, user_id TEXT, rating INTEGER, record TEXT);
CREATE TABLE IF NOT EXISTS prompt_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, version TEXT UNIQUE, template TEXT, guidance TEXT,
  status TEXT, created_at TEXT, approved_at TEXT, approved_by TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, expertise TEXT, preferences TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS batches (batch_id TEXT PRIMARY KEY, created_at TEXT, domain TEXT, rows TEXT, results TEXT);
"""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        if not self.prompts("generator"):
            for p in DEFAULT_PROMPTS:
                self._insert_prompt(p["name"], p["version"], p["template"], p["guidance"], "active", "seed prompt")

    def _exec(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self.conn.execute(sql, args)
            rows = cur.fetchall()
            self.conn.commit()
            return rows

    # ---------------------------------------------------------------- audit log
    def save_trace(self, r: WorkflowTraceRecord) -> None:
        self._exec(
            "INSERT OR REPLACE INTO workflow_trace_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r.trace_id, r.created_at, r.user_id, r.domain, r.intent, r.level, r.model_variant, r.prompt_version,
             r.llm, r.status, r.verdict, r.attempts, r.model_dump_json()),
        )

    def get_trace(self, trace_id: str) -> WorkflowTraceRecord | None:
        rows = self._exec("SELECT record FROM workflow_trace_records WHERE trace_id=?", (trace_id,))
        return WorkflowTraceRecord.model_validate_json(rows[0]["record"]) if rows else None

    def traces(self, limit: int = 50) -> list[dict]:
        rows = self._exec(
            "SELECT trace_id, created_at, user_id, domain, intent, level, model_variant, prompt_version, llm, status, "
            "verdict, attempts FROM workflow_trace_records ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def trace_records(self, limit: int = 1000) -> list[WorkflowTraceRecord]:
        rows = self._exec("SELECT record FROM workflow_trace_records ORDER BY created_at DESC LIMIT ?", (limit,))
        return [WorkflowTraceRecord.model_validate_json(r["record"]) for r in rows]

    # ---------------------------------------------------------------- feedback
    def save_feedback(self, f: FeedbackSignalRecord) -> None:
        self._exec("INSERT INTO feedback_signal_records VALUES (?,?,?,?,?,?)",
                   (f.feedback_id, f.trace_id, f.created_at, f.user_id, f.rating, f.model_dump_json()))

    def feedback(self, limit: int = 5000) -> list[FeedbackSignalRecord]:
        rows = self._exec("SELECT record FROM feedback_signal_records ORDER BY created_at DESC LIMIT ?", (limit,))
        return [FeedbackSignalRecord.model_validate_json(r["record"]) for r in rows]

    # ---------------------------------------------------------------- prompts
    def _insert_prompt(self, name, version, template, guidance, status, notes) -> None:
        self._exec("INSERT INTO prompt_versions (name, version, template, guidance, status, created_at, notes) "
                   "VALUES (?,?,?,?,?,?,?)", (name, version, template, json.dumps(guidance), status, now(), notes))

    def prompts(self, name: str) -> list[dict]:
        rows = self._exec("SELECT * FROM prompt_versions WHERE name=? ORDER BY id", (name,))
        return [dict(r) | {"guidance": json.loads(r["guidance"] or "[]")} for r in rows]

    def active_prompt(self, name: str) -> dict:
        rows = [p for p in self.prompts(name) if p["status"] == "active"]
        return rows[-1] if rows else {}

    def add_candidate(self, name: str, template: str, guidance: list[str], notes: str) -> dict:
        n = len(self.prompts(name)) + 1
        version = f"gen-v{n}"
        self._insert_prompt(name, version, template, guidance, "candidate", notes)
        return next(p for p in self.prompts(name) if p["version"] == version)

    def approve(self, prompt_id: int, by: str) -> dict:
        rows = self._exec("SELECT * FROM prompt_versions WHERE id=?", (prompt_id,))
        if not rows or rows[0]["status"] != "candidate":
            raise ValueError("Only a candidate prompt can be approved.")
        name = rows[0]["name"]
        with self._lock:
            self.conn.execute("UPDATE prompt_versions SET status='retired' WHERE name=? AND status='active'", (name,))
            self.conn.execute("UPDATE prompt_versions SET status='active', approved_at=?, approved_by=? WHERE id=?",
                              (now(), by, prompt_id))
            self.conn.commit()
        return self.active_prompt(name)

    # ---------------------------------------------------------------- users
    def profile(self, user_id: str) -> UserProfile:
        rows = self._exec("SELECT * FROM users WHERE user_id=?", (user_id,))
        if rows:
            return UserProfile(user_id=user_id, expertise=rows[0]["expertise"], preferences=json.loads(rows[0]["preferences"]))
        return UserProfile(user_id=user_id)

    def save_profile(self, p: UserProfile) -> None:
        self._exec("INSERT OR REPLACE INTO users VALUES (?,?,?,?)", (p.user_id, p.expertise, json.dumps(p.preferences), now()))

    # ---------------------------------------------------------------- batches
    def save_batch(self, batch_id: str, domain: str, rows: list[dict], results: list[dict]) -> None:
        self._exec("INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?)",
                   (batch_id, now(), domain, json.dumps(rows), json.dumps(results)))

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        rows = self._exec("SELECT * FROM batches WHERE batch_id=?", (batch_id,))
        if not rows:
            return None
        r = rows[0]
        return {"batch_id": r["batch_id"], "domain": r["domain"], "rows": json.loads(r["rows"]), "results": json.loads(r["results"])}
