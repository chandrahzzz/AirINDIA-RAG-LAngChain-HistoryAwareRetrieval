"""Interested-list storage (SQLite).

Two states:
- `pending`  : a lead the bot captured, awaiting admin approval (HITL gate).
- `approved` : an admin approved it -> it's now on the official interested list.

The admin reviews `list_pending()` and calls `approve()` / `reject()`.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import config


@dataclass
class Lead:
    id: int
    name: str
    contact: str
    routes: str
    status: str
    created: float


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.LEADS_DB)
    c.row_factory = sqlite3.Row
    c.execute(
        """CREATE TABLE IF NOT EXISTS leads (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT NOT NULL,
               contact TEXT NOT NULL,
               routes TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'pending',
               session_id TEXT,
               created REAL NOT NULL
           )"""
    )
    return c


def add_pending(name: str, contact: str, routes: str, session_id: str = "") -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO leads(name, contact, routes, status, session_id, created) "
            "VALUES (?,?,?,'pending',?,?)",
            (name.strip(), contact.strip(), routes.strip(), session_id, time.time()),
        )
        return cur.lastrowid


def _rows(status: str | None = None) -> list[Lead]:
    with _conn() as c:
        if status:
            rows = c.execute("SELECT * FROM leads WHERE status=? ORDER BY id DESC", (status,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM leads ORDER BY id DESC").fetchall()
    return [Lead(r["id"], r["name"], r["contact"], r["routes"], r["status"], r["created"]) for r in rows]


def list_pending() -> list[Lead]:
    return _rows("pending")


def list_approved() -> list[Lead]:
    return _rows("approved")


def _set_status(lead_id: int, status: str) -> bool:
    with _conn() as c:
        cur = c.execute("UPDATE leads SET status=? WHERE id=? AND status='pending'", (status, lead_id))
        return cur.rowcount > 0


def approve(lead_id: int) -> bool:
    return _set_status(lead_id, "approved")


def reject(lead_id: int) -> bool:
    return _set_status(lead_id, "rejected")
