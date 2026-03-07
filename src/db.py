"""SQLite data access for auth and prediction history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .auth import expiry_iso, generate_token, hash_password, is_expired, verify_password
from .config import DB_PATH


@contextmanager
def get_conn(db_path: str = DB_PATH) -> Iterator[sqlite3.Connection]:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                input_json TEXT NOT NULL,
                prediction_label TEXT NOT NULL,
                probability REAL NOT NULL,
                latency_ms REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )


def create_user(username: str, password: str, db_path: str = DB_PATH) -> bool:
    with get_conn(db_path) as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, hash_password(password)),
            )
        except sqlite3.IntegrityError:
            return False
    return True


def authenticate_user(username: str, password: str, db_path: str = DB_PATH) -> int | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        return None

    if not verify_password(password, row["password_hash"]):
        return None

    return int(row["id"])


def create_session(user_id: int, db_path: str = DB_PATH) -> str:
    token = generate_token()
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expiry_iso(hours=24)),
        )
    return token


def resolve_session(token: str, db_path: str = DB_PATH) -> dict[str, Any] | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT s.token, s.user_id, s.expires_at, u.username
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()

        if row is None:
            return None

        expires_at = row["expires_at"]
        if is_expired(expires_at):
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None

        return {
            "user_id": int(row["user_id"]),
            "username": str(row["username"]),
            "token": str(row["token"]),
            "expires_at": str(expires_at),
        }


def store_prediction(
    user_id: int,
    input_payload: dict[str, Any],
    prediction_label: str,
    probability: float,
    latency_ms: float,
    db_path: str = DB_PATH,
) -> int:
    with get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (user_id, input_json, prediction_label, probability, latency_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, json.dumps(input_payload), prediction_label, float(probability), float(latency_ms)),
        )
        return int(cursor.lastrowid)


def get_prediction_history(user_id: int, limit: int = 20, db_path: str = DB_PATH) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, input_json, prediction_label, probability, latency_ms, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    history: list[dict[str, Any]] = []
    for row in rows:
        history.append(
            {
                "id": int(row["id"]),
                "input": json.loads(row["input_json"]),
                "prediction_label": str(row["prediction_label"]),
                "probability": float(row["probability"]),
                "latency_ms": float(row["latency_ms"]),
                "created_at": str(row["created_at"]),
            }
        )
    return history
