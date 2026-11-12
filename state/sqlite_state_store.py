"""SQLite durable checkpoints for in-process state managers.

Connections are explicitly closed after every operation. SQLite connection
context managers commit/rollback transactions but do NOT close connections.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class SQLiteStateStore:
    def __init__(self, path):
        self.path = str(Path(path).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS manager_checkpoints "
                "(manager TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    def load(self, name):
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM manager_checkpoints WHERE manager=?", (name,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, name, payload):
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._connect() as db:
            db.execute(
                "INSERT INTO manager_checkpoints(manager,payload) VALUES (?,?) "
                "ON CONFLICT(manager) DO UPDATE SET payload=excluded.payload",
                (name, serialized),
            )
