"""SQLite index used by the local trajectory explorer."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    run_name TEXT NOT NULL,
    domain TEXT,
    threat_model TEXT,
    status TEXT,
    episode_status TEXT,
    risk_category TEXT,
    attack_success INTEGER,
    evaluation_completed INTEGER,
    placement_applicable INTEGER,
    placement_covered INTEGER,
    placement_actions INTEGER,
    placements_verified INTEGER,
    environment_steps INTEGER,
    policy_events INTEGER,
    victim_events INTEGER,
    artifact_path TEXT NOT NULL,
    source_mtime_ns INTEGER NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_name);
CREATE INDEX IF NOT EXISTS idx_episodes_domain ON episodes(domain);
CREATE INDEX IF NOT EXISTS idx_episodes_threat ON episodes(threat_model);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_episodes_attack ON episodes(attack_success);
"""


def _bool_db(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in ("attack_success", "evaluation_completed", "placement_applicable", "placement_covered"):
        if item.get(key) is not None:
            item[key] = bool(item[key])
    return item


class TrajectoryDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def upsert_episode(self, item: dict[str, Any]) -> None:
        values = {
            **item,
            "attack_success": _bool_db(item.get("attack_success")),
            "evaluation_completed": _bool_db(item.get("evaluation_completed")),
            "placement_applicable": _bool_db(item.get("placement_applicable")),
            "placement_covered": _bool_db(item.get("placement_covered")),
        }
        columns = [
            "episode_id",
            "run_name",
            "domain",
            "threat_model",
            "status",
            "episode_status",
            "risk_category",
            "attack_success",
            "evaluation_completed",
            "placement_applicable",
            "placement_covered",
            "placement_actions",
            "placements_verified",
            "environment_steps",
            "policy_events",
            "victim_events",
            "artifact_path",
            "source_mtime_ns",
        ]
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(f"{c}=excluded.{c}" for c in columns if c != "episode_id")
        sql = f"INSERT INTO episodes ({', '.join(columns)}) VALUES ({placeholders}) ON CONFLICT(episode_id) DO UPDATE SET {updates}, indexed_at=CURRENT_TIMESTAMP"
        with self.connect() as conn:
            conn.execute(sql, [values.get(c) for c in columns])

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
        return _row(row) if row else None

    def list_episodes(
        self,
        *,
        run_name: str | None = None,
        domain: str | None = None,
        threat_model: str | None = None,
        status: str | None = None,
        attack_success: bool | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("run_name", run_name), ("domain", domain), ("threat_model", threat_model), ("status", status)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if attack_success is not None:
            clauses.append("attack_success = ?")
            params.append(1 if attack_success else 0)
        if q:
            clauses.append("(episode_id LIKE ? OR risk_category LIKE ? OR artifact_path LIKE ?)")
            needle = f"%{q}%"
            params.extend([needle, needle, needle])
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        with self.connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM episodes{where}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM episodes{where} ORDER BY indexed_at DESC, domain, threat_model, episode_id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return {"total": total, "items": [_row(r) for r in rows], "limit": limit, "offset": offset}

    def facets(self) -> dict[str, Any]:
        def counts(conn: sqlite3.Connection, column: str) -> list[dict[str, Any]]:
            rows = conn.execute(f"SELECT {column} AS value, COUNT(*) AS count FROM episodes WHERE {column} IS NOT NULL GROUP BY {column} ORDER BY {column}").fetchall()
            return [{"value": row["value"], "count": row["count"]} for row in rows]

        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            attack = conn.execute("SELECT COALESCE(SUM(attack_success), 0) FROM episodes").fetchone()[0]
            return {
                "total": total,
                "attack_successes": attack,
                "runs": counts(conn, "run_name"),
                "domains": counts(conn, "domain"),
                "threat_models": counts(conn, "threat_model"),
                "statuses": counts(conn, "status"),
            }
