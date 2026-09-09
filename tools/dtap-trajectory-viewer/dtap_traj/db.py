"""SQLite index used by the local trajectory explorer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    run_name TEXT NOT NULL,
    task_id TEXT,
    dataset_path TEXT,
    policy_model TEXT,
    victim_model TEXT,
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
    policy_usage_json TEXT,
    victim_usage_json TEXT,
    attempt_count INTEGER,
    h1_attack_success INTEGER,
    h2_attack_success INTEGER,
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

_OPTIONAL_COLUMNS = {
    "task_id": "TEXT",
    "dataset_path": "TEXT",
    "policy_model": "TEXT",
    "victim_model": "TEXT",
    "policy_usage_json": "TEXT",
    "victim_usage_json": "TEXT",
    "attempt_count": "INTEGER",
    "h1_attack_success": "INTEGER",
    "h2_attack_success": "INTEGER",
}


def _bool_db(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in (
        "attack_success", "evaluation_completed", "placement_applicable",
        "placement_covered", "h1_attack_success", "h2_attack_success",
    ):
        if item.get(key) is not None:
            item[key] = bool(item[key])
    for column, key in (
        ("policy_usage_json", "policy_usage"),
        ("victim_usage_json", "victim_usage"),
    ):
        raw = item.pop(column, None)
        try:
            item[key] = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            item[key] = None
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
            existing = {
                row["name"] for row in conn.execute("PRAGMA table_info(episodes)")
            }
            for column, sql_type in _OPTIONAL_COLUMNS.items():
                if column not in existing:
                    conn.execute(
                        f"ALTER TABLE episodes ADD COLUMN {column} {sql_type}"
                    )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_id)"
            )

    def upsert_episode(self, item: dict[str, Any]) -> None:
        values = {
            **item,
            "attack_success": _bool_db(item.get("attack_success")),
            "evaluation_completed": _bool_db(item.get("evaluation_completed")),
            "placement_applicable": _bool_db(item.get("placement_applicable")),
            "placement_covered": _bool_db(item.get("placement_covered")),
            "h1_attack_success": _bool_db(item.get("h1_attack_success")),
            "h2_attack_success": _bool_db(item.get("h2_attack_success")),
            "policy_usage_json": (
                json.dumps(item["policy_usage"], sort_keys=True)
                if item.get("policy_usage") else None
            ),
            "victim_usage_json": (
                json.dumps(item["victim_usage"], sort_keys=True)
                if item.get("victim_usage") else None
            ),
        }
        columns = [
            "episode_id",
            "run_name",
            "task_id",
            "dataset_path",
            "policy_model",
            "victim_model",
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
            "policy_usage_json",
            "victim_usage_json",
            "attempt_count",
            "h1_attack_success",
            "h2_attack_success",
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
        attack_evaluated: bool | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        cohort_clauses: list[str] = []
        cohort_params: list[Any] = []
        for column, value in (("run_name", run_name), ("domain", domain), ("threat_model", threat_model), ("status", status)):
            if value:
                cohort_clauses.append(f"{column} = ?")
                cohort_params.append(value)
        if q:
            cohort_clauses.append(
                "(episode_id LIKE ? OR task_id LIKE ? OR dataset_path LIKE ? "
                "OR policy_model LIKE ? OR victim_model LIKE ? OR risk_category LIKE ? "
                "OR artifact_path LIKE ?)"
            )
            needle = f"%{q}%"
            cohort_params.extend([needle] * 7)
        clauses = list(cohort_clauses)
        params = list(cohort_params)
        if attack_success is not None:
            clauses.append("attack_success = ?")
            params.append(1 if attack_success else 0)
        if attack_evaluated is not None:
            clauses.append("attack_success IS NOT NULL" if attack_evaluated else "attack_success IS NULL")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cohort_where = (
            f" WHERE {' AND '.join(cohort_clauses)}" if cohort_clauses else ""
        )
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        with self.connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM episodes{where}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM episodes{where} ORDER BY indexed_at DESC, domain, threat_model, episode_id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            asr_row = conn.execute(
                "SELECT "
                "SUM(h1_attack_success = 1) AS h1_successes, "
                "SUM(h1_attack_success IS NOT NULL) AS h1_evaluated, "
                "SUM(h2_attack_success = 1) AS h2_successes, "
                "SUM(h2_attack_success IS NOT NULL) AS h2_evaluated, "
                "SUM(h1_attack_success = 1 OR h2_attack_success = 1) AS cumulative_successes, "
                "SUM(h1_attack_success IS NOT NULL OR h2_attack_success IS NOT NULL) AS cumulative_evaluated "
                f"FROM episodes{cohort_where}",
                cohort_params,
            ).fetchone()

        def rate(successes: Any, evaluated: Any) -> dict[str, Any]:
            successes = int(successes or 0)
            evaluated = int(evaluated or 0)
            return {
                "successes": successes,
                "evaluated": evaluated,
                "rate": successes / evaluated if evaluated else None,
            }

        return {
            "total": total,
            "items": [_row(r) for r in rows],
            "limit": limit,
            "offset": offset,
            "asr": {
                "h1": rate(asr_row["h1_successes"], asr_row["h1_evaluated"]),
                "h2": rate(asr_row["h2_successes"], asr_row["h2_evaluated"]),
                "cumulative": rate(
                    asr_row["cumulative_successes"],
                    asr_row["cumulative_evaluated"],
                ),
            },
        }

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
