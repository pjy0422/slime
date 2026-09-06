"""FastAPI application for the local DTAP trajectory explorer."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .bundle import load_episode_bundle
from .db import TrajectoryDB
from .indexer import index_root

_WEB = Path(__file__).resolve().parent / "web"


def create_app(
    root: str | Path,
    *,
    db_path: str | Path | None = None,
    watch: bool = False,
    refresh_seconds: float = 2.0,
) -> FastAPI:
    artifact_root = Path(root).expanduser().resolve()
    db_file = Path(db_path).expanduser().resolve() if db_path else artifact_root / ".dtap-traj.sqlite3"
    db = TrajectoryDB(db_file)
    index_root(artifact_root, db)
    stop = threading.Event()

    def watcher() -> None:
        while not stop.wait(max(0.5, refresh_seconds)):
            index_root(artifact_root, db)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        thread = None
        if watch:
            thread = threading.Thread(target=watcher, name="dtap-traj-indexer", daemon=True)
            thread.start()
        yield
        stop.set()
        if thread:
            thread.join(timeout=2)

    app = FastAPI(title="DTAP Trajectory Explorer", version="0.3.0", lifespan=lifespan)
    app.state.root = artifact_root
    app.state.db = db

    @app.get("/api/health")
    def health():
        return {"ok": True, "root": str(artifact_root), "db": str(db_file), "watch": watch}

    @app.post("/api/index")
    def refresh_index():
        return index_root(artifact_root, db)

    @app.get("/api/facets")
    def facets():
        return db.facets()

    @app.get("/api/episodes")
    def episodes(
        run_name: str | None = None,
        domain: str | None = None,
        threat_model: str | None = None,
        status: str | None = None,
        attack_success: bool | None = None,
        attack_evaluated: bool | None = None,
        q: str | None = None,
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        return db.list_episodes(
            run_name=run_name,
            domain=domain,
            threat_model=threat_model,
            status=status,
            attack_success=attack_success,
            attack_evaluated=attack_evaluated,
            q=q,
            limit=limit,
            offset=offset,
        )

    def episode_or_404(episode_id: str) -> dict:
        item = db.get_episode(episode_id)
        if item is None:
            raise HTTPException(404, "episode not found")
        return item

    def artifact_dir_or_404(item: dict) -> Path:
        path = Path(item["artifact_path"]).resolve()
        try:
            path.relative_to(artifact_root)
        except ValueError:
            # The database is an index, not authority to read arbitrary files.
            raise HTTPException(404, "episode artifact not found") from None
        if not path.is_dir():
            raise HTTPException(404, "episode artifact not found")
        return path

    @app.get("/api/episodes/{episode_id}")
    def episode(episode_id: str):
        return episode_or_404(episode_id)

    @app.get("/api/episodes/{episode_id}/trajectory")
    def trajectory(
        episode_id: str,
        view: Literal["policy", "victim", "combined"] = "combined",
        attempt: int | None = Query(None, ge=1),
    ):
        item = episode_or_404(episode_id)
        try:
            data = load_episode_bundle(artifact_dir_or_404(item), attempt_index=attempt)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        payload = {
            "episode": item,
            "evaluation": data.get("evaluation") or {},
            "payloads": data.get("payloads") or [],
            "warnings": data.get("trajectory_warnings") or [],
            "attempt_index": data.get("attempt_index"),
            "attempts": data.get("attempts") or [],
        }
        if view in {"policy", "combined"}:
            payload["policy"] = data.get("policy_timeline") or []
        if view in {"victim", "combined"}:
            payload["victim"] = data.get("timeline") or []
        return payload

    @app.get("/api/episodes/{episode_id}/config")
    def config(episode_id: str, attempt: int | None = Query(None, ge=1)):
        item = episode_or_404(episode_id)
        try:
            data = load_episode_bundle(artifact_dir_or_404(item), attempt_index=attempt)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        return {"episode": item, "comparison": data.get("config_comparison")}

    @app.get("/api/episodes/{episode_id}/judges")
    def judges(episode_id: str, attempt: int | None = Query(None, ge=1)):
        item = episode_or_404(episode_id)
        try:
            data = load_episode_bundle(artifact_dir_or_404(item), attempt_index=attempt)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        return {"episode": item, "judges": data.get("judges")}

    if _WEB.is_dir():
        app.mount("/assets", StaticFiles(directory=_WEB), name="assets")

    @app.get("/")
    def home():
        return FileResponse(_WEB / "index.html")

    return app
