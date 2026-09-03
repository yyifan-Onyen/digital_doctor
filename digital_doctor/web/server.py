"""FastAPI web server for the OCD ERP support agent.

Wraps a single ``DigitalDoctorSession`` and exposes it to a single-page frontend:

- ``GET  /``           serves the chat page (static/index.html)
- ``GET  /api/state``  returns the current phase / formulation snapshot
- ``POST /api/chat``   runs one turn and returns ``{reply, snapshot, update}``
- ``POST /api/reset``  starts a fresh session and returns the empty snapshot

The core tracking / session logic is untouched; this module only reads
``session.handle_query`` and ``session.tracker.snapshot``.

Run with::

    uvicorn digital_doctor.web.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..core.session import DigitalDoctorSession
from ..core.session_store import reset_session_files
from ..paths import (
    DEFAULT_KNOWLEDGE_TREE_PATH,
    DEFAULT_ALERT_PATH,
    DEFAULT_LONG_TERM_MEMORY_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_MEMORY_PATH,
    DEFAULT_MILESTONE_PATH,
    DEFAULT_STATE_PATH,
    DEFAULT_TRACE_PATH,
    DEFAULT_TRANSCRIPT_PATH,
    resolve_repo_path,
)

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _resolved(path: Optional[str]) -> Optional[str]:
    return str(resolve_repo_path(path)) if path else None


class SessionManager:
    """Owns one live session and rebuilds it on reset.

    A lock serializes turns so concurrent requests cannot corrupt the shared
    tracker / memory state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: Optional[DigitalDoctorSession] = None

    def _build(self) -> DigitalDoctorSession:
        use_helper = _env_flag("USE_HELPER_MODEL", True)
        use_knowledge = _env_flag("USE_KNOWLEDGE_TREE", True)

        memory_path = _resolved(DEFAULT_MEMORY_PATH)
        long_term_memory_path = _resolved(DEFAULT_LONG_TERM_MEMORY_PATH)
        state_path = _resolved(DEFAULT_STATE_PATH)
        log_path = _resolved(DEFAULT_LOG_PATH)
        trace_path = _resolved(DEFAULT_TRACE_PATH)
        alert_path = _resolved(DEFAULT_ALERT_PATH)

        if _env_flag("RESET_SESSION_FILES", True):
            reset_session_files(
                memory_path,
                long_term_memory_path,
                state_path,
                log_path,
                trace_path,
                alert_path,
            )

        return DigitalDoctorSession(
            transcript_path=_resolved(DEFAULT_TRANSCRIPT_PATH),
            milestone_path=_resolved(DEFAULT_MILESTONE_PATH),
            helper_api_url=os.getenv("HELPER_API_URL", "http://localhost:8001/helper/generate"),
            helper_api_key=os.getenv("HELPER_API_KEY"),
            session_config_path=_resolved(os.getenv("SESSION_CONFIG_PATH")),
            memory_path=memory_path,
            long_term_memory_path=long_term_memory_path,
            state_path=state_path,
            log_path=log_path,
            trace_path=trace_path,
            alert_path=alert_path,
            alert_webhook_url=os.getenv("CLINICAL_ALERT_WEBHOOK_URL"),
            memory_summary_threshold_chars=int(
                os.getenv("MEMORY_SUMMARY_THRESHOLD_CHARS", "12000")
            ),
            treatment_min_context_turns=int(
                os.getenv("TREATMENT_MIN_CONTEXT_TURNS", "3")
            ),
            use_helper_model=use_helper,
            knowledge_tree_path=_resolved(DEFAULT_KNOWLEDGE_TREE_PATH),
            use_knowledge_tree=use_knowledge,
        )

    def session(self) -> DigitalDoctorSession:
        with self._lock:
            if self._session is None:
                self._session = self._build()
            return self._session

    def reset(self) -> Dict[str, object]:
        with self._lock:
            self._session = self._build()
            return self._session.snapshot()

    def snapshot(self) -> Dict[str, object]:
        return self.session().snapshot()

    def chat(self, message: str) -> Dict[str, object]:
        # Hold the lock across the whole turn: handle_query mutates the tracker
        # and memory, so overlapping turns would interleave state writes.
        with self._lock:
            if self._session is None:
                self._session = self._build()
            reply, update = self._session.handle_query(message)
            snapshot = self._session.snapshot()
        return {"reply": reply, "snapshot": snapshot, "update": update}


manager = SessionManager()

app = FastAPI(title="Digital Doctor — OCD ERP", docs_url=None, redoc_url=None)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def index() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="Frontend index.html is missing.")
    return FileResponse(INDEX_FILE)


@app.get("/api/state")
def get_state() -> Dict[str, object]:
    return {"snapshot": manager.snapshot()}


@app.post("/api/chat")
def post_chat(request: ChatRequest) -> Dict[str, object]:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return manager.chat(message)


@app.post("/api/reset")
def post_reset() -> Dict[str, object]:
    return {"snapshot": manager.reset()}
