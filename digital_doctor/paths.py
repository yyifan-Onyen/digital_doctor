from __future__ import annotations

from pathlib import Path
from typing import Dict


REPO_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path("data")
KNOWLEDGE_SOURCE_DIR = DATA_DIR / "knowledge_sources"
RUNTIME_DIR = Path("runtime")
LOG_DIR = RUNTIME_DIR / "logs"
INTERACTIVE_LOG_DIR = LOG_DIR / "interactive"
DEMO_LOG_DIR = LOG_DIR / "demo_dialogue"
CACHE_DIR = RUNTIME_DIR / "cache"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

DEFAULT_TRANSCRIPT_PATH = str(TRANSCRIPT_DIR / "101KI_deid.json")
DEFAULT_MILESTONE_PATH = str(DATA_DIR / "milestones.md")
DEFAULT_KNOWLEDGE_TREE_PATH = str(DATA_DIR / "knowledge_trees" / "wilhelm_steketee_2006.tree.json")
DEFAULT_SEGMENT_CACHE_PATH = str(CACHE_DIR / "segment_summaries.jsonl")


def resolve_repo_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_DIR / candidate).resolve()


def repo_relative_path(path: Path | str | None) -> str:
    if path is None:
        return ""
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(REPO_DIR).as_posix()
    except ValueError:
        return str(candidate)


def build_session_paths(base_dir: Path | str, prefix: str = "milestone") -> Dict[str, str]:
    root = Path(base_dir)
    return {
        "memory_path": str(root / f"{prefix}_memory.jsonl"),
        "long_term_memory_path": str(root / f"{prefix}_long_term_memory.json"),
        "state_path": str(root / f"{prefix}_state.jsonl"),
        "log_path": str(root / f"{prefix}_debug.log"),
        "trace_path": str(root / f"{prefix}_trace.jsonl"),
        "alert_path": str(root / f"{prefix}_clinical_alerts.jsonl"),
    }


_interactive_paths = build_session_paths(INTERACTIVE_LOG_DIR)
DEFAULT_MEMORY_PATH = _interactive_paths["memory_path"]
DEFAULT_LONG_TERM_MEMORY_PATH = _interactive_paths["long_term_memory_path"]
DEFAULT_STATE_PATH = _interactive_paths["state_path"]
DEFAULT_LOG_PATH = _interactive_paths["log_path"]
DEFAULT_TRACE_PATH = _interactive_paths["trace_path"]
DEFAULT_ALERT_PATH = _interactive_paths["alert_path"]
