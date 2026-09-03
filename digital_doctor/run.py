from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if __package__ in {None, ""}:  # allow running as `python digital_doctor/run.py`
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(repo_root))
    from digital_doctor.core.session import MilestoneSession, DigitalDoctorSession  # type: ignore
    from digital_doctor.core.session_store import reset_session_files  # type: ignore
    from digital_doctor.core.text_utils import extract_final  # type: ignore
    from digital_doctor.paths import (  # type: ignore
        DEFAULT_KNOWLEDGE_TREE_PATH,
        DEFAULT_ALERT_PATH,
        DEFAULT_LONG_TERM_MEMORY_PATH,
        DEFAULT_LOG_PATH,
        DEFAULT_MEMORY_PATH,
        DEFAULT_MILESTONE_PATH,
        resolve_repo_path,
        DEFAULT_STATE_PATH,
        DEFAULT_TRACE_PATH,
        DEFAULT_TRANSCRIPT_PATH,
    )
    from digital_doctor.tracking.milestones import (  # type: ignore
        Milestone,
        MilestoneState,
        MilestoneTracker,
        load_milestones,
        load_session_config,
    )
else:
    from .core.session import MilestoneSession, DigitalDoctorSession
    from .core.session_store import reset_session_files
    from .core.text_utils import extract_final
    from .paths import (
        DEFAULT_KNOWLEDGE_TREE_PATH,
        DEFAULT_ALERT_PATH,
        DEFAULT_LONG_TERM_MEMORY_PATH,
        DEFAULT_LOG_PATH,
        DEFAULT_MEMORY_PATH,
        DEFAULT_MILESTONE_PATH,
        resolve_repo_path,
        DEFAULT_STATE_PATH,
        DEFAULT_TRACE_PATH,
        DEFAULT_TRANSCRIPT_PATH,
    )
    from .tracking.milestones import Milestone, MilestoneState, MilestoneTracker, load_milestones, load_session_config


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive OCD support agent with phase planning and optional helper model.",
    )
    parser.add_argument(
        "--helper-api-url",
        type=str,
        default=os.getenv("HELPER_API_URL", "http://localhost:8001/helper/generate"),
        help="Helper API URL. Ignored when helper model is disabled.",
    )
    parser.add_argument(
        "--helper-api-key",
        type=str,
        default=os.getenv("HELPER_API_KEY"),
        help="Optional helper API key.",
    )
    parser.add_argument(
        "--transcript-path",
        type=str,
        default=os.getenv("TRANSCRIPT_PATH", DEFAULT_TRANSCRIPT_PATH),
        help="Reference transcript JSON path.",
    )
    parser.add_argument(
        "--milestone-path",
        type=str,
        default=os.getenv("MILESTONE_PATH", DEFAULT_MILESTONE_PATH),
        help="Milestone markdown path.",
    )
    parser.add_argument(
        "--knowledge-tree-path",
        type=str,
        default=os.getenv("KNOWLEDGE_TREE_PATH", DEFAULT_KNOWLEDGE_TREE_PATH),
        help="Knowledge tree JSON path.",
    )
    parser.add_argument(
        "--session-config-path",
        type=str,
        default=os.getenv("SESSION_CONFIG_PATH"),
        help="Optional session config JSON path.",
    )
    parser.add_argument("--user-id", type=str, default="user")
    parser.add_argument("--episode-id", type=str, default="session")
    parser.add_argument("--memory-path", type=str, default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--long-term-memory-path", type=str, default=DEFAULT_LONG_TERM_MEMORY_PATH)
    parser.add_argument("--state-path", type=str, default=DEFAULT_STATE_PATH)
    parser.add_argument("--log-path", type=str, default=DEFAULT_LOG_PATH)
    parser.add_argument("--trace-path", type=str, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--alert-path", type=str, default=DEFAULT_ALERT_PATH)
    parser.add_argument(
        "--memory-summary-threshold-chars",
        type=int,
        default=int(os.getenv("MEMORY_SUMMARY_THRESHOLD_CHARS", "12000")),
    )
    parser.add_argument(
        "--treatment-min-context-turns",
        type=int,
        default=int(os.getenv("TREATMENT_MIN_CONTEXT_TURNS", "3")),
    )
    parser.add_argument("--single-turn", action="store_true", help="Run without saving turn history.")

    helper_group = parser.add_mutually_exclusive_group()
    helper_group.add_argument(
        "--use-helper-model",
        dest="use_helper_model",
        action="store_true",
        help="Enable helper model calls.",
    )
    helper_group.add_argument(
        "--no-helper-model",
        dest="use_helper_model",
        action="store_false",
        help="Disable helper model calls.",
    )

    knowledge_group = parser.add_mutually_exclusive_group()
    knowledge_group.add_argument(
        "--use-knowledge-tree",
        dest="use_knowledge_tree",
        action="store_true",
        help="Enable knowledge tree retrieval.",
    )
    knowledge_group.add_argument(
        "--no-knowledge-tree",
        dest="use_knowledge_tree",
        action="store_false",
        help="Disable knowledge tree retrieval.",
    )

    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--reset-session-files",
        dest="reset_session",
        action="store_true",
        help="Clear memory/state/log/trace files before starting.",
    )
    reset_group.add_argument(
        "--no-reset-session-files",
        dest="reset_session",
        action="store_false",
        help="Keep existing memory/state/log/trace files.",
    )

    parser.set_defaults(
        use_helper_model=_env_flag("USE_HELPER_MODEL", True),
        use_knowledge_tree=_env_flag("USE_KNOWLEDGE_TREE", True),
        reset_session=_env_flag("RESET_SESSION_FILES", True),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    transcript_path = str(resolve_repo_path(args.transcript_path))
    milestone_path = str(resolve_repo_path(args.milestone_path))
    knowledge_tree_path = str(resolve_repo_path(args.knowledge_tree_path))
    session_config_path = (
        str(resolve_repo_path(args.session_config_path)) if args.session_config_path else None
    )
    memory_path = str(resolve_repo_path(args.memory_path))
    long_term_memory_path = str(resolve_repo_path(args.long_term_memory_path))
    state_path = str(resolve_repo_path(args.state_path))
    log_path = str(resolve_repo_path(args.log_path))
    trace_path = str(resolve_repo_path(args.trace_path))
    alert_path = str(resolve_repo_path(args.alert_path))

    if args.reset_session:
        reset_session_files(
            memory_path,
            long_term_memory_path,
            state_path,
            log_path,
            trace_path,
            alert_path,
        )

    session = DigitalDoctorSession(
        transcript_path=transcript_path,
        helper_api_url=args.helper_api_url,
        helper_api_key=args.helper_api_key,
        milestone_path=milestone_path,
        session_config_path=session_config_path,
        user_id=args.user_id,
        episode_id=args.episode_id,
        memory_path=memory_path,
        long_term_memory_path=long_term_memory_path,
        state_path=state_path,
        log_path=log_path,
        trace_path=trace_path,
        alert_path=alert_path,
        alert_webhook_url=os.getenv("CLINICAL_ALERT_WEBHOOK_URL"),
        memory_summary_threshold_chars=args.memory_summary_threshold_chars,
        treatment_min_context_turns=args.treatment_min_context_turns,
        single_turn=args.single_turn,
        use_helper_model=args.use_helper_model,
        knowledge_tree_path=knowledge_tree_path,
        use_knowledge_tree=args.use_knowledge_tree,
    )

    helper_status = "on" if args.use_helper_model else "off"
    knowledge_status = "on" if args.use_knowledge_tree else "off"
    print(
        "Interactive OCD support agent "
        f"[phase planning + helper {helper_status} + knowledge {knowledge_status}]. "
        "Type 'exit' or 'quit' to stop."
    )

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        reply, update = session.handle_query(query)
        source_candidates = update.get("source_candidates", {})
        if isinstance(source_candidates, dict):
            transcript_candidate = str(source_candidates.get("transcript", "")).strip()
            knowledge_candidate = str(source_candidates.get("knowledge_tree", "")).strip()
            if transcript_candidate:
                print(f"[transcript] {transcript_candidate}")
            if knowledge_candidate:
                print(f"[knowledge_tree] {knowledge_candidate}")
        print(reply)
        route = update.get("route", "analysis")
        milestone_health = update.get("milestone_health", {})
        planner_status = (
            milestone_health.get("status", "unknown")
            if isinstance(milestone_health, dict)
            else "unknown"
        )
        print(
            f"[{route}] coverage {update['covered_count']}/{update['total']} "
            f"| next: {update['next_target']} | planner: {planner_status}"
        )


__all__ = [
    "DEFAULT_KNOWLEDGE_TREE_PATH",
    "DEFAULT_ALERT_PATH",
    "DEFAULT_LONG_TERM_MEMORY_PATH",
    "DEFAULT_LOG_PATH",
    "DEFAULT_MEMORY_PATH",
    "DEFAULT_MILESTONE_PATH",
    "DEFAULT_STATE_PATH",
    "DEFAULT_TRACE_PATH",
    "DEFAULT_TRANSCRIPT_PATH",
    "Milestone",
    "MilestoneSession",
    "MilestoneState",
    "MilestoneTracker",
    "DigitalDoctorSession",
    "build_arg_parser",
    "extract_final",
    "load_milestones",
    "load_session_config",
    "main",
    "reset_session_files",
]


if __name__ == "__main__":
    main()
