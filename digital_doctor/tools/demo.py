from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

load_dotenv()

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from digital_doctor.core.session import DigitalDoctorSession  # type: ignore
    from digital_doctor.paths import (  # type: ignore
        DEFAULT_KNOWLEDGE_TREE_PATH,
        DEFAULT_MILESTONE_PATH,
        DEFAULT_TRANSCRIPT_PATH,
        DEMO_LOG_DIR,
        build_session_paths,
        repo_relative_path,
        resolve_repo_path,
    )
    from digital_doctor.core.text_utils import extract_final  # type: ignore
    from digital_doctor.services.openai_client import call_model  # type: ignore
else:
    from ..core.session import DigitalDoctorSession
    from ..core.text_utils import extract_final
    from ..paths import (
        DEFAULT_KNOWLEDGE_TREE_PATH,
        DEFAULT_MILESTONE_PATH,
        DEFAULT_TRANSCRIPT_PATH,
        DEMO_LOG_DIR,
        build_session_paths,
        repo_relative_path,
        resolve_repo_path,
    )
    from ..services.openai_client import call_model


PATIENT_STAGE_HINTS = [
    "Start with contamination fear and repeated handwashing after touching public objects.",
    "Say the relief after washing is brief and the urge returns quickly.",
    "Ask whether ERP is supposed to make anxiety disappear before life feels safe again.",
    "Report a small exposure attempt, but admit you still washed soon after.",
    "Describe one everyday contamination trigger you could practice with again.",
    "Name the feared consequence you predict if you resist the ritual.",
    "Mention one time you delayed the ritual longer than usual and what happened.",
    "Reflect on whether the fear still feels emotionally true or more like an OCD alarm.",
    "Bring up uncertainty and wanting complete proof that nothing bad will happen.",
    "Ask how to keep practicing alone and avoid slipping back into rituals.",
]


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class GptPatientSimulator:
    def __init__(self, name: str = "Patient"):
        self.name = name

    def generate(
        self,
        turn_index: int,
        history: List[Dict[str, str]],
        last_update: Optional[Dict[str, object]],
    ) -> str:
        stage_hint = PATIENT_STAGE_HINTS[min(turn_index, len(PATIENT_STAGE_HINTS) - 1)]
        recent = history[-6:]
        recent_block = "\n".join(f"{item['role']}: {item['text']}" for item in recent) if recent else "(none)"
        next_target = str(last_update.get("next_target", "")) if last_update else ""
        coverage = (
            f"{last_update.get('covered_count', 0)}/{last_update.get('total', 0)}" if last_update else "0/0"
        )
        status_changes = last_update.get("status_changes", []) if last_update else []
        last_artifacts = last_update.get("artifacts", {}) if last_update else {}
        transcript_refs = last_artifacts.get("transcript_refs", []) if isinstance(last_artifacts, dict) else []
        knowledge_hits = last_artifacts.get("knowledge_hits", []) if isinstance(last_artifacts, dict) else []

        prompt = f"""
You are simulating the PATIENT side of an ERP therapy session.
Write exactly one patient message for the next turn.

Hard constraints:
- 1-2 sentences only.
- First person only.
- Natural spoken English.
- Do not roleplay the doctor.
- Do not write lists.
- Keep continuity with the previous turns.
- Stay clinically plausible for OCD with contamination / checking themes.

Patient name:
{self.name}

Current turn goal:
{stage_hint}

Milestone tracker status from the doctor's side:
- coverage: {coverage}
- next target: {next_target}
- recent status changes: {json.dumps(status_changes, ensure_ascii=False)}

Recent retrieval hints the doctor used last turn:
- transcript refs used: {json.dumps(transcript_refs, ensure_ascii=False)}
- knowledge hits used: {json.dumps(knowledge_hits, ensure_ascii=False)}

Recent dialogue:
{recent_block}
""".strip()

        raw = call_model(prompt, json_mode=False)
        text = extract_final(raw).strip()
        if not text:
            return "I keep feeling contaminated after touching things outside, and then I wash my hands over and over."
        return " ".join(text.splitlines()).strip()


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, items: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_text(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _format_knowledge_hits(hits: List[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for item in hits:
        title = str(item.get("title", "")).strip()
        parent_title = str(item.get("parent_title", "")).strip()
        score = item.get("score", 0.0)
        summary = str(item.get("summary", "")).strip()
        label = f"{parent_title} -> {title}" if parent_title else title
        lines.append(f"- {label} (score={score}): {summary}")
    return lines


def run_demo(
    turns: int,
    output_root: Path,
    helper_api_url: str,
    helper_api_key: Optional[str],
    milestone_path: str,
    transcript_path: str,
    knowledge_tree_path: str,
    session_config_path: Optional[str],
    patient_name: str,
    use_helper_model: bool,
    use_knowledge_tree: bool,
    run_label: Optional[str] = None,
) -> Path:
    run_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir = output_root / (run_label or run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    session_paths = build_session_paths(run_dir, prefix="demo_session")
    session = DigitalDoctorSession(
        transcript_path=str(resolve_repo_path(transcript_path)),
        helper_api_url=helper_api_url,
        helper_api_key=helper_api_key,
        milestone_path=str(resolve_repo_path(milestone_path)),
        session_config_path=str(resolve_repo_path(session_config_path)) if session_config_path else None,
        knowledge_tree_path=str(resolve_repo_path(knowledge_tree_path)),
        user_id="virtual_patient",
        episode_id="demo_dialogue",
        patient_role_label="virtual_patient",
        doctor_role_label="virtual_doctor",
        memory_path=session_paths["memory_path"],
        long_term_memory_path=session_paths["long_term_memory_path"],
        state_path=session_paths["state_path"],
        log_path=session_paths["log_path"],
        trace_path=session_paths["trace_path"],
        alert_path=session_paths["alert_path"],
        single_turn=False,
        use_helper_model=use_helper_model,
        use_knowledge_tree=use_knowledge_tree,
    )

    simulator = GptPatientSimulator(name=patient_name)
    dialogue_history: List[Dict[str, str]] = []
    turn_logs: List[Dict[str, object]] = []
    last_update: Optional[Dict[str, object]] = None

    turn_iter = range(turns)
    if tqdm is not None:
        turn_iter = tqdm(turn_iter, total=turns, desc=f"Dialogue[{run_label or run_id}]", leave=False)

    for turn_index in turn_iter:
        patient_msg = simulator.generate(turn_index, dialogue_history, last_update)
        doctor_msg, update = session.handle_query(patient_msg)

        dialogue_history.append({"role": "virtual_patient", "text": patient_msg})
        dialogue_history.append({"role": "virtual_doctor", "text": doctor_msg})
        last_update = update

        artifacts = update.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        transcript_refs = artifacts.get("transcript_refs", [])
        knowledge_hits = artifacts.get("knowledge_hits", [])
        source_candidates = artifacts.get("source_candidates", update.get("source_candidates", {}))
        turn_logs.append(
            {
                "turn": turn_index + 1,
                "virtual_patient": patient_msg,
                "virtual_doctor": doctor_msg,
                "route": update.get("route"),
                "coverage": f"{update.get('covered_count', 0)}/{update.get('total', 0)}",
                "next_target": update.get("next_target"),
                "status_changes": update.get("status_changes", []),
                "top_scores": update.get("top_scores", []),
                "transcript_refs": transcript_refs if isinstance(transcript_refs, list) else [],
                "knowledge_hits": knowledge_hits if isinstance(knowledge_hits, list) else [],
                "milestone_context_before_turn": artifacts.get("milestone_context_before_turn", ""),
                "milestone_snapshot_before_turn": artifacts.get("milestone_snapshot_before_turn", {}),
                "milestone_snapshot_after_turn": artifacts.get("milestone_snapshot_after_turn", {}),
                "helper_query": artifacts.get("helper_query", ""),
                "helper_answer": artifacts.get("helper_answer", ""),
                "source_candidates": source_candidates if isinstance(source_candidates, dict) else {},
            }
        )

    dialogue_path = run_dir / "dialogue.json"
    turns_path = run_dir / "turn_logs.jsonl"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "dialogue_report.txt"

    _write_json(dialogue_path, {"run_id": run_id, "turns": turn_logs})
    _write_jsonl(turns_path, turn_logs)

    report_lines: List[str] = []
    for item in turn_logs:
        report_lines.append(f"Turn {item['turn']}")
        report_lines.append(f"virtual_patient: {item['virtual_patient']}")
        report_lines.append(f"virtual_doctor: {item['virtual_doctor']}")
        report_lines.append(f"Route={item['route']} Coverage={item['coverage']} Next={item['next_target']}")
        report_lines.append("Transcript refs:")
        refs = item.get("transcript_refs", [])
        if isinstance(refs, list) and refs:
            report_lines.extend([f"- {ref}" for ref in refs])
        else:
            report_lines.append("- (none)")
        report_lines.append("Knowledge hits:")
        hits = item.get("knowledge_hits", [])
        if isinstance(hits, list) and hits:
            report_lines.extend(_format_knowledge_hits(hits))
        else:
            report_lines.append("- (none)")
        report_lines.append("Milestone context before turn:")
        report_lines.append(str(item.get("milestone_context_before_turn", "")).strip() or "(none)")
        report_lines.append(f"Status changes: {json.dumps(item.get('status_changes', []), ensure_ascii=False)}")
        report_lines.append("Source candidates:")
        candidates = item.get("source_candidates", {})
        if isinstance(candidates, dict) and candidates:
            for label in ("transcript", "knowledge_tree", "combined"):
                text = str(candidates.get(label, "")).strip()
                if text:
                    report_lines.append(f"[{label}] {text}")
        else:
            report_lines.append("(none)")
        report_lines.append("")
    _write_text(report_path, report_lines)

    summary = {
        "run_id": run_id,
        "turns": turns,
        "final_update": last_update,
        "final_snapshot": session.tracker.snapshot(),
        "paths": {
            "dialogue_json": repo_relative_path(dialogue_path),
            "turn_logs_jsonl": repo_relative_path(turns_path),
            "dialogue_report_txt": repo_relative_path(report_path),
            "summary_json": repo_relative_path(summary_path),
            **session_paths,
        },
    }
    _write_json(summary_path, summary)

    print(f"Saved demo dialogue to: {run_dir}")
    if last_update:
        print(f"Final coverage: {last_update.get('covered_count', 0)}/{last_update.get('total', 0)}")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a 10-turn GPT patient vs OCD doctor dialogue and save per-turn retrieval logs."
    )
    parser.add_argument("--turns", type=int, default=10, help="Number of dialogue turns (default: 10).")
    parser.add_argument("--output-dir", type=str, default=str(DEMO_LOG_DIR), help="Output root folder.")
    parser.add_argument("--run-label", type=str, default=None, help="Optional stable folder name for this run.")
    parser.add_argument("--patient-name", type=str, default="Patient", help="Display name for the simulated patient.")
    parser.add_argument(
        "--helper-api-url",
        type=str,
        default=os.getenv("HELPER_API_URL", "http://localhost:8001/helper/generate"),
        help="Helper API URL.",
    )
    parser.add_argument(
        "--helper-api-key",
        type=str,
        default=os.getenv("HELPER_API_KEY"),
        help="Optional helper API key.",
    )
    parser.add_argument(
        "--milestone-path",
        type=str,
        default=os.getenv("MILESTONE_PATH", DEFAULT_MILESTONE_PATH),
        help="Milestone markdown path.",
    )
    parser.add_argument(
        "--transcript-path",
        type=str,
        default=os.getenv("TRANSCRIPT_PATH", DEFAULT_TRANSCRIPT_PATH),
        help="Transcript JSON path.",
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

    helper_group = parser.add_mutually_exclusive_group()
    helper_group.add_argument("--use-helper-model", dest="use_helper_model", action="store_true")
    helper_group.add_argument("--no-helper-model", dest="use_helper_model", action="store_false")

    knowledge_group = parser.add_mutually_exclusive_group()
    knowledge_group.add_argument("--use-knowledge-tree", dest="use_knowledge_tree", action="store_true")
    knowledge_group.add_argument("--no-knowledge-tree", dest="use_knowledge_tree", action="store_false")

    parser.set_defaults(
        use_helper_model=_env_flag("USE_HELPER_MODEL", True),
        use_knowledge_tree=_env_flag("USE_KNOWLEDGE_TREE", True),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_demo(
        turns=max(1, int(args.turns)),
        output_root=Path(repo_relative_path(args.output_dir)),
        helper_api_url=args.helper_api_url,
        helper_api_key=args.helper_api_key,
        milestone_path=args.milestone_path,
        transcript_path=args.transcript_path,
        knowledge_tree_path=args.knowledge_tree_path,
        session_config_path=args.session_config_path,
        patient_name=args.patient_name,
        use_helper_model=args.use_helper_model,
        use_knowledge_tree=args.use_knowledge_tree,
        run_label=args.run_label,
    )


if __name__ == "__main__":
    main()
