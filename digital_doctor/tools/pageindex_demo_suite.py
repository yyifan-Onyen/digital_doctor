from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from digital_doctor.paths import (  # type: ignore
        DEFAULT_KNOWLEDGE_TREE_PATH,
        DEFAULT_MILESTONE_PATH,
        DEFAULT_TRANSCRIPT_PATH,
        DEMO_LOG_DIR,
        repo_relative_path,
        resolve_repo_path,
    )
    from digital_doctor.tools.demo import run_demo  # type: ignore
    from digital_doctor.view.export_presentation import run_export  # type: ignore
else:
    from ..paths import DEFAULT_KNOWLEDGE_TREE_PATH, DEFAULT_MILESTONE_PATH, DEFAULT_TRANSCRIPT_PATH, DEMO_LOG_DIR, repo_relative_path, resolve_repo_path
    from .demo import run_demo
    from ..view.export_presentation import run_export


DEFAULT_EXPORT_DIR = Path("data") / "demo_output" / "runs"
DEFAULT_MANIFEST_PATH = DEFAULT_EXPORT_DIR / "demo_suite_manifest.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple standalone PageIndex-focused demos and export a minimal CSV for each run."
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of standalone demo runs.")
    parser.add_argument("--turns", type=int, default=5, help="Turns per demo run. Minimum is 5.")
    parser.add_argument("--run-prefix", type=str, default="pageindex_demo", help="Stable prefix for run labels.")
    parser.add_argument("--demo-output-dir", type=str, default=str(DEMO_LOG_DIR), help="Runtime folder for raw demo logs.")
    parser.add_argument("--export-dir", type=str, default=str(DEFAULT_EXPORT_DIR), help="Folder for exported CSV/JSON files.")
    parser.add_argument("--manifest-path", type=str, default=str(DEFAULT_MANIFEST_PATH), help="Manifest JSON output path.")
    parser.add_argument("--patient-name", type=str, default="Patient", help="Base patient display name.")
    parser.add_argument(
        "--helper-api-url",
        type=str,
        default=os.getenv("HELPER_API_URL", "http://localhost:8001/helper/generate"),
        help="Helper API URL.",
    )
    parser.add_argument("--helper-api-key", type=str, default=os.getenv("HELPER_API_KEY"), help="Optional helper API key.")
    parser.add_argument("--milestone-path", type=str, default=os.getenv("MILESTONE_PATH", DEFAULT_MILESTONE_PATH))
    parser.add_argument("--transcript-path", type=str, default=os.getenv("TRANSCRIPT_PATH", DEFAULT_TRANSCRIPT_PATH))
    parser.add_argument("--knowledge-tree-path", type=str, default=os.getenv("KNOWLEDGE_TREE_PATH", DEFAULT_KNOWLEDGE_TREE_PATH))
    parser.add_argument("--session-config-path", type=str, default=os.getenv("SESSION_CONFIG_PATH"))
    return parser


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_count = max(1, int(args.runs))
    turns = max(5, int(args.turns))
    demo_output_dir = Path(resolve_repo_path(args.demo_output_dir))
    export_dir = Path(resolve_repo_path(args.export_dir))
    manifest_path = Path(resolve_repo_path(args.manifest_path))
    knowledge_tree_path = Path(resolve_repo_path(args.knowledge_tree_path))

    demo_output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    manifest_runs: List[Dict[str, str]] = []
    for idx in range(1, run_count + 1):
        run_label = f"{args.run_prefix}_{idx}"
        run_dir = demo_output_dir / run_label
        if run_dir.exists():
            shutil.rmtree(run_dir)

        csv_path = export_dir / f"{run_label}.csv"
        json_path = export_dir / f"{run_label}.json"

        demo_dir = run_demo(
            turns=turns,
            output_root=demo_output_dir,
            helper_api_url=args.helper_api_url,
            helper_api_key=args.helper_api_key,
            milestone_path=args.milestone_path,
            transcript_path=args.transcript_path,
            knowledge_tree_path=args.knowledge_tree_path,
            session_config_path=args.session_config_path,
            patient_name=f"{args.patient_name} {idx}",
            use_helper_model=False,
            use_knowledge_tree=True,
            run_label=run_label,
        )
        state_path = demo_dir / "demo_session_state.jsonl"
        run_export(
            state_path=state_path,
            knowledge_tree_path=knowledge_tree_path,
            output_csv=csv_path,
            output_json=json_path,
            top_k=3,
        )
        manifest_runs.append(
            {
                "run_label": run_label,
                "demo_dir": repo_relative_path(demo_dir),
                "csv_path": repo_relative_path(csv_path),
                "json_path": repo_relative_path(json_path),
            }
        )
        print(f"Completed {run_label}")
        print(f"  CSV: {csv_path}")

    write_json(
        manifest_path,
        {
            "runs": manifest_runs,
            "turns_per_run": turns,
            "knowledge_tree_path": repo_relative_path(knowledge_tree_path),
        },
    )
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
