from __future__ import annotations

import shutil
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]

REMOVE_DIRS = [
    "__pycache__",
    "digital_doctor/__pycache__",
    "digital_doctor/core/__pycache__",
    "digital_doctor/retrieval/__pycache__",
    "digital_doctor/services/__pycache__",
    "digital_doctor/tracking/__pycache__",
    "digital_doctor/tools/__pycache__",
    "digital_doctor/view/__pycache__",
    "scripts/__pycache__",
    "pageindex/__pycache__",
    "runtime/logs",
    "mile_res",
    "data/knowledge_trees/.cache",
]

REMOVE_FILES = [
    "runtime/cache/segment_summaries.jsonl",
]

RECREATE_DIRS = [
    "runtime/cache",
    "runtime/cache/knowledge_trees",
    "runtime/logs/interactive",
    "runtime/logs/demo_dialogue",
]


def main() -> None:
    for relative in REMOVE_DIRS:
        target = REPO_DIR / relative
        if target.exists():
            shutil.rmtree(target)

    for relative in REMOVE_FILES:
        target = REPO_DIR / relative
        if target.exists():
            target.unlink()

    for relative in RECREATE_DIRS:
        (REPO_DIR / relative).mkdir(parents=True, exist_ok=True)

    print("Workspace cleaned.")


if __name__ == "__main__":
    main()
