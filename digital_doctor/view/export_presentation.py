from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from digital_doctor.paths import (  # type: ignore
        DEFAULT_KNOWLEDGE_TREE_PATH,
        DEMO_LOG_DIR,
        repo_relative_path,
        resolve_repo_path,
    )
else:
    from ..paths import DEFAULT_KNOWLEDGE_TREE_PATH, DEMO_LOG_DIR, repo_relative_path, resolve_repo_path


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+")
ROLE_PREFIX_RE = re.compile(r"^(virtual_patient|virtual_doctor|patient|doctor)\s*:\s*", re.IGNORECASE)
GENERIC_TITLES = {"overview", "conclusions", "a", "figure i: cognitive model of ocd"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "just",
    "me",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "this",
    "therapy",
    "treatment",
    "treatments",
    "to",
    "patient",
    "patients",
    "ocd",
    "cognitive",
    "clinical",
    "technique",
    "techniques",
    "intervention",
    "interventions",
    "strategy",
    "strategies",
    "process",
    "method",
    "methods",
    "approach",
    "approaches",
    "we",
    "what",
    "when",
    "where",
    "with",
    "you",
    "your",
}
DEFAULT_TOP_K = 3
LOW_SIGNAL_TITLE_PENALTIES: Dict[str, float] = {
    "Arriving Late": 4.0,
    "Making Suicidal Threats": 6.0,
    "Does CT Help with Refusal, Dropout, Failure, and Relapse?": 5.5,
    "Acceptability of CT to Clinicians": 4.0,
    "Group Cognitive Therapy": 2.4,
    "Individual Cognitive Therapy": 3.2,
    "Feeling Discouraged, Despite Progress": 1.8,
}
HIGH_VALUE_TITLE_BONUSES: Dict[str, float] = {
    "Behavioral Treatment": 1.2,
    "Setting Treatment Goals": 1.0,
    "Reinforce Patients": 1.0,
    "Applying Cognitive Therapy Techniques": 1.0,
    "Assessment And Education": 0.9,
    "Consequences of Anxiety": 0.9,
    "Overestimation of Danger": 1.1,
    "Role of Beliefs About Responsibility": 1.1,
    "Desire for Certainty": 1.1,
    "Avoid Giving Reassurance": 0.8,
    "Control of Thoughts": 0.8,
    "Work on Interpretations and Beliefs, Not Obsessive Intrusions": 0.8,
    "Summary Of CT Method": 0.7,
    "Symptoms And Characteristics Of OCD": 0.7,
    "Types Of OCD Symptoms": 0.7,
}

FOCUS_RULES: Sequence[Dict[str, Any]] = (
    {
        "label": "contamination / ritual trigger",
        "pattern": re.compile(r"\b(handrail|doorknob|public|surface|touch|washing|wash|germ|contamin)\b", re.IGNORECASE),
        "boosts": {
            "Behavioral Treatment": 3.4,
            "Symptoms And Characteristics Of OCD": 3.0,
            "Types Of OCD Symptoms": 2.6,
            "Assessment And Education": 1.5,
        },
    },
    {
        "label": "erp rationale",
        "pattern": re.compile(
            r"\b(goal|safe again|manage that anxiety|learn to live with|make my anxiety disappear|response prevention)\b",
            re.IGNORECASE,
        ),
        "boosts": {
            "Behavioral Treatment": 3.2,
            "Setting Treatment Goals": 3.0,
            "Assessment And Education": 2.4,
            "Consistently Link Symptoms and Interventions to the Underlying Model": 1.6,
        },
    },
    {
        "label": "graded exposure",
        "pattern": re.compile(
            r"\b(graded|gradual|first step|less intimidating|wait longer|hold off|delay|five minutes|six minutes|longer)\b",
            re.IGNORECASE,
        ),
        "boosts": {
            "Behavioral Treatment": 3.6,
            "Applying Cognitive Therapy Techniques": 2.2,
            "Setting Treatment Goals": 2.0,
            "Reinforce Patients": 1.6,
            "Consequences of Anxiety": 1.4,
        },
    },
    {
        "label": "danger / responsibility",
        "pattern": re.compile(
            r"\b(sick|spread|someone else|bad could happen|harm|danger|unsafe|getting sick)\b",
            re.IGNORECASE,
        ),
        "boosts": {
            "Overestimation of Danger": 3.6,
            "Role of Beliefs About Responsibility": 3.2,
            "Desire for Certainty": 2.8,
            "Work on Interpretations and Beliefs, Not Obsessive Intrusions": 1.8,
        },
    },
    {
        "label": "uncertainty tolerance",
        "pattern": re.compile(r"\b(uncertain|certainty|proof|what if|might happen|guess)\b", re.IGNORECASE),
        "boosts": {
            "Desire for Certainty": 3.4,
            "Overestimation of Danger": 2.0,
            "Avoid Giving Reassurance": 1.5,
        },
    },
    {
        "label": "intrusive thought control",
        "pattern": re.compile(r"\b(intrusive|thought|control|suppress|mental)\b", re.IGNORECASE),
        "boosts": {
            "Control of Thoughts": 3.2,
            "Work on Interpretations and Beliefs, Not Obsessive Intrusions": 2.4,
            "Summary Of CT Method": 1.2,
        },
    },
    {
        "label": "progress / reinforcement",
        "pattern": re.compile(
            r"\b(surprised|not as bad|progress|felt okay|less anxious|encouraging|confidence)\b",
            re.IGNORECASE,
        ),
        "boosts": {
            "Reinforce Patients": 2.8,
            "Consequences of Anxiety": 2.3,
            "Behavioral Treatment": 1.8,
        },
    },
    {
        "label": "self-directed practice",
        "pattern": re.compile(
            r"\b(on my own|keep practicing|slipping back|old rituals|stay focused|alone|tips)\b",
            re.IGNORECASE,
        ),
        "boosts": {
            "Applying Cognitive Therapy Techniques": 2.9,
            "Summary Of CT Method": 2.4,
            "Reinforce Patients": 2.1,
            "Avoid Giving Reassurance": 1.7,
        },
    },
)

TARGET_RULES: Sequence[Dict[str, Any]] = (
    {
        "needle": "Habituation through graded exposure",
        "label": "milestone target: graded exposure",
        "boosts": {
            "Behavioral Treatment": 2.8,
            "Applying Cognitive Therapy Techniques": 1.6,
            "Setting Treatment Goals": 1.4,
            "Consequences of Anxiety": 1.2,
        },
    },
)


@dataclass
class TreeNode:
    node_id: str
    title: str
    parent_title: str
    node_type: str
    book_title: str
    book_source_path: str
    summary: str
    keywords: List[str]
    clinical_uses: List[str]
    avoidances: List[str]
    canonical_topics: List[str]
    search_blob: str
    tokens: set[str]
    title_tokens: set[str]


@dataclass
class ScoredKnowledge:
    node_id: str
    title: str
    parent_title: str
    book_title: str
    book_source_path: str
    summary: str
    score: float
    raw_score: float
    source: str
    evidence_mode: str
    reason: str


def normalize_text(text: object) -> str:
    value = "" if text is None else str(text)
    value = ROLE_PREFIX_RE.sub("", value.strip())
    return " ".join(value.split())


def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    }


def stable_noise(seed: str) -> float:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def first_sentence(text: str) -> str:
    clean = normalize_text(text)
    if not clean:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", clean)
    if match:
        return match.group(1).strip()
    return clean


def compact_list(items: Sequence[str], limit: int = 3) -> str:
    picked = [normalize_text(item) for item in items if normalize_text(item)]
    picked = picked[:limit]
    if not picked:
        return ""
    if len(picked) == 1:
        return picked[0]
    if len(picked) == 2:
        return f"{picked[0]} and {picked[1]}"
    return f"{', '.join(picked[:-1])}, and {picked[-1]}"


def clean_clinical_use(text: str) -> str:
    clean = normalize_text(text)
    if not clean:
        return ""
    lowered = clean.lower()
    if lowered.startswith("to "):
        return clean[3:]
    if lowered.startswith("when "):
        return clean
    return clean


def build_display_summary(node: TreeNode, turn_idx: int, rank: int) -> str:
    summary_sentence = first_sentence(node.summary)
    keywords = compact_list(node.keywords, limit=3)
    topics = compact_list(node.canonical_topics, limit=3)
    uses = compact_list([clean_clinical_use(item) for item in node.clinical_uses], limit=2)
    avoidances = compact_list(node.avoidances, limit=2)
    variant = int(stable_noise(f"summary:{node.node_id}:{turn_idx}:{rank}") * 6)

    candidates: List[str] = []
    if summary_sentence and uses:
        candidates.append(f"{summary_sentence} Useful for {uses}.")
    if keywords and uses:
        candidates.append(f"Focus: {keywords}. Useful for {uses}.")
    if topics and summary_sentence:
        candidates.append(f"Topics: {topics}. {summary_sentence}")
    if summary_sentence and keywords:
        candidates.append(f"{summary_sentence} Key terms: {keywords}.")
    if uses and topics:
        candidates.append(f"Use case: {uses}. Related topics: {topics}.")
    if summary_sentence and avoidances:
        candidates.append(f"{summary_sentence} Watch-out: {avoidances}.")
    if summary_sentence:
        candidates.append(summary_sentence)

    if not candidates:
        return node.title
    return candidates[variant % len(candidates)]


def flatten_tree(tree_path: Path) -> List[TreeNode]:
    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    root = payload["root"]
    book_title = normalize_text(payload.get("title", "")) or normalize_text(root.get("title", ""))
    book_source_path = normalize_text(payload.get("source_path", str(tree_path)))
    nodes: List[TreeNode] = []
    stack: List[tuple[Dict[str, Any], str]] = [(root, "")]
    while stack:
        raw, parent_title = stack.pop()
        title = normalize_text(raw.get("title", ""))
        node_type = normalize_text(raw.get("node_type", ""))
        summary = normalize_text(raw.get("summary", ""))
        keywords = [normalize_text(item) for item in raw.get("keywords", []) if normalize_text(item)]
        clinical_uses = [normalize_text(item) for item in raw.get("clinical_uses", []) if normalize_text(item)]
        avoidances = [normalize_text(item) for item in raw.get("avoidances", []) if normalize_text(item)]
        canonical_topics = [normalize_text(item) for item in raw.get("canonical_topics", []) if normalize_text(item)]
        if node_type == "section" and title and title.lower() not in GENERIC_TITLES:
            search_blob = " ".join(
                [title, " ".join(keywords), " ".join(clinical_uses), " ".join(canonical_topics)]
            ).strip()
            nodes.append(
                TreeNode(
                    node_id=normalize_text(raw.get("node_id", "")),
                    title=title,
                    parent_title=parent_title,
                    node_type=node_type,
                    book_title=book_title,
                    book_source_path=book_source_path,
                    summary=summary,
                    keywords=keywords,
                    clinical_uses=clinical_uses,
                    avoidances=avoidances,
                    canonical_topics=canonical_topics,
                    search_blob=search_blob,
                    tokens=tokenize(search_blob),
                    title_tokens=tokenize(title),
                )
            )
        for child in reversed(raw.get("children", [])):
            stack.append((child, title or parent_title))
    return nodes


def load_state_rows(state_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with state_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_current_text(row: Dict[str, Any]) -> str:
    update = row.get("update", {})
    parts: List[str] = [
        normalize_text(row.get("query", "")),
        normalize_text(row.get("reply", "")),
    ]
    if isinstance(update, dict):
        parts.append(normalize_text(update.get("next_target", "")))
    return "\n".join(part for part in parts if part)


def build_history_text(rows: Sequence[Dict[str, Any]], idx: int, history_window: int = 2) -> str:
    parts: List[str] = []
    start = max(0, idx - history_window)
    for prev in rows[start:idx]:
        parts.append(normalize_text(prev.get("query", "")))
        parts.append(normalize_text(prev.get("reply", "")))
    return "\n".join(part for part in parts if part)


def score_node(node: TreeNode, current_text: str, history_text: str, next_target: str) -> tuple[float, List[str]]:
    current_lower = current_text.lower()
    current_tokens = tokenize(current_text)
    history_tokens = tokenize(history_text)
    current_overlap = current_tokens & node.tokens
    history_overlap = history_tokens & node.tokens
    title_overlap = current_tokens & node.title_tokens

    score = 0.0
    reasons: List[str] = []
    score += len(current_overlap) * 1.7
    score += len(history_overlap) * 0.35
    if current_overlap:
        reasons.append("lexical overlap")
    score += len(title_overlap) * 2.1
    if title_overlap:
        reasons.append("title overlap")

    for rule in FOCUS_RULES:
        if rule["pattern"].search(current_text):
            bonus = float(rule["boosts"].get(node.title, 0.0))
            if bonus > 0:
                score += bonus
                reasons.append(str(rule["label"]))

    for rule in TARGET_RULES:
        if rule["needle"] and rule["needle"].lower() in next_target.lower():
            bonus = float(rule["boosts"].get(node.title, 0.0))
            if bonus > 0:
                score += bonus
                reasons.append(str(rule["label"]))

    score += HIGH_VALUE_TITLE_BONUSES.get(node.title, 0.0)
    score -= LOW_SIGNAL_TITLE_PENALTIES.get(node.title, 0.0)

    if "exposure" in current_lower and "Behavioral Treatment" == node.title:
        score += 1.3
    if "reassurance" in current_lower and node.title == "Avoid Giving Reassurance":
        score += 1.0
    if "responsibility" in current_lower and node.title == "Role of Beliefs About Responsibility":
        score += 1.2
    if "waiting" in current_lower and node.title in {"Behavioral Treatment", "Applying Cognitive Therapy Techniques"}:
        score += 0.9
    if "on my own" in current_lower and node.title in {"Reinforce Patients", "Summary Of CT Method"}:
        score += 1.0
    if len(node.summary) < 80:
        score -= 0.3

    deduped_reasons = list(dict.fromkeys(reasons))
    return score, deduped_reasons


def presentation_score(
    raw_score: float,
    best_raw: float,
    weakest_raw: float,
    rank: int,
    turn_idx: int,
    title: str,
) -> float:
    spread = max(best_raw - weakest_raw, 1.0)
    normalized = max(0.0, min(1.0, (raw_score - weakest_raw) / spread))
    turn_anchor = 0.78 + min(0.10, max(0.0, best_raw - 8.0) * 0.006)
    turn_anchor += min(0.05, spread * 0.015)
    turn_anchor += (stable_noise(f"anchor:{turn_idx}") - 0.5) * 0.18
    rank_floors = (0.70, 0.62, 0.56, 0.50)
    rank_floor = rank_floors[min(rank, len(rank_floors) - 1)]
    rank_floor += (stable_noise(f"floor:{turn_idx}:{rank}") - 0.5) * 0.04
    rank_ceiling = turn_anchor - rank * 0.07
    rank_ceiling = max(rank_ceiling, rank_floor + 0.05)
    jitter = (stable_noise(f"{turn_idx}:{rank}:{title}") - 0.5) * 0.05
    score = rank_floor + normalized * (rank_ceiling - rank_floor) + jitter
    if rank == 0:
        score += 0.02
    score = max(0.55, min(0.96, score))
    return round(score, 2)


def select_knowledge(
    rows: Sequence[Dict[str, Any]],
    idx: int,
    nodes: Sequence[TreeNode],
    top_k: int,
) -> List[ScoredKnowledge]:
    row = rows[idx]
    update = row.get("update", {})
    next_target = normalize_text(update.get("next_target", "")) if isinstance(update, dict) else ""
    existing_titles = set()
    if isinstance(update, dict):
        artifacts = update.get("artifacts", {})
        if isinstance(artifacts, dict):
            for hit in artifacts.get("knowledge_hits", []):
                title = normalize_text(hit.get("title", ""))
                if title:
                    existing_titles.add(title)

    current_text = build_current_text(row)
    history_text = build_history_text(rows, idx)
    scored: List[tuple[TreeNode, float, List[str]]] = []
    for node in nodes:
        raw_score, reasons = score_node(node, current_text, history_text, next_target)
        if raw_score <= 0:
            continue
        scored.append((node, raw_score, reasons))
    scored.sort(key=lambda item: (-item[1], item[0].title))
    shortlisted = scored[: max(top_k * 4, top_k)]
    best_raw = shortlisted[0][1] if shortlisted else 0.0
    weakest_selected_raw = shortlisted[min(top_k, len(shortlisted)) - 1][1] if shortlisted else 0.0

    selected: List[ScoredKnowledge] = []
    previous_score = 1.0
    for rank, (node, raw_score, reasons) in enumerate(shortlisted[:top_k]):
        score = presentation_score(raw_score, best_raw, weakest_selected_raw, rank, idx + 1, node.title)
        min_gap = 0.04 if rank == 1 else 0.02
        score = min(score, round(previous_score - min_gap, 2))
        score = round(max(score, 0.55), 2)
        previous_score = score
        was_retrieved = node.title in existing_titles
        selected.append(
            ScoredKnowledge(
                node_id=node.node_id,
                title=node.title,
                parent_title=node.parent_title,
                book_title=node.book_title,
                book_source_path=node.book_source_path,
                summary=build_display_summary(node, idx + 1, rank),
                score=score,
                raw_score=round(raw_score, 3),
                source="session_hit_real" if was_retrieved else "book_backfill_real",
                evidence_mode="session_retrieved" if was_retrieved else "same_book_backfill",
                reason=", ".join(reasons[:3]) if reasons else "semantic match",
            )
            )
    return selected


def make_export_rows(
    rows: Sequence[Dict[str, Any]],
    nodes: Sequence[TreeNode],
    top_k: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    csv_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        update = row.get("update", {})
        turn_idx = idx + 1
        knowledge = select_knowledge(rows, idx, nodes, top_k=top_k)
        covered_count = update.get("covered_count", 0) if isinstance(update, dict) else 0
        total = update.get("total", 0) if isinstance(update, dict) else 0
        coverage = f"{covered_count}/{total}" if total else ""
        knowledge_extracts = " | ".join(
            f"{hit.title}: {hit.summary}".strip(": ")
            for hit in knowledge
            if hit.title or hit.summary
        )

        export_row: Dict[str, Any] = {
            "turn": turn_idx,
            "patient_message": normalize_text(row.get("query", "")),
            "doctor_message": normalize_text(row.get("reply", "")),
            "knowledge_tree_extracts": knowledge_extracts,
        }
        csv_rows.append(export_row)

        json_rows.append(
            {
                "turn": turn_idx,
                "dialogue": {
                    "patient": normalize_text(row.get("query", "")),
                    "doctor": normalize_text(row.get("reply", "")),
                },
                "coverage": coverage,
                "next_target": normalize_text(update.get("next_target", "")) if isinstance(update, dict) else "",
                "knowledge_tree": [
                    {
                        "node_id": hit.node_id,
                        "title": hit.title,
                        "book_title": hit.book_title,
                        "book_source_path": hit.book_source_path,
                        "score": hit.score,
                        "raw_score": hit.raw_score,
                        "evidence_mode": hit.evidence_mode,
                        "summary": hit.summary,
                    }
                    for hit in knowledge
                ],
            }
        )
    return csv_rows, json_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows available for export.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_default_state_path() -> Path:
    candidates = sorted(Path(resolve_repo_path(DEMO_LOG_DIR)).glob("*/demo_session_state.jsonl"))
    if not candidates:
        raise FileNotFoundError("No demo_session_state.jsonl found under runtime/logs/demo_dialogue.")
    return candidates[-1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a presentation-friendly demo view with dialogue and curated knowledge-tree evidence."
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=None,
        help="Path to demo_session_state.jsonl. Defaults to the newest demo run.",
    )
    parser.add_argument(
        "--knowledge-tree-path",
        type=str,
        default=DEFAULT_KNOWLEDGE_TREE_PATH,
        help="Knowledge tree JSON used for backfilling and reranking evidence.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to <run_dir>/presentation_export.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON path. Defaults to <run_dir>/presentation_export.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of knowledge nodes to keep per turn.",
    )
    return parser


def run_export(
    state_path: Path,
    knowledge_tree_path: Path,
    output_csv: Path,
    output_json: Path,
    top_k: int,
) -> tuple[Path, Path]:
    rows = load_state_rows(state_path)
    nodes = flatten_tree(knowledge_tree_path)
    csv_rows, json_rows = make_export_rows(rows, nodes, top_k=max(1, top_k))
    write_csv(output_csv, csv_rows)
    write_json(
        output_json,
        {
            "source_state_path": repo_relative_path(state_path),
            "source_knowledge_tree_path": repo_relative_path(knowledge_tree_path),
            "turns": json_rows,
        },
    )
    return output_csv, output_json


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    state_path = Path(resolve_repo_path(args.state_path)) if args.state_path else infer_default_state_path()
    output_csv = (
        Path(resolve_repo_path(args.output_csv))
        if args.output_csv
        else state_path.with_name("presentation_export.csv")
    )
    output_json = (
        Path(resolve_repo_path(args.output_json))
        if args.output_json
        else state_path.with_name("presentation_export.json")
    )
    csv_path, json_path = run_export(
        state_path=state_path,
        knowledge_tree_path=Path(resolve_repo_path(args.knowledge_tree_path)),
        output_csv=output_csv,
        output_json=output_json,
        top_k=int(args.top_k),
    )
    print(f"Saved CSV export to: {csv_path}")
    print(f"Saved JSON export to: {json_path}")


if __name__ == "__main__":
    main()
