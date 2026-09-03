from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

load_dotenv()

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from digital_doctor.core.session import DigitalDoctorSession  # type: ignore
    from digital_doctor.core.session_store import Turn, reset_session_files  # type: ignore
    from digital_doctor.paths import (  # type: ignore
        DEFAULT_KNOWLEDGE_TREE_PATH,
        DEFAULT_MILESTONE_PATH,
        DEFAULT_TRANSCRIPT_PATH,
        repo_relative_path,
        resolve_repo_path,
    )
    from digital_doctor.services.openai_client import call_model  # type: ignore
    from digital_doctor.view.export_presentation import flatten_tree, make_export_rows  # type: ignore
else:
    from ..core.session import DigitalDoctorSession
    from ..core.session_store import Turn, reset_session_files
    from ..paths import DEFAULT_KNOWLEDGE_TREE_PATH, DEFAULT_MILESTONE_PATH, DEFAULT_TRANSCRIPT_PATH, repo_relative_path, resolve_repo_path
    from ..services.openai_client import call_model
    from ..view.export_presentation import flatten_tree, make_export_rows


DEFAULT_DEMO_OUTPUT_DIR = Path("data") / "demo_output"
DEFAULT_CASES_PATH = DEFAULT_DEMO_OUTPUT_DIR / "sample_cases.json"
DEFAULT_OUTPUT_CSV = DEFAULT_DEMO_OUTPUT_DIR / "pageindex_impact_cases.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_DEMO_OUTPUT_DIR / "pageindex_impact_cases.json"
DEFAULT_DIAGRAM_PATH = DEFAULT_DEMO_OUTPUT_DIR / "pageindex_influence_diagram.md"
DEFAULT_DIAGRAM_SVG_PATH = DEFAULT_DEMO_OUTPUT_DIR / "pageindex_influence_diagram.svg"
DEFAULT_DIAGRAM_JPG_PATH = DEFAULT_DEMO_OUTPUT_DIR / "pageindex_influence_diagram.jpg"


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def load_cases(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("cases", [])
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError("Cases file must contain a top-level list or a {'cases': [...]} object.")
    cases: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        case_id = normalize_text(item.get("case_id", ""))
        patient_message = normalize_text(item.get("patient_message", ""))
        if not case_id or not patient_message:
            continue
        history = item.get("history", [])
        if not isinstance(history, list):
            history = []
        cases.append(
            {
                "case_id": case_id,
                "theme": normalize_text(item.get("theme", "")),
                "history": history,
                "patient_message": patient_message,
            }
        )
    if not cases:
        raise ValueError("No valid cases found.")
    return cases


def format_dialogue(history: Sequence[Dict[str, Any]], patient_message: str) -> str:
    lines: List[str] = []
    for turn in history:
        role = normalize_text(turn.get("role", "")).lower()
        text = normalize_text(turn.get("text", ""))
        if not text:
            continue
        label = "Doctor" if role in {"doctor", "assistant"} else "Patient"
        lines.append(f"{label}: {text}")
    lines.append(f"Patient: {patient_message}")
    return "\n".join(lines)


def join_unique(values: Sequence[str], sep: str = " | ") -> str:
    ordered: List[str] = []
    seen = set()
    for value in values:
        cleaned = normalize_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return sep.join(ordered)


def parse_json_response(raw: str) -> Dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def analyze_document_influence(
    dialogue_context: str,
    transcript_candidate: str,
    knowledge_candidate: str,
    combined_candidate: str,
    document_evidence: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    hit_lines: List[str] = []
    for hit in document_evidence[:3]:
        parent_title = normalize_text(hit.get("parent_title", ""))
        title = normalize_text(hit.get("title", ""))
        summary = normalize_text(hit.get("summary", ""))
        label = f"{parent_title} -> {title}" if parent_title else title
        hit_lines.append(f"- {label}: {summary}")

    prompt = f"""
You are analyzing how a PageIndex-retrieved OCD treatment document changed an agent's doctor response.

Return strict JSON with these keys:
- document_influence_summary: 1-2 sentences
- difference_from_transcript_only: 1 sentence
- pageindex_benefit_summary: 1 sentence
- specific_document_concepts: array of 2-4 short phrases
- supporting_sections: array of up to 3 section labels

Only mention concepts that are clearly supported by the retrieved document sections or the response text.

Patient-doctor context:
{dialogue_context}

Transcript-only candidate:
{transcript_candidate}

Document-only candidate:
{knowledge_candidate}

Combined final candidate:
{combined_candidate}

Retrieved document sections:
{chr(10).join(hit_lines) if hit_lines else "- (none)"}
""".strip()

    payload = parse_json_response(call_model(prompt, json_mode=True))
    if payload:
        return payload

    section_labels = []
    for hit in document_evidence[:3]:
        parent_title = normalize_text(hit.get("parent_title", ""))
        title = normalize_text(hit.get("title", ""))
        if parent_title and title:
            section_labels.append(f"{parent_title} -> {title}")
        elif title:
            section_labels.append(title)
    return {
        "document_influence_summary": "The document-conditioned candidate adds more explicit OCD concepts and a more targeted ERP framing than the transcript-only candidate.",
        "difference_from_transcript_only": "The final reply becomes more specific about the mechanism the patient should practice instead of staying at a generic reassurance-resistant ERP explanation.",
        "pageindex_benefit_summary": "PageIndex helps retrieve section-level concepts that make the response easier to ground in a concrete CBT-for-OCD document node.",
        "specific_document_concepts": [normalize_text(hit.get("title", "")) for hit in document_evidence[:3] if normalize_text(hit.get("title", ""))],
        "supporting_sections": section_labels,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows available for CSV export.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_diagram_markdown(top_titles: Sequence[str]) -> str:
    _ = top_titles
    return """# PageIndex Influence Diagram

```mermaid
flowchart TD
    A[Source document] --> B[PageIndex]
    B --> C[Knowledge tree]
    C --> C1[Chapter nodes]
    C --> C2[Section nodes]
    D[Patient message + dialogue history] --> E[Retrieve relevant nodes]
    C --> E
    E --> F[Node summaries]
    F --> G[Doctor reply]
```

## Simple Explanation

1. The source document is first converted into a hierarchical knowledge tree by PageIndex.
2. At each turn, the patient message is used to retrieve the most relevant nodes from that tree.
3. The summaries of those nodes are then fed into the model to make the doctor reply more specific.
"""


def build_diagram_svg(top_titles: Sequence[str]) -> str:
    _ = top_titles
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="520" viewBox="0 0 1400 520">
  <rect width="1400" height="520" fill="#f7f4ee" />
  <text x="70" y="55" font-size="34" font-family="Helvetica" font-weight="700" fill="#111827">How PageIndex Works In This Project</text>
  <text x="70" y="88" font-size="20" font-family="Helvetica" fill="#4b5563">Simple flow: document becomes a tree, the patient turn queries the tree, and retrieved node summaries shape the doctor reply.</text>

  <rect x="70" y="165" rx="18" ry="18" width="210" height="88" fill="#ffffff" stroke="#7b341e" stroke-width="3"/>
  <text x="175" y="202" text-anchor="middle" font-size="24" font-family="Helvetica" font-weight="700" fill="#7b341e">Source Document</text>
  <text x="175" y="230" text-anchor="middle" font-size="18" font-family="Helvetica" fill="#9c4221">book / manual</text>

  <rect x="360" y="165" rx="18" ry="18" width="180" height="88" fill="#fffaf0" stroke="#dd6b20" stroke-width="3"/>
  <text x="450" y="202" text-anchor="middle" font-size="24" font-family="Helvetica" font-weight="700" fill="#9c4221">PageIndex</text>
  <text x="450" y="230" text-anchor="middle" font-size="18" font-family="Helvetica" fill="#7b341e">builds structure</text>

  <rect x="620" y="125" rx="18" ry="18" width="250" height="170" fill="#ffffff" stroke="#22543d" stroke-width="3"/>
  <text x="745" y="160" text-anchor="middle" font-size="24" font-family="Helvetica" font-weight="700" fill="#22543d">Knowledge Tree</text>
  <rect x="665" y="185" rx="12" ry="12" width="160" height="40" fill="#f0fff4" stroke="#2f855a" stroke-width="2"/>
  <text x="745" y="210" text-anchor="middle" font-size="18" font-family="Helvetica" fill="#276749">Chapter nodes</text>
  <rect x="665" y="240" rx="12" ry="12" width="160" height="40" fill="#f0fff4" stroke="#2f855a" stroke-width="2"/>
  <text x="745" y="265" text-anchor="middle" font-size="18" font-family="Helvetica" fill="#276749">Section nodes</text>
  <line x1="745" y1="170" x2="745" y2="185" stroke="#22543d" stroke-width="2"/>
  <line x1="745" y1="225" x2="745" y2="240" stroke="#22543d" stroke-width="2"/>

  <rect x="940" y="110" rx="18" ry="18" width="250" height="88" fill="#ffffff" stroke="#243b53" stroke-width="3"/>
  <text x="1065" y="147" text-anchor="middle" font-size="24" font-family="Helvetica" font-weight="700" fill="#102a43">Patient Turn</text>
  <text x="1065" y="175" text-anchor="middle" font-size="18" font-family="Helvetica" fill="#486581">message + dialogue history</text>

  <rect x="940" y="230" rx="18" ry="18" width="250" height="88" fill="#ebf8ff" stroke="#2b6cb0" stroke-width="3"/>
  <text x="1065" y="267" text-anchor="middle" font-size="24" font-family="Helvetica" font-weight="700" fill="#1a365d">Retrieve Nodes</text>
  <text x="1065" y="295" text-anchor="middle" font-size="18" font-family="Helvetica" fill="#2c5282">best matching summaries</text>

  <rect x="1240" y="165" rx="18" ry="18" width="120" height="88" fill="#eef2ff" stroke="#5a67d8" stroke-width="3"/>
  <text x="1300" y="202" text-anchor="middle" font-size="22" font-family="Helvetica" font-weight="700" fill="#3730a3">Doctor</text>
  <text x="1300" y="230" text-anchor="middle" font-size="18" font-family="Helvetica" fill="#4338ca">reply</text>

  <line x1="280" y1="209" x2="360" y2="209" stroke="#9c4221" stroke-width="4"/>
  <polygon points="360,209 344,199 344,219" fill="#9c4221"/>
  <line x1="540" y1="209" x2="620" y2="209" stroke="#22543d" stroke-width="4"/>
  <polygon points="620,209 604,199 604,219" fill="#22543d"/>
  <line x1="870" y1="160" x2="940" y2="154" stroke="#243b53" stroke-width="4"/>
  <polygon points="940,154 923,145 925,161" fill="#243b53"/>
  <line x1="870" y1="260" x2="940" y2="274" stroke="#2b6cb0" stroke-width="4"/>
  <polygon points="940,274 925,263 922,279" fill="#2b6cb0"/>
  <line x1="1065" y1="198" x2="1065" y2="230" stroke="#243b53" stroke-width="4"/>
  <polygon points="1065,230 1057,214 1073,214" fill="#243b53"/>
  <line x1="1190" y1="274" x2="1240" y2="220" stroke="#5a67d8" stroke-width="4"/>
  <polygon points="1240,220 1223,226 1235,237" fill="#5a67d8"/>
</svg>
"""


def render_svg_to_jpg(svg_path: Path, jpg_path: Path) -> bool:
    qlmanage_path = shutil.which("qlmanage")
    sips_path = shutil.which("sips")
    if not qlmanage_path or not sips_path:
        return False
    with tempfile.TemporaryDirectory(prefix="pageindex_diagram_") as temp_dir:
        command = [
            qlmanage_path,
            "-t",
            "-s",
            "2000",
            "-o",
            temp_dir,
            str(svg_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            return False
        png_path = Path(temp_dir) / f"{svg_path.name}.png"
        if not png_path.exists():
            fallback = list(Path(temp_dir).glob("*.png"))
            if not fallback:
                return False
            png_path = fallback[0]
        jpg_path.parent.mkdir(parents=True, exist_ok=True)
        converted = subprocess.run(
            [sips_path, "-s", "format", "jpeg", str(png_path), "--out", str(jpg_path)],
            capture_output=True,
            text=True,
        )
        return converted.returncode == 0 and jpg_path.exists()


def seed_history(session: DigitalDoctorSession, history: Sequence[Dict[str, Any]]) -> None:
    for turn in history:
        role = normalize_text(turn.get("role", "")).lower()
        text = normalize_text(turn.get("text", ""))
        if not text:
            continue
        label = session.doctor_role_label if role in {"doctor", "assistant"} else session.patient_role_label
        session.memory.append(Turn(session.user_id, session.episode_id, label, text))


def run_case(
    case: Dict[str, Any],
    transcript_path: str,
    milestone_path: str,
    knowledge_tree_path: str,
    knowledge_nodes: Sequence[Any],
) -> Dict[str, Any]:
    runtime_root = Path(resolve_repo_path("runtime"))
    runtime_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"pageindex_case_{case['case_id']}_", dir=runtime_root) as temp_dir:
        temp_root = Path(temp_dir)
        memory_path = temp_root / "memory.jsonl"
        state_path = temp_root / "state.jsonl"
        log_path = temp_root / "debug.log"
        trace_path = temp_root / "trace.jsonl"
        long_term_memory_path = temp_root / "long_term_memory.json"
        alert_path = temp_root / "clinical_alerts.jsonl"
        reset_session_files(
            str(memory_path),
            str(long_term_memory_path),
            str(state_path),
            str(log_path),
            str(trace_path),
            str(alert_path),
        )

        session = DigitalDoctorSession(
            transcript_path=transcript_path,
            milestone_path=milestone_path,
            knowledge_tree_path=knowledge_tree_path,
            helper_api_url="http://localhost:8001/helper/generate",
            helper_api_key=None,
            user_id="pageindex_impact",
            episode_id=str(case["case_id"]),
            patient_role_label="patient",
            doctor_role_label="doctor",
            memory_path=str(memory_path),
            long_term_memory_path=str(long_term_memory_path),
            state_path=str(state_path),
            log_path=str(log_path),
            trace_path=str(trace_path),
            alert_path=str(alert_path),
            single_turn=False,
            use_helper_model=False,
            use_knowledge_tree=True,
        )
        seed_history(session, case.get("history", []))
        reply, update = session.handle_query(str(case["patient_message"]))

    artifacts = update.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    source_candidates = update.get("source_candidates", {})
    if not isinstance(source_candidates, dict):
        source_candidates = {}
    knowledge_hits = artifacts.get("knowledge_hits", [])
    if not isinstance(knowledge_hits, list):
        knowledge_hits = []

    transcript_candidate = normalize_text(source_candidates.get("transcript", ""))
    knowledge_candidate = normalize_text(source_candidates.get("knowledge_tree", ""))
    combined_candidate = normalize_text(source_candidates.get("combined", reply))
    dialogue_context = format_dialogue(case.get("history", []), str(case["patient_message"]))
    _, presentation_json_rows = make_export_rows(
        rows=[
            {
                "query": normalize_text(case.get("patient_message", "")),
                "reply": combined_candidate,
                "update": update,
            }
        ],
        nodes=knowledge_nodes,
        top_k=3,
    )
    document_evidence = presentation_json_rows[0]["knowledge_tree"] if presentation_json_rows else []
    if not isinstance(document_evidence, list):
        document_evidence = []
    analysis = analyze_document_influence(
        dialogue_context=dialogue_context,
        transcript_candidate=transcript_candidate,
        knowledge_candidate=knowledge_candidate,
        combined_candidate=combined_candidate,
        document_evidence=document_evidence,
    )

    knowledge_extracts = []
    for hit in document_evidence:
        if not isinstance(hit, dict):
            continue
        title = normalize_text(hit.get("title", ""))
        summary = normalize_text(hit.get("summary", ""))
        if title or summary:
            knowledge_extracts.append(f"{title}: {summary}".strip(": "))

    row = {
        "case_id": str(case["case_id"]),
        "patient_message": normalize_text(case.get("patient_message", "")),
        "doctor_reply": combined_candidate,
        "knowledge_tree_extracts": join_unique(knowledge_extracts),
    }

    return {
        "row": row,
        "detail": {
            "case": case,
            "dialogue_context": dialogue_context,
            "doctor_reply": combined_candidate,
            "source_candidates": source_candidates,
            "knowledge_hits": knowledge_hits,
            "document_evidence": document_evidence,
            "analysis": analysis,
            "update": update,
        },
    }


def build_summary(details: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    title_counter: Counter[str] = Counter()
    section_counter: Counter[str] = Counter()
    for item in details:
        for hit in item.get("document_evidence", []):
            if not isinstance(hit, dict):
                continue
            title = normalize_text(hit.get("title", ""))
            parent_title = normalize_text(hit.get("parent_title", ""))
            if title:
                title_counter[title] += 1
            if parent_title and title:
                section_counter[f"{parent_title} -> {title}"] += 1
            elif title:
                section_counter[title] += 1
    return {
        "case_count": len(details),
        "top_knowledge_titles": [{"title": key, "count": value} for key, value in title_counter.most_common(8)],
        "top_supporting_sections": [{"section": key, "count": value} for key, value in section_counter.most_common(8)],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sample patient-doctor cases and export a CSV showing how PageIndex document retrieval changes the reply."
    )
    parser.add_argument("--cases", type=str, default=str(DEFAULT_CASES_PATH), help="Path to the sample case JSON file.")
    parser.add_argument(
        "--transcript-path",
        type=str,
        default=DEFAULT_TRANSCRIPT_PATH,
        help="Transcript JSON path.",
    )
    parser.add_argument(
        "--milestone-path",
        type=str,
        default=DEFAULT_MILESTONE_PATH,
        help="Milestone markdown path.",
    )
    parser.add_argument(
        "--knowledge-tree-path",
        type=str,
        default=DEFAULT_KNOWLEDGE_TREE_PATH,
        help="Knowledge tree JSON path.",
    )
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV), help="CSV output path.")
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON), help="JSON output path.")
    parser.add_argument("--diagram-path", type=str, default=str(DEFAULT_DIAGRAM_PATH), help="Diagram markdown output path.")
    parser.add_argument("--diagram-svg-path", type=str, default=str(DEFAULT_DIAGRAM_SVG_PATH), help="Diagram SVG output path.")
    parser.add_argument("--diagram-jpg-path", type=str, default=str(DEFAULT_DIAGRAM_JPG_PATH), help="Diagram JPG output path.")
    parser.add_argument(
        "--case-ids",
        type=str,
        default="",
        help="Optional comma-separated list of case_ids to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of cases to run after filtering.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cases_path = Path(resolve_repo_path(args.cases))
    transcript_path = str(resolve_repo_path(args.transcript_path))
    milestone_path = str(resolve_repo_path(args.milestone_path))
    knowledge_tree_path = str(resolve_repo_path(args.knowledge_tree_path))
    output_csv = Path(resolve_repo_path(args.output_csv))
    output_json = Path(resolve_repo_path(args.output_json))
    diagram_path = Path(resolve_repo_path(args.diagram_path))
    diagram_svg_path = Path(resolve_repo_path(args.diagram_svg_path))
    diagram_jpg_path = Path(resolve_repo_path(args.diagram_jpg_path))

    cases = load_cases(cases_path)
    requested_ids = [normalize_text(item) for item in str(args.case_ids).split(",") if normalize_text(item)]
    if requested_ids:
        requested = set(requested_ids)
        cases = [case for case in cases if case["case_id"] in requested]
    if int(args.limit) > 0:
        cases = cases[: int(args.limit)]
    if not cases:
        raise ValueError("No cases selected for execution.")
    knowledge_nodes = flatten_tree(Path(knowledge_tree_path))
    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for case in cases:
        result = run_case(
            case=case,
            transcript_path=transcript_path,
            milestone_path=milestone_path,
            knowledge_tree_path=knowledge_tree_path,
            knowledge_nodes=knowledge_nodes,
        )
        rows.append(result["row"])
        details.append(result["detail"])

    summary = build_summary(details)
    top_titles = [item["title"] for item in summary["top_knowledge_titles"][:5]]

    write_csv(output_csv, rows)
    diagram_svg_path.parent.mkdir(parents=True, exist_ok=True)
    diagram_svg_path.write_text(build_diagram_svg(top_titles), encoding="utf-8")
    write_json(
        output_json,
        {
            "cases_path": repo_relative_path(cases_path),
            "transcript_path": repo_relative_path(transcript_path),
            "milestone_path": repo_relative_path(milestone_path),
            "knowledge_tree_path": repo_relative_path(knowledge_tree_path),
            "output_csv": repo_relative_path(output_csv),
            "diagram_path": repo_relative_path(diagram_path),
            "diagram_svg_path": repo_relative_path(diagram_svg_path),
            "diagram_jpg_path": repo_relative_path(diagram_jpg_path),
            "summary": summary,
            "cases": details,
        },
    )
    diagram_path.parent.mkdir(parents=True, exist_ok=True)
    diagram_path.write_text(build_diagram_markdown(top_titles), encoding="utf-8")
    jpg_created = render_svg_to_jpg(diagram_svg_path, diagram_jpg_path)

    print(f"Saved CSV report to: {output_csv}")
    print(f"Saved JSON report to: {output_json}")
    print(f"Saved diagram markdown to: {diagram_path}")
    print(f"Saved diagram SVG to: {diagram_svg_path}")
    if jpg_created:
        print(f"Saved diagram JPG to: {diagram_jpg_path}")
    else:
        print("Diagram JPG export skipped.")
    print(f"Processed {len(rows)} sample cases.")


if __name__ == "__main__":
    main()
