from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from docx import Document
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BOOK_TITLE = "Cognitive Therapy for OCD: A Guide for Professionals"
DEFAULT_DOC_ID = "wilhelm_steketee_2006"
REPO_DIR = Path(__file__).resolve().parents[2]

if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

from pageindex.page_index_md import md_to_tree

DEFAULT_OUTPUT_PATH = Path("data") / "knowledge_trees" / f"{DEFAULT_DOC_ID}.tree.json"
DEFAULT_CACHE_PATH = Path("runtime") / "cache" / "knowledge_trees" / f"{DEFAULT_DOC_ID}.pageindex.md"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


@dataclass
class BuildOptions:
    input_path: Path
    output_path: Path = DEFAULT_OUTPUT_PATH
    cache_path: Path = DEFAULT_CACHE_PATH
    model: str = DEFAULT_MODEL
    openai_api_key: Optional[str] = None
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    max_chapters: Optional[int] = None
    max_section_chars: int = 2400
    timeout_s: int = 120
    skip_llm: bool = False
    include_source_text: bool = True


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").replace("\u200b", " ")).strip()


def clip_text(text: str, limit: int) -> str:
    cleaned = normalize_space(text)
    if len(cleaned) <= limit:
        return cleaned
    keep = max(80, limit - 40)
    return cleaned[:keep].rstrip() + f"... [truncated {len(cleaned) - keep} chars]"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "node"


def repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(REPO_DIR).as_posix()
    except ValueError:
        return str(candidate)


def normalize_heading(text: str) -> str:
    cleaned = normalize_space(text.replace("HAN DOUTS", "HANDOUTS"))
    if not cleaned:
        return ""
    letters = [ch for ch in cleaned if ch.isalpha()]
    if letters and sum(ch.isupper() for ch in letters) / len(letters) >= 0.85:
        cleaned = cleaned.title()
    replacements = {
        "Ocd": "OCD",
        "Ct": "CT",
        "Erp": "ERP",
        "Dsm-Iv": "DSM-IV",
        "Nimh": "NIMH",
        "Obq-Ext": "OBQ-Ext",
        "Ocsrs": "OCSRS",
    }
    for before, after in replacements.items():
        cleaned = re.sub(rf"\b{re.escape(before)}\b", after, cleaned)
    return cleaned


def read_docx_paragraphs(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {path}")
    document = Document(str(path))
    paragraphs: List[str] = []
    for paragraph in document.paragraphs:
        text = normalize_space(paragraph.text)
        if text:
            paragraphs.append(text)
    return paragraphs


def is_page_artifact(text: str) -> bool:
    patterns = (
        r"^[0-9ivxlcdmIVXLCDM]+\s*Cognitive Therapy for OCD$",
        r"^.+Cognitive Therapy for OCD\s+\d+$",
        r"^[A-Za-z][A-Za-z0-9 ,/&()'’\-:]+\s+\d+$",
    )
    return any(re.match(pattern, text) for pattern in patterns)


def is_chapter_marker(text: str) -> bool:
    return bool(re.match(r"^Chapter\s+\d+$", text))


def is_terminal_marker(text: str) -> bool:
    return text in {"Appendix", "References"} or text.startswith("Appendix ")


def looks_like_body(text: str) -> bool:
    if len(text) > 100:
        return True
    if text.endswith((".", "!", "?")):
        return True
    return len(text.split()) >= 14


def heading_score(text: str) -> float:
    if not text or len(text) > 95 or is_page_artifact(text) or is_chapter_marker(text):
        return 0.0
    if text.endswith((".", ":", ";")):
        return 0.0
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*", text)
    if not words:
        return 0.0
    capitalized = sum(1 for word in words if word[0].isupper() or word.isupper())
    score = capitalized / len(words)
    if len(words) <= 10:
        score += 0.1
    if score > 0.85:
        score += 0.1
    return score


def collect_chapter_title(paragraphs: Sequence[str], start_idx: int) -> Tuple[str, int]:
    title_parts: List[str] = []
    idx = start_idx
    while idx < len(paragraphs):
        text = paragraphs[idx]
        if is_chapter_marker(text) or is_terminal_marker(text) or is_page_artifact(text):
            break
        if looks_like_body(text):
            break
        title_parts.append(text)
        idx += 1
        if len(title_parts) >= 3:
            break
    if not title_parts and idx < len(paragraphs):
        title_parts.append(paragraphs[idx])
        idx += 1
    title = normalize_heading(" ".join(title_parts))
    title = re.sub(r"\s+\d+$", "", title).strip()
    return title, idx


def find_chapters(paragraphs: Sequence[str]) -> List[Tuple[int, int, str, int]]:
    chapters: List[Tuple[int, int, str, int]] = []
    for idx, text in enumerate(paragraphs):
        if not is_chapter_marker(text):
            continue
        title, body_start = collect_chapter_title(paragraphs, idx + 1)
        lookahead = [item for item in paragraphs[body_start : min(body_start + 4, len(paragraphs))] if not is_page_artifact(item)]
        if not lookahead:
            continue
        if any(is_chapter_marker(item) for item in lookahead[:2]):
            continue
        if not any(looks_like_body(item) or heading_score(item) >= 0.7 for item in lookahead):
            continue
        chapter_no_match = re.search(r"\d+", text)
        if not chapter_no_match:
            continue
        chapter_no = int(chapter_no_match.group(0))
        if title:
            chapters.append((idx, chapter_no, title, body_start))

    if not chapters:
        raise RuntimeError("No chapter structure detected in the DOCX.")

    first_chapter_index = next((i for i, (_, chapter_no, _, _) in enumerate(chapters) if chapter_no == 1), 0)
    filtered = chapters[first_chapter_index:]
    if filtered:
        expected = filtered[0][1]
        contiguous: List[Tuple[int, int, str, int]] = []
        for item in filtered:
            if item[1] != expected:
                break
            contiguous.append(item)
            expected += 1
        if contiguous:
            return contiguous
    return filtered


def next_content_line(lines: Sequence[str], start_idx: int) -> str:
    for idx in range(start_idx, len(lines)):
        text = lines[idx]
        if not is_page_artifact(text):
            return text
    return ""


def collect_sections(paragraphs: Sequence[str]) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    intro_buffer: List[str] = []
    current_title: Optional[str] = None
    current_buffer: List[str] = []

    def flush_current() -> None:
        nonlocal current_title, current_buffer
        if current_title and current_buffer:
            sections.append((current_title, normalize_space(" ".join(current_buffer))))
        current_title = None
        current_buffer = []

    for idx, text in enumerate(paragraphs):
        if is_page_artifact(text):
            continue
        next_text = next_content_line(paragraphs, idx + 1)
        if heading_score(text) >= 0.7 and next_text and looks_like_body(next_text):
            if current_title is None and intro_buffer:
                sections.append(("Overview", normalize_space(" ".join(intro_buffer))))
                intro_buffer = []
            flush_current()
            current_title = normalize_heading(text)
            continue
        if current_title is None:
            intro_buffer.append(text)
        else:
            current_buffer.append(text)

    flush_current()
    if intro_buffer and not sections:
        sections.append(("Overview", normalize_space(" ".join(intro_buffer))))
    return [(title, body) for title, body in sections if title and body]


def docx_to_markdown(path: Path, cache_path: Path, max_chapters: Optional[int]) -> Tuple[Path, str, int]:
    paragraphs = read_docx_paragraphs(path)
    chapters = find_chapters(paragraphs)
    if max_chapters is not None:
        chapters = chapters[:max_chapters]

    lines: List[str] = [f"# {DEFAULT_BOOK_TITLE}", ""]
    for chapter_idx, (_, _, chapter_title, body_start) in enumerate(chapters):
        body_end = len(paragraphs)
        if chapter_idx + 1 < len(chapters):
            body_end = chapters[chapter_idx + 1][0]
        for idx in range(body_start, body_end):
            if is_terminal_marker(paragraphs[idx]):
                body_end = idx
                break
        chapter_paragraphs = paragraphs[body_start:body_end]
        sections = collect_sections(chapter_paragraphs)
        if not sections:
            continue
        lines.extend([f"## {chapter_title}", ""])
        for section_title, section_text in sections:
            lines.extend([f"### {section_title}", "", section_text, ""])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return cache_path, DEFAULT_BOOK_TITLE, len(paragraphs)


def prepare_markdown_input(options: BuildOptions) -> Tuple[Path, str, int]:
    input_path = options.input_path.resolve()
    if input_path.suffix.lower() in {".md", ".markdown"}:
        return input_path, normalize_heading(input_path.stem), 0
    if input_path.suffix.lower() != ".docx":
        raise ValueError("PageIndex integration currently supports DOCX and Markdown inputs.")
    return docx_to_markdown(input_path, options.cache_path.resolve(), options.max_chapters)


def derive_keywords(*parts: str, limit: int = 6) -> List[str]:
    keywords: List[str] = []
    seen = set()
    for token in re.findall(r"[A-Za-z][A-Za-z'/-]*", " ".join(parts)):
        lowered = token.lower()
        if len(lowered) <= 2 or lowered in STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(token.upper() if token.upper() in {"OCD", "CT", "ERP"} else lowered)
        if len(keywords) >= limit:
            break
    return keywords


def default_summary(title: str, text: str, fallback: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", normalize_space(text))
    for sentence in sentences:
        if sentence:
            return clip_text(sentence, 280)
    return f"{fallback} covering {title}." if title else fallback


def normalize_pageindex_nodes(
    raw_nodes: Sequence[Dict[str, Any]],
    parent_id: str,
    depth: int,
    doc_id: str,
    order_counter: List[int],
    include_source_text: bool,
) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for raw in raw_nodes:
        order_counter[0] += 1
        title = normalize_heading(str(raw.get("title", "")).strip()) or f"Untitled {order_counter[0]}"
        text = normalize_space(str(raw.get("text", "")))
        summary = normalize_space(str(raw.get("summary") or raw.get("prefix_summary") or ""))
        node_id = f"doc.{doc_id}.node_{str(raw.get('node_id', order_counter[0])).zfill(4)}"
        children = normalize_pageindex_nodes(
            raw_nodes=raw.get("nodes", []),
            parent_id=node_id,
            depth=depth + 1,
            doc_id=doc_id,
            order_counter=order_counter,
            include_source_text=include_source_text,
        )
        if not summary:
            summary_basis = text or " ".join(child.get("summary", "") for child in children)
            summary = default_summary(title, summary_basis, fallback=f"Section on {title}")

        payload: Dict[str, Any] = {
            "node_id": node_id,
            "parent_id": parent_id,
            "title": title,
            "raw_title": str(raw.get("title", title)),
            "node_type": "chapter" if depth == 1 else "section",
            "order": order_counter[0],
            "summary": summary,
            "keywords": derive_keywords(title, summary),
            "clinical_uses": [],
            "avoidances": [],
            "canonical_topics": derive_keywords(title, summary, limit=8),
            "source_span": {
                "paragraph_start": int(raw.get("line_num", order_counter[0])),
                "paragraph_end": int(raw.get("line_num", order_counter[0])),
            },
            "source_excerpt": clip_text(text, 400) if text else "",
            "text_char_count": len(text),
            "children": children,
        }
        if include_source_text and text:
            payload["source_text"] = text
        nodes.append(payload)
    return nodes


def count_nodes(nodes: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for node in nodes:
        total += 1
        total += count_nodes(node.get("children", []))
    return total


def build_knowledge_tree(options: BuildOptions) -> Dict[str, Any]:
    input_path = options.input_path.resolve()
    markdown_path, doc_title, paragraph_count = prepare_markdown_input(options)
    doc_id = slugify(input_path.stem) or DEFAULT_DOC_ID

    api_key = options.openai_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not options.skip_llm and not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for PageIndex knowledge-tree generation.")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ.setdefault("CHATGPT_API_KEY", api_key)
    if options.openai_base_url:
        os.environ["OPENAI_API_BASE"] = options.openai_base_url

    raw_pageindex = asyncio.run(
        md_to_tree(
            md_path=str(markdown_path),
            if_thinning=False,
            min_token_threshold=5000,
            if_add_node_summary="no" if options.skip_llm else "yes",
            summary_token_threshold=1,
            model=options.model,
            if_add_doc_description="no" if options.skip_llm else "yes",
            if_add_node_text="yes",
            if_add_node_id="yes",
        )
    )

    raw_output_path = markdown_path.with_suffix(".raw.json")
    raw_output_path.write_text(json.dumps(raw_pageindex, ensure_ascii=False, indent=2), encoding="utf-8")

    structure = raw_pageindex.get("structure", [])
    if (
        len(structure) == 1
        and normalize_heading(str(structure[0].get("title", "")).strip()) == doc_title
        and structure[0].get("nodes")
    ):
        structure = structure[0]["nodes"]

    order_counter = [0]
    root_children = normalize_pageindex_nodes(
        raw_nodes=structure,
        parent_id=f"doc.{doc_id}",
        depth=1,
        doc_id=doc_id,
        order_counter=order_counter,
        include_source_text=options.include_source_text,
    )
    child_summaries = " ".join(node.get("summary", "") for node in root_children)
    doc_description = normalize_space(str(raw_pageindex.get("doc_description", "")).strip())

    payload = {
        "doc_id": doc_id,
        "title": doc_title,
        "source_path": repo_relative(input_path),
        "builder": {
            "name": "PageIndex",
            "model": options.model,
            "input_format": input_path.suffix.lower().lstrip("."),
            "markdown_path": repo_relative(markdown_path),
            "pageindex_raw_path": repo_relative(raw_output_path),
            "pageindex_source": "repo/pageindex (unmodified upstream snapshot)",
        },
        "stats": {
            "paragraph_count": paragraph_count,
            "chapter_count": len(root_children),
            "node_count": count_nodes(root_children) + 1,
            "section_count": max(0, count_nodes(root_children) - len(root_children)),
        },
        "root": {
            "node_id": f"doc.{doc_id}",
            "parent_id": None,
            "title": doc_title,
            "raw_title": doc_title,
            "node_type": "document",
            "order": 1,
            "summary": doc_description or default_summary(doc_title, child_summaries, f"Document tree for {doc_title}."),
            "keywords": derive_keywords(doc_title, child_summaries),
            "clinical_uses": [],
            "avoidances": [],
            "canonical_topics": derive_keywords(doc_title, child_summaries, limit=8),
            "source_span": {
                "paragraph_start": 1,
                "paragraph_end": max(paragraph_count, 1),
            },
            "source_excerpt": "",
            "text_char_count": 0,
            "children": root_children,
        },
    }
    return payload


def write_tree(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "BuildOptions",
    "DEFAULT_CACHE_PATH",
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_PATH",
    "build_knowledge_tree",
    "write_tree",
]
