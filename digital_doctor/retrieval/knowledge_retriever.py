from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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
    "do",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
}


@dataclass
class KnowledgeNode:
    node_id: str
    parent_id: Optional[str]
    title: str
    node_type: str
    summary: str
    keywords: List[str]
    clinical_uses: List[str]
    avoidances: List[str]
    canonical_topics: List[str]
    source_excerpt: str
    parent_title: str = ""
    chapter_title: str = ""


@dataclass
class KnowledgeHit:
    node: KnowledgeNode
    score: float


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9][a-z0-9_/-]*", text.lower())
    tokens = []
    for word in words:
        if len(word) <= 1 or word in STOPWORDS:
            continue
        token = word
        for suffix in ("ing", "ed", "es", "s"):
            if len(token) > 4 and token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        tokens.append(token)
    return tokens


def _score_overlap(query_tokens: Sequence[str], target_tokens: Sequence[str], weight: float) -> float:
    if not query_tokens or not target_tokens:
        return 0.0
    query_set = set(query_tokens)
    target_set = set(target_tokens)
    return float(len(query_set & target_set)) * weight


class KnowledgeTreeRetriever:
    def __init__(self, path: str):
        self.path = str(path)
        self.doc_title = ""
        self.root_summary = ""
        self.chapter_nodes: Dict[str, KnowledgeNode] = {}
        self.section_nodes: List[KnowledgeNode] = []
        self._load(Path(path))

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Knowledge tree not found: {path}")

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        root = payload.get("root", {})
        self.doc_title = str(payload.get("title") or root.get("title") or "").strip()
        self.root_summary = str(root.get("summary", "")).strip()

        for child in root.get("children", []):
            self._collect_nodes(child, parent_title=self.doc_title, chapter_title="")

    def _collect_nodes(self, raw: Dict[str, object], parent_title: str, chapter_title: str) -> None:
        node = self._build_node(raw, parent_title=parent_title, chapter_title=chapter_title)
        current_chapter = chapter_title
        if node.node_type == "chapter" or not chapter_title:
            current_chapter = node.title
            self.chapter_nodes[node.node_id] = node
        elif node.node_type != "document":
            node.chapter_title = chapter_title
            self.section_nodes.append(node)

        for child in raw.get("children", []):
            if not isinstance(child, dict):
                continue
            self._collect_nodes(child, parent_title=node.title, chapter_title=current_chapter)

    def _build_node(self, raw: Dict[str, object], parent_title: str, chapter_title: str) -> KnowledgeNode:
        return KnowledgeNode(
            node_id=str(raw.get("node_id", "")),
            parent_id=str(raw.get("parent_id")) if raw.get("parent_id") is not None else None,
            title=str(raw.get("title", "")).strip(),
            node_type=str(raw.get("node_type", "")).strip(),
            summary=str(raw.get("summary", "")).strip(),
            keywords=[str(item).strip() for item in raw.get("keywords", []) if str(item).strip()],
            clinical_uses=[str(item).strip() for item in raw.get("clinical_uses", []) if str(item).strip()],
            avoidances=[str(item).strip() for item in raw.get("avoidances", []) if str(item).strip()],
            canonical_topics=[str(item).strip() for item in raw.get("canonical_topics", []) if str(item).strip()],
            source_excerpt=str(raw.get("source_excerpt", "")).strip(),
            parent_title=parent_title,
            chapter_title=chapter_title,
        )

    def _score_node(self, weighted_query_tokens: Sequence[Tuple[str, float]], node: KnowledgeNode) -> float:
        if not weighted_query_tokens:
            return 0.0
        weighted_query: Dict[str, float] = {}
        for token, weight in weighted_query_tokens:
            weighted_query[token] = max(weighted_query.get(token, 0.0), weight)
        target_tokens = {
            "title": set(_tokenize(node.title)),
            "keywords": set(_tokenize(" ".join(node.keywords))),
            "canonical_topics": set(_tokenize(" ".join(node.canonical_topics))),
            "summary": set(_tokenize(node.summary)),
            "clinical_uses": set(_tokenize(" ".join(node.clinical_uses))),
            "avoidances": set(_tokenize(" ".join(node.avoidances))),
            "parent_title": set(_tokenize(node.parent_title)),
            "chapter_title": set(_tokenize(node.chapter_title)),
        }
        all_target_tokens = set().union(*target_tokens.values())
        score = 0.0
        matched_tokens = set()
        field_weights = {
            "title": 4.0,
            "keywords": 3.0,
            "canonical_topics": 2.5,
            "summary": 1.5,
            "clinical_uses": 1.25,
            "avoidances": 1.0,
            "parent_title": 0.75,
            "chapter_title": 0.75,
        }
        for token, query_weight in weighted_query.items():
            if token in all_target_tokens:
                matched_tokens.add(token)
            for field_name, target_set in target_tokens.items():
                if token in target_set:
                    score += query_weight * field_weights[field_name]
        if len(matched_tokens) < 2:
            return 0.0
        if node.node_type == "section":
            score += 0.1
        return score

    def retrieve(
        self,
        query: str,
        history: str = "",
        milestone_context: str = "",
        top_k_sections: int = 3,
    ) -> List[KnowledgeHit]:
        weighted_query_tokens: List[Tuple[str, float]] = []
        weighted_query_tokens.extend((token, 1.0) for token in _tokenize(query))
        weighted_query_tokens.extend((token, 0.35) for token in _tokenize(history))
        weighted_query_tokens.extend((token, 0.15) for token in _tokenize(milestone_context))
        if not weighted_query_tokens:
            return []

        hits: List[KnowledgeHit] = []
        for section in self.section_nodes:
            score = self._score_node(weighted_query_tokens, section)
            if score > 0:
                hits.append(KnowledgeHit(node=section, score=score))

        hits.sort(key=lambda item: item.score, reverse=True)
        if not hits:
            return []

        selected: List[KnowledgeHit] = []
        seen_titles = set()
        for hit in hits:
            key = (
                (hit.node.chapter_title or hit.node.parent_title).lower(),
                hit.node.parent_title.lower(),
                hit.node.title.lower(),
            )
            if key in seen_titles:
                continue
            seen_titles.add(key)
            selected.append(hit)
            if len(selected) >= top_k_sections:
                break
        return selected

    def format_block(self, hits: Sequence[KnowledgeHit], max_items: int = 3) -> str:
        if not hits:
            return ""

        chapter_summaries: List[Tuple[str, str]] = []
        seen_chapters = set()
        for hit in hits:
            chapter_title = (hit.node.chapter_title or hit.node.parent_title).strip()
            if not chapter_title:
                continue
            chapter_key = chapter_title.lower()
            if chapter_key in seen_chapters:
                continue
            seen_chapters.add(chapter_key)
            chapter_node = next(
                (node for node in self.chapter_nodes.values() if node.title.strip().lower() == chapter_key),
                None,
            )
            if chapter_node is not None:
                chapter_summaries.append((chapter_node.title, chapter_node.summary))

        lines = [f"External knowledge from {self.doc_title}:"]
        for title, summary in chapter_summaries[:1]:
            lines.append(f"- Chapter anchor: {title} | {summary}")

        for hit in hits[:max_items]:
            node = hit.node
            section_parent = node.parent_title.strip() or node.chapter_title.strip()
            lines.append(f"- Section: {section_parent} -> {node.title} | {node.summary}")
            if node.clinical_uses:
                lines.append(f"  Useful for: {'; '.join(node.clinical_uses[:2])}")
            if node.avoidances:
                lines.append(f"  Avoid: {'; '.join(node.avoidances[:2])}")

        return "\n".join(lines)
