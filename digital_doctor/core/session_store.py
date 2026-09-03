from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..paths import DEFAULT_LOG_PATH, DEFAULT_MEMORY_PATH, resolve_repo_path


def reset_session_files(*paths: str) -> None:
    for path in paths:
        resolved = str(resolve_repo_path(path))
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(resolved):
            os.remove(resolved)


@dataclass
class Turn:
    user_id: str
    episode_id: str
    role: str
    text: str
    kind: str = ""


@dataclass
class LongTermMemory:
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    reminders: List[str] = field(default_factory=list)
    summarized_turn_count: int = 0
    compaction_count: int = 0
    updated_at: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "summary": self.summary,
            "topics": list(self.topics),
            "reminders": list(self.reminders),
            "summarized_turn_count": self.summarized_turn_count,
            "compaction_count": self.compaction_count,
            "updated_at": self.updated_at,
        }


MemorySummarizer = Callable[[str, Sequence[Tuple[str, str]]], Mapping[str, object] | str]

_FORGETTING_RE = re.compile(
    r"\b(i(?:\s+\w+){0,3}\s+(forgot|forget|can'?t remember|cannot remember|do not remember|don'?t remember)|"
    r"i'?m(?:\s+\w+){0,3}\s+not(?:\s+\w+){0,3}\s+remembering|"
    r"remind me|what did we (say|discuss|decide|talk about)|what was (that|it)|"
    r"you mentioned earlier)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")
_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "before",
    "but", "can", "could", "did", "does", "for", "from", "have", "how", "into",
    "just", "more", "not", "that", "the", "then", "there", "they", "this", "was",
    "what", "when", "where", "which", "with", "would", "you", "your",
}


class MemoryStore:
    """Append-only turn archive plus an incrementally compacted long-term memory."""

    def __init__(
        self,
        path: str = DEFAULT_MEMORY_PATH,
        window_size: int = 8,
        default_user_id: str = "user",
        summary_path: Optional[str] = None,
        summary_threshold_chars: Optional[int] = None,
    ):
        self.path = str(resolve_repo_path(path))
        self.window_size = window_size
        self.default_user_id = default_user_id
        default_summary_path = str(Path(self.path).with_suffix(".long_term.json"))
        self.summary_path = str(resolve_repo_path(summary_path or default_summary_path))
        configured_threshold = summary_threshold_chars
        if configured_threshold is None:
            configured_threshold = int(os.getenv("MEMORY_SUMMARY_THRESHOLD_CHARS", "12000"))
        self.summary_threshold_chars = max(1, configured_threshold)
        self.turns: List[Turn] = []
        self.long_term: Dict[str, LongTermMemory] = {}
        if os.path.exists(self.path):
            self._load()
        if os.path.exists(self.summary_path):
            self._load_long_term()

    @staticmethod
    def _key(user_id: str, episode_id: str) -> str:
        return f"{user_id}\u001f{episode_id}"

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                self.turns.append(
                    Turn(
                        user_id=str(item.get("user_id", self.default_user_id)),
                        episode_id=item["episode_id"],
                        role=item["role"],
                        text=item["text"],
                        kind=str(item.get("kind", "")),
                    )
                )

    def _load_long_term(self) -> None:
        try:
            with open(self.summary_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        records = payload.get("records", {}) if isinstance(payload, dict) else {}
        if not isinstance(records, dict):
            return
        for key, raw in records.items():
            if not isinstance(raw, dict):
                continue
            raw_topics = raw.get("topics", [])
            raw_reminders = raw.get("reminders", [])
            self.long_term[str(key)] = LongTermMemory(
                summary=str(raw.get("summary", "")),
                topics=(
                    [str(item) for item in raw_topics if str(item).strip()][:30]
                    if isinstance(raw_topics, list)
                    else []
                ),
                reminders=(
                    [str(item) for item in raw_reminders if str(item).strip()][:20]
                    if isinstance(raw_reminders, list)
                    else []
                ),
                summarized_turn_count=max(0, int(raw.get("summarized_turn_count", 0))),
                compaction_count=max(0, int(raw.get("compaction_count", 0))),
                updated_at=str(raw.get("updated_at", "")),
            )

    def _save_long_term(self) -> None:
        os.makedirs(os.path.dirname(self.summary_path), exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.utcnow().isoformat(),
            "records": {key: value.to_dict() for key, value in self.long_term.items()},
        }
        temporary = f"{self.summary_path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, self.summary_path)

    def append(self, turn: Turn) -> None:
        self.turns.append(turn)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        record = {
            "user_id": turn.user_id,
            "episode_id": turn.episode_id,
            "role": turn.role,
            "text": turn.text,
            "kind": turn.kind,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def window(self, user_id: str, episode_id: str) -> List[Tuple[str, str]]:
        items = [
            (turn.role, turn.text)
            for turn in self.turns
            if turn.user_id == user_id and turn.episode_id == episode_id
        ]
        return items[-self.window_size :]

    def clinical_user_turn_count(self, user_id: str, episode_id: str, patient_role: str) -> int:
        return sum(
            1
            for turn in self.turns
            if turn.user_id == user_id
            and turn.episode_id == episode_id
            and turn.role == patient_role
            and turn.kind in {"", "analysis", "clinical"}
        )

    def _session_turns(self, user_id: str, episode_id: str) -> List[Turn]:
        return [
            turn
            for turn in self.turns
            if turn.user_id == user_id and turn.episode_id == episode_id
        ]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.lower()
            for token in _TOKEN_RE.findall(text)
            if token.lower() not in _STOP_WORDS
        }

    def recall(self, user_id: str, episode_id: str, query: str, limit: int = 3) -> List[str]:
        """Return a few relevant older details, with a recency bias."""
        turns = self._session_turns(user_id, episode_id)
        older = turns[:-self.window_size] if len(turns) > self.window_size else []
        forgetting = bool(_FORGETTING_RE.search(query))
        if forgetting and not older:
            older = turns[:-2]
        query_tokens = self._tokens(query)
        ranked: List[Tuple[float, str]] = []
        for idx, turn in enumerate(older):
            overlap = len(query_tokens & self._tokens(turn.text))
            if overlap or forgetting:
                score = float(overlap * 10) + (idx / max(1, len(older)))
                ranked.append((score, f"{turn.role}: {turn.text[:1200]}"))
        ranked.sort(key=lambda item: item[0], reverse=True)
        recalled = [text for _, text in ranked[:limit]]

        state = self.long_term.get(self._key(user_id, episode_id))
        if state:
            for topic in state.topics:
                if len(recalled) >= limit:
                    break
                if forgetting or query_tokens & self._tokens(topic):
                    recalled.append(f"Long-term topic: {topic}")
        return recalled

    def context(self, user_id: str, episode_id: str, query: str = "") -> Dict[str, object]:
        state = self.long_term.get(self._key(user_id, episode_id), LongTermMemory())
        recent = self.window(user_id, episode_id)
        recalled = self.recall(user_id, episode_id, query)
        reminder_needed = bool(_FORGETTING_RE.search(query)) and bool(
            recalled or state.summary or state.reminders
        )
        blocks: List[str] = []
        if state.summary:
            blocks.append(f"Long-term dialogue summary:\n{state.summary}")
        if recalled:
            blocks.append("Relevant earlier details:\n" + "\n".join(f"- {item}" for item in recalled))
        if state.reminders and reminder_needed:
            blocks.append("Prior plans or reminders:\n" + "\n".join(f"- {item}" for item in state.reminders[:5]))
        if recent:
            recent_text = "\n".join(f"{speaker}: {text[:2000]}" for speaker, text in recent)
            blocks.append(f"Recent dialogue:\n{recent_text}")
        if reminder_needed:
            blocks.append(
                "Continuity instruction: the patient may have forgotten prior context; "
                "briefly and naturally remind them of the relevant earlier discussion without claiming facts not present above."
            )
        return {
            "rendered": "\n\n".join(blocks),
            "long_term_summary": state.summary,
            "topics": list(state.topics),
            "reminders": list(state.reminders),
            "recalled": recalled,
            "reminder_needed": reminder_needed,
            "recent_turn_count": len(recent),
            "archived_turn_count": len(self._session_turns(user_id, episode_id)),
            "compaction_count": state.compaction_count,
        }

    def compact_if_needed(
        self,
        user_id: str,
        episode_id: str,
        summarizer: MemorySummarizer,
    ) -> Dict[str, object]:
        """Incrementally summarize all turns added since the previous compaction."""
        turns = self._session_turns(user_id, episode_id)
        key = self._key(user_id, episode_id)
        state = self.long_term.get(key, LongTermMemory())
        start = min(state.summarized_turn_count, len(turns))
        pending = turns[start:]
        pending_chars = sum(len(turn.text) for turn in pending)
        if not pending or pending_chars < self.summary_threshold_chars:
            return {
                "compacted": False,
                "pending_chars": pending_chars,
                "threshold_chars": self.summary_threshold_chars,
                "summarized_turn_count": state.summarized_turn_count,
                "compaction_count": state.compaction_count,
            }

        dialogue = [(turn.role, turn.text) for turn in pending]
        try:
            result = summarizer(state.summary, dialogue)
        except Exception:
            result = self._fallback_summary(state.summary, dialogue)
        if isinstance(result, str):
            normalized: Mapping[str, object] = {"summary": result}
        else:
            normalized = result

        summary = str(normalized.get("summary", "")).strip()
        if not summary:
            normalized = self._fallback_summary(state.summary, dialogue)
            summary = str(normalized.get("summary", "")).strip()
        topics = self._merge_unique(state.topics, normalized.get("topics", []), limit=30)
        reminders = self._merge_unique(state.reminders, normalized.get("reminders", []), limit=20)
        state.summary = summary[:8000]
        state.topics = topics
        state.reminders = reminders
        state.summarized_turn_count = len(turns)
        state.compaction_count += 1
        state.updated_at = datetime.utcnow().isoformat()
        self.long_term[key] = state
        self._save_long_term()
        return {
            "compacted": True,
            "pending_chars": 0,
            "threshold_chars": self.summary_threshold_chars,
            "summarized_turn_count": state.summarized_turn_count,
            "compaction_count": state.compaction_count,
            "summary_chars": len(state.summary),
            "topics": list(state.topics),
        }

    def update_topics(
        self,
        user_id: str,
        episode_id: str,
        topics: Sequence[str],
        reminders: Sequence[str] = (),
    ) -> Dict[str, object]:
        """Continuously persist the structured topic/reminder ledger between compactions."""
        key = self._key(user_id, episode_id)
        state = self.long_term.get(key, LongTermMemory())
        merged_topics = self._merge_unique(state.topics, list(topics), limit=30)
        merged_reminders = self._merge_unique(state.reminders, list(reminders), limit=20)
        changed = merged_topics != state.topics or merged_reminders != state.reminders
        if changed:
            state.topics = merged_topics
            state.reminders = merged_reminders
            state.updated_at = datetime.utcnow().isoformat()
            self.long_term[key] = state
            self._save_long_term()
        return {
            "changed": changed,
            "topics": list(state.topics),
            "reminders": list(state.reminders),
        }

    @staticmethod
    def _merge_unique(existing: Sequence[str], incoming: object, limit: int) -> List[str]:
        candidates = list(existing)
        if isinstance(incoming, list):
            candidates.extend(str(item).strip() for item in incoming)
        result: List[str] = []
        seen = set()
        for item in candidates:
            normalized = item.strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            result.append(normalized[:300])
        return result[-limit:]

    def _fallback_summary(
        self, existing_summary: str, dialogue: Sequence[Tuple[str, str]]
    ) -> Mapping[str, object]:
        lines = [f"{role}: {text.strip()}" for role, text in dialogue if text.strip()]
        combined = "\n".join(filter(None, [existing_summary.strip(), *lines]))
        token_counts = Counter(
            token
            for line in lines
            for token in self._tokens(line)
        )
        topics = [token for token, _ in token_counts.most_common(8)]
        return {
            "summary": combined[-8000:],
            "topics": topics,
            "reminders": [],
        }


def log_debug(message: str, path: str = DEFAULT_LOG_PATH, stage: str | None = None) -> None:
    timestamp = datetime.utcnow().isoformat()
    prefix = f"[{stage}] " if stage else ""
    resolved = str(resolve_repo_path(path))
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    with open(resolved, "a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {prefix}{message}\n")
