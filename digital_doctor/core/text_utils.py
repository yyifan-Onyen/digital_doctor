from __future__ import annotations

import re


def extract_final(text: str) -> str:
    match = re.search(r"<\|channel\|>final<\|message\|>(.*?)(<\|end\|>|$)", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def clean_inline(text: str) -> str:
    return text.replace("**", "").replace("“", '"').replace("”", '"').strip()
