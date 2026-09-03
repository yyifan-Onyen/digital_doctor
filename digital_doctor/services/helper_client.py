from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional


class HelperApiClient:
    def __init__(self, api_url: str, api_key: Optional[str] = None, timeout_s: int = 30):
        self.api_url = api_url
        self.api_key = api_key
        self.timeout_s = timeout_s

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.api_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Helper API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Helper API unreachable: {exc}") from exc

        try:
            response_payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Helper API returned non-JSON: {body}") from exc

        text = response_payload.get("text")
        if text is None:
            raise RuntimeError(f"Helper API response missing 'text': {response_payload}")
        return str(text)
