"""Durable human-escalation outbox with optional webhook delivery."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Callable, Dict, Optional
from urllib import request

from ..paths import DEFAULT_ALERT_PATH, resolve_repo_path
from .risk import MoodAssessment


DeliveryCallback = Callable[[Dict[str, object]], None]


class ClinicalAlertNotifier:
    def __init__(
        self,
        path: str = DEFAULT_ALERT_PATH,
        webhook_url: Optional[str] = None,
        delivery_callback: Optional[DeliveryCallback] = None,
    ) -> None:
        self.path = str(resolve_repo_path(path))
        self.webhook_url = webhook_url or os.getenv("CLINICAL_ALERT_WEBHOOK_URL", "").strip()
        self.delivery_callback = delivery_callback

    def notify(
        self,
        assessment: MoodAssessment,
        user_text: str,
        user_id: str,
        episode_id: str,
        session_id: str,
    ) -> Dict[str, object]:
        record: Dict[str, object] = {
            "alert_id": uuid.uuid4().hex,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "patient_safety_escalation",
            "status": "queued_local",
            "user_id": user_id,
            "episode_id": episode_id,
            "session_id": session_id,
            "risk": assessment.to_dict(),
            "latest_patient_message": user_text[:4000],
            "requires_human_review": True,
        }
        try:
            if self.delivery_callback is not None:
                self.delivery_callback(record)
                record["status"] = "delivered_callback"
            elif self.webhook_url:
                body = json.dumps(record, ensure_ascii=False).encode("utf-8")
                req = request.Request(
                    self.webhook_url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=5) as response:
                    if 200 <= int(response.status) < 300:
                        record["status"] = "delivered_webhook"
                    else:
                        record["status"] = f"webhook_http_{response.status}"
        except Exception as exc:
            record["status"] = "delivery_failed_queued_local"
            record["delivery_error"] = type(exc).__name__

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
