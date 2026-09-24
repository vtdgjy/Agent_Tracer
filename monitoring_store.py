"""Independent runtime monitoring event store.

Monitoring events deliberately remain separate from trace spans.  They can be
correlated later through ``trace_id``, ``session_id`` and a time window, but a
missing trace must never prevent infrastructure/runtime telemetry ingestion.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4


MONITORING_LAYERS = frozenset({"hardware", "virtualization", "communication", "application"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_monitoring_event(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one monitoring event."""
    if not isinstance(raw, dict):
        raise ValueError("event must be a JSON object")

    system_id = str(raw.get("system_id") or "").strip()
    if not system_id:
        raise ValueError("system_id is required")

    layer = str(raw.get("layer") or "").strip().lower()
    if layer not in MONITORING_LAYERS:
        raise ValueError(f"layer must be one of {sorted(MONITORING_LAYERS)}")

    event_type = str(raw.get("event_type") or "measurement").strip()
    if not event_type:
        raise ValueError("event_type must not be empty")

    timestamp = str(raw.get("timestamp") or _utc_now()).strip()
    try:
        _parse_timestamp(timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be ISO-8601") from exc

    measurement = raw.get("measurement", {})
    labels = raw.get("labels", {})
    correlation = raw.get("correlation", {})
    for name, value in (
        ("measurement", measurement),
        ("labels", labels),
        ("correlation", correlation),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object")

    return {
        "event_id": str(raw.get("event_id") or uuid4()),
        "timestamp": timestamp,
        "system_id": system_id,
        "layer": layer,
        "event_type": event_type,
        "measurement": deepcopy(measurement),
        "labels": deepcopy(labels),
        "correlation": deepcopy(correlation),
    }


class MonitoringEventStore:
    """Bounded, thread-safe store used by the prototype ingestion API."""

    def __init__(self, max_events: int = 20_000) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = RLock()

    def append_batch(self, raw_events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_events):
            try:
                accepted.append(normalize_monitoring_event(raw))
            except ValueError as exc:
                rejected.append({"index": index, "error": str(exc)})
        with self._lock:
            self._events.extend(accepted)
        return deepcopy(accepted), rejected

    def query(
        self,
        system_id: str,
        *,
        layer: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        start_ts = _parse_timestamp(start) if start else None
        end_ts = _parse_timestamp(end) if end else None
        bounded_limit = min(max(int(limit), 1), 5_000)
        with self._lock:
            candidates = list(self._events)

        result: list[dict[str, Any]] = []
        for event in reversed(candidates):
            if event["system_id"] != system_id:
                continue
            if layer and event["layer"] != layer:
                continue
            timestamp = _parse_timestamp(event["timestamp"])
            if start_ts and timestamp < start_ts:
                continue
            if end_ts and timestamp > end_ts:
                continue
            result.append(event)
            if len(result) >= bounded_limit:
                break
        result.reverse()
        return deepcopy(result)

    def snapshot(self, system_id: str) -> dict[str, Any]:
        events = self.query(system_id, limit=5_000)
        latest: dict[str, dict[str, Any]] = {}
        for event in events:
            latest[event["layer"]] = event
        return {
            "system_id": system_id,
            "generated_at": _utc_now(),
            "event_count": len(events),
            "layers": {layer: latest.get(layer) for layer in sorted(MONITORING_LAYERS)},
            "source": "independent-monitoring-events",
        }


monitoring_event_store = MonitoringEventStore()
