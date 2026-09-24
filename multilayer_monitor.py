"""Four-layer monitoring aggregation for Agent Tracer.

The module intentionally uses the Python standard library only.  ``psutil`` is
used when it is already installed, but it is not required for the web demo.
Trace-derived values are kept separate from host snapshot values so the UI can
state where every observation came from.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import re
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LAYER_IDS = ("hardware", "virtualization", "communication", "application")
COMMUNICATION_EDGE_TYPES = {"Call", "Observation", "Retry", "AgentSpawn", "SubagentReturn"}
COMMUNICATION_SPAN_KINDS = {"client", "server", "producer", "consumer"}


def _round(value: Any, digits: int = 2) -> float | None:
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, digits)
    except (TypeError, ValueError):
        return None


def _percent(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return _round(100.0 * numerator / denominator, 1)


def _bytes_label(value: int | float | None) -> str:
    if value is None:
        return "不可用"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _metric(key: str, label: str, value: Any, unit: str = "", source: str = "") -> dict[str, Any]:
    return {"key": key, "label": label, "value": value, "unit": unit, "source": source}


def _status_from_percent(value: float | None, warning: float, critical: float) -> str:
    if value is None:
        return "unknown"
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "healthy"


def _worst_status(*statuses: str) -> str:
    rank = {"unknown": 0, "healthy": 1, "warning": 2, "critical": 3}
    return max(statuses or ("unknown",), key=lambda item: rank.get(item, 0))


def _windows_cpu_times() -> tuple[int, int, int] | None:
    if os.name != "nt":
        return None

    class FILETIME(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    idle = FILETIME()
    kernel = FILETIME()
    user = FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None

    def value(item: FILETIME) -> int:
        return (int(item.high) << 32) | int(item.low)

    return value(idle), value(kernel), value(user)


def _linux_cpu_times() -> tuple[int, int, int] | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        return idle, total, 0
    except (OSError, ValueError, IndexError):
        return None


def _cpu_percent(sample_interval: float = 0.06) -> float | None:
    """Return a short host CPU utilization sample without a hard psutil dependency."""
    try:
        import psutil  # type: ignore

        return _round(psutil.cpu_percent(interval=max(0.0, sample_interval)), 1)
    except ImportError:
        pass

    reader = _windows_cpu_times if os.name == "nt" else _linux_cpu_times
    before = reader()
    if before is None:
        return None
    time.sleep(max(0.02, sample_interval))
    after = reader()
    if after is None:
        return None
    idle_delta = after[0] - before[0]
    total_delta = (after[1] - before[1]) + (after[2] - before[2])
    if total_delta <= 0:
        return None
    return _round(100.0 * (1.0 - idle_delta / total_delta), 1)


def _memory_snapshot() -> tuple[int | None, int | None, float | None]:
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        return int(mem.total), int(mem.used), _round(mem.percent, 1)
    except ImportError:
        pass

    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32),
                ("memory_load", ctypes.c_uint32),
                ("total_phys", ctypes.c_uint64),
                ("avail_phys", ctypes.c_uint64),
                ("total_page_file", ctypes.c_uint64),
                ("avail_page_file", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64),
                ("avail_virtual", ctypes.c_uint64),
                ("avail_extended_virtual", ctypes.c_uint64),
            ]

        state = MEMORYSTATUSEX()
        state.length = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            used = int(state.total_phys - state.avail_phys)
            return int(state.total_phys), used, float(state.memory_load)

    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
        total = values["MemTotal"]
        available = values.get("MemAvailable", values.get("MemFree", 0))
        used = total - available
        return total, used, _percent(used, total)
    except (OSError, ValueError, KeyError):
        return None, None, None


def collect_hardware_snapshot(sample_interval: float = 0.06) -> dict[str, Any]:
    """Collect a lightweight host snapshot suitable for the monitoring UI."""
    cpu_percent = _cpu_percent(sample_interval)
    memory_total, memory_used, memory_percent = _memory_snapshot()
    disk_root = Path(os.environ.get("SystemDrive", "C:") + os.sep) if os.name == "nt" else Path("/")
    try:
        disk = shutil.disk_usage(disk_root)
        disk_percent = _percent(disk.used, disk.total)
        disk_total, disk_used = disk.total, disk.used
    except OSError:
        disk_total = disk_used = None
        disk_percent = None

    status = _worst_status(
        _status_from_percent(cpu_percent, 80, 95),
        _status_from_percent(memory_percent, 80, 95),
        _status_from_percent(disk_percent, 85, 95),
    )
    highlights = []
    if cpu_percent is None:
        highlights.append("CPU 利用率采集不可用，仍提供核心数信息")
    if status in {"warning", "critical"}:
        highlights.append("检测到资源利用率超过阈值")
    if not highlights:
        highlights.append("主机资源处于正常阈值范围")

    return {
        "id": "hardware",
        "name": "硬件层",
        "status": status,
        "telemetry_available": any(value is not None for value in (cpu_percent, memory_percent, disk_percent)),
        "source": "本机实时采样",
        "metrics": [
            _metric("cpu_percent", "CPU 利用率", cpu_percent, "%", "host"),
            _metric("cpu_count", "逻辑 CPU", os.cpu_count(), "核", "host"),
            _metric("memory_percent", "内存利用率", memory_percent, "%", "host"),
            _metric("memory_used", "已用内存", _bytes_label(memory_used), "", "host"),
            _metric("memory_total", "总内存", _bytes_label(memory_total), "", "host"),
            _metric("disk_percent", "磁盘利用率", disk_percent, "%", "host"),
            _metric("disk_used", "已用磁盘", _bytes_label(disk_used), "", "host"),
            _metric("disk_total", "磁盘容量", _bytes_label(disk_total), "", "host"),
        ],
        "highlights": highlights,
        "details": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }


def _walk_spans(spans: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for span in spans or []:
        if not isinstance(span, dict):
            continue
        yield span
        yield from _walk_spans(span.get("child_spans") or [])


def _resource_attributes(spans: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for span in spans:
        attrs = span.get("resource_attributes") or {}
        if isinstance(attrs, dict):
            for key, value in attrs.items():
                if value not in (None, ""):
                    merged.setdefault(str(key), value)
    return merged


def _virtualization_layer(spans: list[dict[str, Any]]) -> dict[str, Any]:
    resource_attrs = _resource_attributes(spans)
    interesting_prefixes = ("container.", "k8s.", "cloud.", "host.", "process.", "service.", "deployment.")
    context = {key: value for key, value in resource_attrs.items() if key.startswith(interesting_prefixes)}

    runtime = "物理机/未知"
    if any(key.startswith("k8s.") for key in context) or os.getenv("KUBERNETES_SERVICE_HOST"):
        runtime = "Kubernetes"
    elif any(key.startswith("container.") for key in context) or Path("/.dockerenv").exists():
        runtime = "Container"
    elif any(key.startswith("cloud.") for key in context):
        runtime = "Cloud VM"
    elif resource_attrs.get("host.id") or resource_attrs.get("host.name"):
        runtime = "Host/VM"

    services = sorted(
        {
            str((span.get("resource_attributes") or {}).get("service.name"))
            for span in spans
            if (span.get("resource_attributes") or {}).get("service.name")
        }
    )
    containers = {
        str(value)
        for key, value in context.items()
        if key in {"container.id", "container.name", "k8s.pod.uid", "k8s.pod.name"}
    }
    telemetry_available = bool(context)
    highlights = [f"识别运行环境：{runtime}"]
    if telemetry_available:
        highlights.append(f"保留 {len(context)} 项 OTLP 资源属性，可关联容器、Pod、主机和服务")
    else:
        highlights.append("Trace 未携带容器/Kubernetes/云资源属性")

    return {
        "id": "virtualization",
        "name": "虚拟化层",
        "status": "healthy" if telemetry_available else "unknown",
        "telemetry_available": telemetry_available,
        "source": "OTLP resource attributes + 运行环境探测",
        "metrics": [
            _metric("runtime", "运行环境", runtime, "", "resource"),
            _metric("service_count", "服务数", len(services), "", "trace"),
            _metric("container_count", "容器/Pod 数", len(containers), "", "resource"),
            _metric("resource_attribute_count", "资源属性", len(context), "项", "resource"),
        ],
        "highlights": highlights,
        "details": {"services": services, "resource_attributes": context},
    }


_DURATION_PATTERN = re.compile(
    r"^P(?:[0-9.]+D)?T(?:(?P<h>[0-9.]+)H)?(?:(?P<m>[0-9.]+)M)?(?:(?P<s>[0-9.]+)S)?$",
    re.IGNORECASE,
)


def _duration_ms(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return _round(float(value) * 1000.0, 3)
    text = str(value or "").strip()
    if not text:
        return None
    match = _DURATION_PATTERN.match(text)
    if match:
        hours = float(match.group("h") or 0)
        minutes = float(match.group("m") or 0)
        seconds = float(match.group("s") or 0)
        return _round((hours * 3600 + minutes * 60 + seconds) * 1000.0, 3)
    try:
        return _round(float(text) * 1000.0, 3)
    except ValueError:
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return _round(ordered[0], 2)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return _round(ordered[lower], 2)
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return _round(interpolated, 2)


def _communication_layer(spans: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    communication_spans: list[dict[str, Any]] = []
    protocols: set[str] = set()
    endpoints: set[str] = set()
    latencies: list[float] = []
    error_count = 0

    for span in spans:
        attrs = span.get("span_attributes") or {}
        kind = str(span.get("span_kind") or "").lower()
        has_protocol = any(
            key.startswith(("http.", "rpc.", "server.", "network.", "net.peer.", "messaging."))
            for key in attrs
        )
        if kind not in COMMUNICATION_SPAN_KINDS and not has_protocol:
            continue
        communication_spans.append(span)
        duration = _duration_ms(span.get("duration"))
        if duration is not None:
            latencies.append(duration)
        if str(span.get("status_code") or "").lower() == "error":
            error_count += 1
        for key in ("rpc.system", "network.protocol.name", "http.request.method", "http.method", "messaging.system"):
            value = attrs.get(key)
            if value not in (None, ""):
                protocols.add(str(value))
        for key in ("server.address", "net.peer.name", "http.host", "url.full", "http.url", "rpc.service"):
            value = attrs.get(key)
            if value not in (None, ""):
                endpoints.add(str(value)[:160])

    logical_edges = [edge for edge in edges if edge.get("edge_type") in COMMUNICATION_EDGE_TYPES]
    retry_count = sum(1 for edge in logical_edges if edge.get("edge_type") == "Retry")
    total = len(communication_spans)
    error_rate = _percent(error_count, total) or 0.0
    p95 = _percentile(latencies, 0.95)
    status = "critical" if error_rate >= 20 else "warning" if error_rate > 0 or retry_count > 0 else "healthy"
    telemetry_available = bool(communication_spans or logical_edges)
    if not telemetry_available:
        status = "unknown"

    highlights = []
    if logical_edges:
        highlights.append(f"识别 {len(logical_edges)} 条 Agent/工具逻辑通信边")
    if communication_spans:
        highlights.append(f"识别 {total} 个网络/RPC/消息 Span")
    if error_count or retry_count:
        highlights.append(f"通信异常 {error_count} 次，重试 {retry_count} 次")
    if not highlights:
        highlights.append("Trace 未携带可识别的通信遥测")

    return {
        "id": "communication",
        "name": "通信层",
        "status": status,
        "telemetry_available": telemetry_available,
        "source": "OTLP span kind/semantic attributes + 因果图调用边",
        "metrics": [
            _metric("span_count", "通信 Span", total, "", "trace"),
            _metric("logical_edge_count", "逻辑调用边", len(logical_edges), "", "graph"),
            _metric("error_rate", "错误率", error_rate, "%", "trace"),
            _metric("retry_count", "重试次数", retry_count, "", "graph"),
            _metric("p50_latency_ms", "P50 时延", _percentile(latencies, 0.50), "ms", "trace"),
            _metric("p95_latency_ms", "P95 时延", p95, "ms", "trace"),
            _metric("endpoint_count", "远端端点", len(endpoints), "", "trace"),
        ],
        "highlights": highlights,
        "details": {"protocols": sorted(protocols), "endpoints": sorted(endpoints)},
    }


def _application_layer(
    spans: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    agents: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = [node for node in nodes if node.get("type") == "Error"]
    llm_calls = sum(1 for span in spans if span.get("span_name") == "LiteLLMModel.__call__")
    tool_calls = sum(
        1
        for span in spans
        if (span.get("span_attributes") or {}).get("tool.name")
        or str(span.get("span_name") or "").endswith("Tool")
    )
    failed_spans = sum(1 for span in spans if str(span.get("status_code") or "").lower() == "error")
    telemetry_available = bool(spans or nodes)
    status = "critical" if failed_spans or errors else "healthy" if telemetry_available else "unknown"
    return {
        "id": "application",
        "name": "应用层",
        "status": status,
        "telemetry_available": telemetry_available,
        "source": "Agent OTel Trace + 结构化因果图",
        "metrics": [
            _metric("span_count", "Span 数", len(spans), "", "trace"),
            _metric("agent_count", "Agent 数", len(agents), "", "graph"),
            _metric("llm_call_count", "LLM 调用", llm_calls, "", "trace"),
            _metric("tool_call_count", "工具调用", tool_calls, "", "trace"),
            _metric("graph_node_count", "因果节点", len(nodes), "", "graph"),
            _metric("graph_edge_count", "因果边", len(edges), "", "graph"),
            _metric("error_count", "错误节点", len(errors), "", "graph"),
        ],
        "highlights": [
            f"已覆盖 Agent、LLM、工具调用和因果图，共 {len(spans)} 个 Span",
            f"检测到 {failed_spans} 个失败 Span、{len(errors)} 个错误节点",
        ],
        "details": {
            "failed_span_count": failed_spans,
            "node_types": dict(sorted({str(node.get('type')): sum(1 for n in nodes if n.get('type') == node.get('type')) for node in nodes}.items())),
        },
    }


def build_multilayer_monitoring(
    normalized_trace: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    agents: list[dict[str, Any]],
    *,
    hardware_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the unified hardware/virtualization/communication/application view."""
    spans = list(_walk_spans(normalized_trace.get("spans") or []))
    layers = [
        hardware_snapshot or collect_hardware_snapshot(),
        _virtualization_layer(spans),
        _communication_layer(spans, edges),
        _application_layer(spans, nodes, edges, agents),
    ]
    available_count = sum(1 for layer in layers if layer.get("telemetry_available"))
    overall_status = _worst_status(*(str(layer.get("status") or "unknown") for layer in layers))
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "coverage": {
            "layer_count": len(layers),
            "available_count": available_count,
            "percentage": _round(100.0 * available_count / len(layers), 1),
            "layer_ids": list(LAYER_IDS),
        },
        "layers": layers,
    }


def build_live_system_snapshot() -> dict[str, Any]:
    """Return the host-only subset used before a trace is uploaded."""
    hardware = collect_hardware_snapshot()
    empty_trace = {"spans": []}
    payload = build_multilayer_monitoring(empty_trace, [], [], [], hardware_snapshot=hardware)
    payload["mode"] = "live_host"
    return payload
