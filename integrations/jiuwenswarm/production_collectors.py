"""Production-oriented collectors used by the JiuwenSwarm monitoring sidecar.

The collectors have no third-party runtime dependency.  Each integration is
isolated: an unavailable Kubernetes API or accelerator exporter becomes a
collector-health observation instead of stopping the remaining collectors.
"""

from __future__ import annotations

import json
import os
import re
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


PROMETHEUS_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]Inf)(?:\s+\d+)?$"
)
PROMETHEUS_LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def _number(value: str) -> float | None:
    try:
        result = float(value)
        return result if result == result and result not in (float("inf"), float("-inf")) else None
    except ValueError:
        return None


def parse_prometheus(text: str) -> list[dict[str, Any]]:
    """Parse the numeric subset of Prometheus/OpenMetrics exposition format."""
    samples: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_LINE.match(line)
        if not match:
            continue
        value = _number(match.group("value"))
        if value is None:
            continue
        labels = {
            key: bytes(raw, "utf-8").decode("unicode_escape")
            for key, raw in PROMETHEUS_LABEL.findall(match.group("labels") or "")
        }
        samples.append({"name": match.group("name"), "labels": labels, "value": value})
    return samples


def _fetch_text(url: str, timeout: float = 3.0) -> str:
    request = Request(url, headers={"Accept": "text/plain"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator supplied endpoint
        return response.read().decode("utf-8", errors="replace")


def _safe_endpoint(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _values(samples: list[dict[str, Any]], name: str) -> list[float]:
    return [sample["value"] for sample in samples if sample["name"] == name]


def _scope_accelerator_samples(
    samples: list[dict[str, Any]], pod_name: str, namespace: str
) -> tuple[list[dict[str, Any]], str]:
    """Restrict exporter samples to devices assigned to the current Pod when labels exist."""
    if not pod_name:
        return samples, "exporter"
    pod_keys = ("pod", "pod_name")
    namespace_keys = ("namespace", "pod_namespace")
    device_keys = ("UUID", "gpu", "device", "npu_id", "id", "chip_id")

    def label_value(labels: dict[str, str], keys: tuple[str, ...]) -> str:
        return next((str(labels[key]) for key in keys if labels.get(key)), "")

    matching = [
        sample
        for sample in samples
        if label_value(sample["labels"], pod_keys) == pod_name
        and (not namespace or label_value(sample["labels"], namespace_keys) in {"", namespace})
    ]
    if not matching:
        return samples, "exporter"
    device_ids = {
        label_value(sample["labels"], device_keys)
        for sample in matching
        if label_value(sample["labels"], device_keys)
    }
    scoped = [
        sample
        for sample in samples
        if sample in matching or label_value(sample["labels"], device_keys) in device_ids
    ]
    return scoped, "pod"


def _aggregate_accelerator(samples: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    if kind == "gpu":
        names = {
            "utilization_percent": "DCGM_FI_DEV_GPU_UTIL",
            "memory_used_mib": "DCGM_FI_DEV_FB_USED",
            "memory_free_mib": "DCGM_FI_DEV_FB_FREE",
            "temperature_celsius": "DCGM_FI_DEV_GPU_TEMP",
            "power_watts": "DCGM_FI_DEV_POWER_USAGE",
            "error": "DCGM_FI_DEV_XID_ERRORS",
            "health": "DCGM_EXP_GPU_HEALTH_STATUS",
        }
        prefix = "DCGM_"
        identity_labels = ("gpu", "UUID", "device", "pod", "namespace", "container")
    else:
        names = {
            "utilization_percent": "npu_chip_info_utilization",
            "memory_used_mib": "npu_chip_info_used_memory",
            "memory_total_mib": "npu_chip_info_total_memory",
            "hbm_used_mib": "npu_chip_info_hbm_used_memory",
            "hbm_total_mib": "npu_chip_info_hbm_total_memory",
            "temperature_celsius": "npu_chip_info_temperature",
            "power_watts": "npu_chip_info_power",
            "error": "npu_chip_info_error_code",
            "health": "npu_chip_info_health_status",
        }
        prefix = "npu_"
        identity_labels = ("id", "npu_id", "device", "chip_id", "pod", "namespace", "container")

    related = [sample for sample in samples if sample["name"].startswith(prefix)]
    devices: dict[str, dict[str, Any]] = {}
    for sample in related:
        labels = sample["labels"]
        identity = next((str(labels[key]) for key in identity_labels if labels.get(key)), "aggregate")
        device = devices.setdefault(identity, {"id": identity})
        for output_name, metric_name in names.items():
            if sample["name"] == metric_name:
                device[output_name] = sample["value"]
        workload = {key: labels[key] for key in ("pod", "namespace", "container") if labels.get(key)}
        if workload:
            device["workload"] = workload

    def values(field: str) -> list[float]:
        return [float(item[field]) for item in devices.values() if field in item]

    util = values("utilization_percent")
    temps = values("temperature_celsius")
    errors = values("error")
    health = values("health")
    used = values("memory_used_mib") or values("hbm_used_mib")
    total = values("memory_total_mib") or values("hbm_total_mib")
    powers = values("power_watts")
    temperature_max = max(temps) if temps else None
    utilization_max = max(util) if util else None
    error_count = sum(1 for value in errors if value != 0)
    unhealthy_count = sum(1 for value in health if value <= 0)
    return {
        "accelerator_type": kind,
        "device_count": len(devices),
        "sample_count": len(related),
        "utilization_avg_percent": round(sum(util) / len(util), 2) if util else None,
        "utilization_max_percent": utilization_max,
        "memory_used_mib": sum(used) if used else None,
        "memory_total_mib": sum(total) if total else None,
        "temperature_max_celsius": temperature_max,
        "power_total_watts": sum(powers) if powers else None,
        "error_device_count": error_count,
        "unhealthy_device_count": unhealthy_count,
        "anomaly_flags": {
            "utilization_saturated": utilization_max is not None and utilization_max >= 95,
            "temperature_critical": temperature_max is not None and temperature_max >= (85 if kind == "gpu" else 90),
            "device_error": error_count > 0,
            "device_unhealthy": unhealthy_count > 0,
        },
        "devices": list(devices.values())[:64],
    }


class AcceleratorExporterCollector:
    def __init__(
        self,
        kind: str,
        url: str,
        timeout: float = 3.0,
        pod_name: str | None = None,
        namespace: str | None = None,
    ) -> None:
        self.kind = kind
        self.url = url
        self.timeout = timeout
        self.pod_name = pod_name if pod_name is not None else os.getenv("POD_NAME", "")
        self.namespace = namespace if namespace is not None else os.getenv("POD_NAMESPACE", "")

    def collect(self) -> dict[str, Any]:
        samples = parse_prometheus(_fetch_text(self.url, self.timeout))
        samples, scope = _scope_accelerator_samples(samples, self.pod_name, self.namespace)
        measurement = _aggregate_accelerator(samples, self.kind)
        if measurement["sample_count"] == 0:
            raise RuntimeError(f"no {self.kind} metric family found at exporter endpoint")
        measurement["exporter_url"] = _safe_endpoint(self.url)
        measurement["collection_scope"] = scope
        return measurement


def _read_flat_file(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return result


def _read_limit(path: Path) -> int | None:
    raw = path.read_text(encoding="utf-8").strip()
    return None if raw == "max" else int(raw)


def _read_pressure(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        result[parts[0]] = {
            key: float(value) if key.startswith("avg") else int(value)
            for key, value in (item.split("=", 1) for item in parts[1:] if "=" in item)
        }
    return result


class CgroupV2Collector:
    def __init__(self, root: Path = Path("/sys/fs/cgroup"), proc_cgroup: Path = Path("/proc/self/cgroup")) -> None:
        self.root = root
        self.proc_cgroup = proc_cgroup
        self.previous_memory_events: dict[str, int] | None = None

    def _directory(self) -> Path:
        for line in self.proc_cgroup.read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.split("::", 1)[1].lstrip("/")
                directory = (self.root / relative).resolve()
                if self.root.resolve() not in directory.parents and directory != self.root.resolve():
                    raise RuntimeError("resolved cgroup path escaped cgroup root")
                return directory
        raise RuntimeError("cgroup v2 unified hierarchy not found")

    def collect(self) -> dict[str, Any]:
        directory = self._directory()
        memory_events = _read_flat_file(directory / "memory.events")
        cpu_stat = _read_flat_file(directory / "cpu.stat")
        previous = self.previous_memory_events or memory_events
        deltas = {key: max(0, value - previous.get(key, 0)) for key, value in memory_events.items()}
        self.previous_memory_events = memory_events
        current = _read_limit(directory / "memory.current")
        maximum = _read_limit(directory / "memory.max")
        oom_detected = deltas.get("oom", 0) > 0 or deltas.get("oom_kill", 0) > 0
        cpu_throttled = cpu_stat.get("nr_throttled", 0) > 0
        memory_utilization = round(current * 100.0 / maximum, 2) if current is not None and maximum else None
        return {
            "cgroup_path": str(directory),
            "memory_current_bytes": current,
            "memory_max_bytes": maximum,
            "memory_utilization_percent": memory_utilization,
            "memory_events": memory_events,
            "memory_event_deltas": deltas,
            "oom_detected": oom_detected,
            "cpu_stat": cpu_stat,
            "cpu_throttled": cpu_throttled,
            "anomaly_flags": {
                "oom": oom_detected,
                "memory_near_limit": memory_utilization is not None and memory_utilization >= 90,
                "cpu_throttled": cpu_throttled,
            },
            "memory_pressure": _read_pressure(directory / "memory.pressure"),
            "cpu_pressure": _read_pressure(directory / "cpu.pressure"),
        }


CPU_MULTIPLIERS = {"n": 1e-6, "u": 1e-3, "m": 1.0, "": 1000.0}
MEMORY_MULTIPLIERS = {
    "": 1,
    "K": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
}


def parse_kubernetes_quantity(value: str, resource: str) -> float:
    match = re.fullmatch(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))([A-Za-z]*)", str(value).strip())
    if not match:
        raise ValueError(f"invalid Kubernetes quantity: {value}")
    number = float(match.group(1))
    suffix = match.group(2)
    multipliers = CPU_MULTIPLIERS if resource == "cpu" else MEMORY_MULTIPLIERS
    if suffix not in multipliers:
        raise ValueError(f"unsupported Kubernetes quantity suffix: {suffix}")
    return number * multipliers[suffix]


class KubernetesPodCollector:
    SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")

    def __init__(
        self,
        *,
        host: str | None = None,
        port: str | None = None,
        namespace: str | None = None,
        pod_name: str | None = None,
        token_path: Path | None = None,
        ca_path: Path | None = None,
        timeout: float = 3.0,
    ) -> None:
        self.host = host or os.getenv("KUBERNETES_SERVICE_HOST", "")
        self.port = port or os.getenv("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        namespace_path = self.SERVICE_ACCOUNT / "namespace"
        self.namespace = namespace or os.getenv("POD_NAMESPACE") or (
            namespace_path.read_text(encoding="utf-8").strip() if namespace_path.exists() else "default"
        )
        self.pod_name = pod_name or os.getenv("POD_NAME") or os.getenv("HOSTNAME", "")
        self.token_path = token_path or self.SERVICE_ACCOUNT / "token"
        self.ca_path = ca_path or self.SERVICE_ACCOUNT / "ca.crt"
        self.timeout = timeout
        self.seen_event_uids: set[str] = set()

    @property
    def available(self) -> bool:
        return bool(self.host and self.pod_name and self.token_path.exists())

    def _get(self, path: str) -> dict[str, Any]:
        token = self.token_path.read_text(encoding="utf-8").strip()
        context = ssl.create_default_context(cafile=str(self.ca_path)) if self.ca_path.exists() else ssl.create_default_context()
        request = Request(
            f"https://{self.host}:{self.port}{path}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        with urlopen(request, timeout=self.timeout, context=context) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _resource_totals(pod: dict[str, Any]) -> dict[str, float]:
        totals = {"cpu_request_millicores": 0.0, "cpu_limit_millicores": 0.0, "memory_request_bytes": 0.0, "memory_limit_bytes": 0.0}
        for container in pod.get("spec", {}).get("containers", []):
            resources = container.get("resources", {})
            for kind, suffix in (("requests", "request"), ("limits", "limit")):
                values = resources.get(kind, {})
                if values.get("cpu"):
                    totals[f"cpu_{suffix}_millicores"] += parse_kubernetes_quantity(values["cpu"], "cpu")
                if values.get("memory"):
                    totals[f"memory_{suffix}_bytes"] += parse_kubernetes_quantity(values["memory"], "memory")
        return totals

    @staticmethod
    def _container_status(pod: dict[str, Any]) -> dict[str, Any]:
        statuses = pod.get("status", {}).get("containerStatuses", []) + pod.get("status", {}).get("initContainerStatuses", [])
        containers: list[dict[str, Any]] = []
        oom_containers: list[str] = []
        for status in statuses:
            terminated = status.get("state", {}).get("terminated") or status.get("lastState", {}).get("terminated") or {}
            reason = terminated.get("reason")
            if reason == "OOMKilled":
                oom_containers.append(status.get("name", "unknown"))
            containers.append({
                "name": status.get("name"),
                "ready": bool(status.get("ready")),
                "restart_count": int(status.get("restartCount", 0)),
                "last_termination_reason": reason,
                "last_exit_code": terminated.get("exitCode"),
            })
        return {
            "containers": containers,
            "restart_count": sum(item["restart_count"] for item in containers),
            "oom_killed": bool(oom_containers),
            "oom_killed_containers": oom_containers,
        }

    def _pod_metrics(self) -> dict[str, Any]:
        encoded_ns, encoded_pod = quote(self.namespace, safe=""), quote(self.pod_name, safe="")
        errors: list[str] = []
        for version in ("v1", "v1beta1"):
            try:
                metrics = self._get(f"/apis/metrics.k8s.io/{version}/namespaces/{encoded_ns}/pods/{encoded_pod}")
                cpu = sum(parse_kubernetes_quantity(item.get("usage", {}).get("cpu", "0"), "cpu") for item in metrics.get("containers", []))
                memory = sum(parse_kubernetes_quantity(item.get("usage", {}).get("memory", "0"), "memory") for item in metrics.get("containers", []))
                return {"cpu_usage_millicores": cpu, "memory_working_set_bytes": memory, "metrics_api_version": version}
            except Exception as exc:  # isolated optional API, reported below
                errors.append(f"{version}: {type(exc).__name__}")
        return {"metrics_api_available": False, "metrics_api_errors": errors}

    def collect(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not self.available:
            raise RuntimeError("Kubernetes in-cluster identity is unavailable")
        encoded_ns, encoded_pod = quote(self.namespace, safe=""), quote(self.pod_name, safe="")
        pod = self._get(f"/api/v1/namespaces/{encoded_ns}/pods/{encoded_pod}")
        metadata, status = pod.get("metadata", {}), pod.get("status", {})
        conditions = {item.get("type"): item.get("status") for item in status.get("conditions", [])}
        measurement = {
            "namespace": self.namespace,
            "pod": self.pod_name,
            "pod_uid": metadata.get("uid"),
            "node": pod.get("spec", {}).get("nodeName"),
            "phase": status.get("phase"),
            "ready": conditions.get("Ready") == "True",
            "conditions": conditions,
            **self._container_status(pod),
            **self._resource_totals(pod),
            **self._pod_metrics(),
        }
        measurement["anomaly_flags"] = {
            "oom_killed": measurement["oom_killed"],
            "restart_loop": measurement["restart_count"] >= 3,
            "not_ready": not measurement["ready"],
            "failed_phase": measurement["phase"] in {"Failed", "Unknown"},
        }

        field_selector = quote(f"involvedObject.uid={metadata.get('uid', '')}", safe="=,")
        event_list = self._get(f"/api/v1/namespaces/{encoded_ns}/events?fieldSelector={field_selector}")
        new_events: list[dict[str, Any]] = []
        for item in event_list.get("items", []):
            uid = str(item.get("metadata", {}).get("uid") or "")
            if not uid or uid in self.seen_event_uids:
                continue
            self.seen_event_uids.add(uid)
            if item.get("type") == "Warning" or item.get("reason") in {"OOMKilling", "Evicted", "BackOff", "Failed"}:
                new_events.append({
                    "uid": uid,
                    "type": item.get("type"),
                    "reason": item.get("reason"),
                    "message": str(item.get("message") or "")[:2048],
                    "count": item.get("count"),
                    "first_timestamp": item.get("firstTimestamp") or item.get("eventTime"),
                    "last_timestamp": item.get("lastTimestamp") or item.get("eventTime"),
                })
        return measurement, new_events


class ProductionCollectorSuite:
    """Orchestrate optional collectors and report their health independently."""

    def __init__(
        self,
        *,
        kubernetes: str = "auto",
        cgroup: str = "auto",
        dcgm_url: str = "",
        npu_url: str = "",
    ) -> None:
        self.kubernetes_mode = kubernetes
        self.cgroup_mode = cgroup
        self.kubernetes = KubernetesPodCollector()
        self.cgroup = CgroupV2Collector()
        self.accelerators = [
            collector
            for collector in (
                AcceleratorExporterCollector("gpu", dcgm_url) if dcgm_url else None,
                AcceleratorExporterCollector("npu", npu_url) if npu_url else None,
            )
            if collector is not None
        ]

    @staticmethod
    def _event(common: dict[str, Any], layer: str, event_type: str, measurement: dict[str, Any]) -> dict[str, Any]:
        return {**common, "layer": layer, "event_type": event_type, "measurement": measurement}

    def collect(self, common: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        health: dict[str, Any] = {}

        kube_enabled = self.kubernetes_mode == "on" or (self.kubernetes_mode == "auto" and self.kubernetes.available)
        if kube_enabled:
            try:
                measurement, kube_events = self.kubernetes.collect()
                events.append(self._event(common, "virtualization", "kubernetes.pod.snapshot", measurement))
                events.extend(self._event(common, "virtualization", "kubernetes.warning", item) for item in kube_events)
                health["kubernetes"] = {"status": "ok", "warning_events": len(kube_events)}
            except Exception as exc:
                health["kubernetes"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        else:
            health["kubernetes"] = {"status": "disabled" if self.kubernetes_mode == "off" else "not_detected"}

        cgroup_enabled = self.cgroup_mode == "on" or (self.cgroup_mode == "auto" and os.name != "nt" and Path("/sys/fs/cgroup/cgroup.controllers").exists())
        if cgroup_enabled:
            try:
                events.append(self._event(common, "virtualization", "cgroup.v2.snapshot", self.cgroup.collect()))
                health["cgroup_v2"] = {"status": "ok"}
            except Exception as exc:
                health["cgroup_v2"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        else:
            health["cgroup_v2"] = {"status": "disabled" if self.cgroup_mode == "off" else "not_detected"}

        for collector in self.accelerators:
            try:
                measurement = collector.collect()
                events.append(self._event(common, "hardware", f"accelerator.{collector.kind}.snapshot", measurement))
                health[collector.kind] = {"status": "ok", "device_count": measurement["device_count"]}
            except Exception as exc:
                health[collector.kind] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        events.append(self._event(common, "application", "collector.health", health))
        return events
