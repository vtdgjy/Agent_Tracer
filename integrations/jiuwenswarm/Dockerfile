FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/agent-tracer
COPY multilayer_monitor.py monitoring_store.py ./
COPY integrations/jiuwenswarm ./integrations/jiuwenswarm

USER 65532:65532
ENTRYPOINT ["python", "integrations/jiuwenswarm/runtime_probe.py"]
