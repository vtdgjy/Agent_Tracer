# Agent Tracer

Agent Tracer 是一个面向智能体系统的 OTel Trace 分析与故障定位项目。它接收 Agent、LLM 和工具调用产生的 Trace，把分散的 Span 还原成带有时间、调用、观察、重试和语义传播关系的因果图，再从最终答案或终止错误反向抽取故障路径，辅助判断“哪个步骤最可能导致了最终失败”。

项目同时提供 Web 原型、命令行流水线和四层监控接口。老师或代码阅读工具可以优先阅读 [项目总体说明](docs/项目总体说明.md)、[故障特征说明](docs/故障特征说明.md) 和 [结构化故障特征目录](config/fault_features.json)，再进入具体源码。

仓库只保留核心源码、部署配置和说明文件，不包含业务 Trace、实验数据、运行结果、日志、论文材料或真实 API 密钥。

## 系统组成

- **输入层**：`trace_input_adapter.py` 将不同 JSON/OTel 表达统一为 Span 树。
- **分析层**：`OTL_trace.py` 构建因果图；`semantic_trace_validator.py` 检查工具执行和证据链；`agenttrace_target_pruning.py` 保留与故障相关的目标路径。
- **诊断层**：`deepseek_judge_trace.py` 执行 LLM 辅助根因判断，`diagnostic_report.py` 输出结构化报告。
- **监控层**：`multilayer_monitor.py` 聚合硬件、虚拟化、通信和应用四层信息，`monitoring_store.py` 提供事件存储与查询。
- **交互层**：`web_app.py` 提供 Trace 上传、解析结果、诊断和监控 API；`web_static/index.html` 是配套页面。
- **集成层**：`integrations/jiuwenswarm/` 提供 JiuwenSwarm 探针、OTLP 文件接收器、生产采集器和 Kubernetes 配置。

## 环境准备

需要 Python 3.10 或更新版本。安装 `uv` 后，在本目录执行：

```powershell
uv sync
```

如需使用在线数据集模式，再执行：

```powershell
uv sync --extra dataset
```

如需启用 JiuwenSwarm 的 OTLP 文件接收器，再执行：

```powershell
uv sync --extra otlp-collector
```

将 `.env.example` 复制为 `.env`，填入可用的模型服务 API Key 和服务地址。也可在系统环境变量中设置 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`。请勿将包含真实密钥的 `.env` 分享或提交。

## 启动 Web 系统

```powershell
uv run python .\web_app.py
```

浏览器打开 `http://127.0.0.1:5000`。页面提供 Trace 上传解析、图结构查看、根因诊断和多层监控数据展示。

## 命令行入口

```powershell
uv run otl-trace --help
uv run pruning-strategies --help
uv run deepseek-judge --help
uv run agent-tracer --help
uv run deepseek-align --help
uv run deepseek-unprocessed-eval --help
uv run pipeline-run --help
uv run run-all-in-one --help
```

需要部署 JiuwenSwarm 四层监控探针时，核心采集器位于 `integrations/jiuwenswarm/`，其中包含运行探针、生产监控采集器、OTLP 文件接收器及 Kubernetes 配置。OTLP 文件接收器需额外安装：`uv sync --extra otlp-collector`。探针可在项目根目录运行 `python integrations/jiuwenswarm/runtime_probe.py --help` 查看参数。

具体输入路径和处理参数通过对应命令的 `--help` 查看。Trace 数据及处理结果由使用者在部署环境中提供，本包不包含业务数据或历史运行结果。

## 故障特征

当前代码已经实现并整理了以下故障线索：工具执行失败、HTTP 4xx/5xx、缺少必填参数、参数与结果不一致、空结果、日期/时间不一致、参数来源不可追溯、证据未传播或错误传播、重试恢复失败、规划与动作漂移、最终答案不收敛，以及硬件/虚拟化/通信层压力。每项特征的触发信号、代码来源和可解释含义见 `docs/故障特征说明.md`；机器可读版本见 `config/fault_features.json`。

