import argparse
import ast
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_THINKING_MODE = "disabled"
IMPACT_LEVELS = ["LOW", "MEDIUM", "HIGH"]
ID_IN_TEXT_RE = re.compile(r"\b(?:[0-9a-f]{16}|[A-Z]_[0-9]+)\b", re.IGNORECASE)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
DEFAULT_MAX_INPUT_CHARS = 28000
MIN_RETRY_INPUT_CHARS = 6000
DEFAULT_MODEL_CONTEXT_TOKENS = 131072
DEFAULT_MAX_INPUT_TOKENS = 0
MIN_RETRY_INPUT_TOKENS = 4000
DEFAULT_JUDGE_MAX_TOKENS = 2200
DEFAULT_TOKEN_SAFETY_BUFFER = 4096


_thread_local = threading.local()


def create_chat_completion(client, **request):
    base_url = str(getattr(client, "base_url", "") or "")
    model = str(request.get("model") or "")
    if "api.deepseek.com" in base_url and model.startswith("deepseek"):
        request["extra_body"] = {"thinking": {"type": DEEPSEEK_THINKING_MODE}}
    return client.chat.completions.create(**request)

def load_json_lenient(path: Path):
    text = path.read_text(encoding="utf-8")
    s = (text or "").strip().replace("\ufeff", "")
    if not s:
        raise ValueError(f"empty file: {path}")

    try:
        data = json.loads(s)
    except Exception:
        data = None

    if isinstance(data, str):
        # Decode a few rounds for JSON-encoded JSON payloads.
        for _ in range(3):
            try:
                data = json.loads(data)
            except Exception:
                break
            if not isinstance(data, str):
                break

    if data is not None and not isinstance(data, str):
        return data

    if s.startswith("```"):
        s2 = re.sub(r"^```(?:json)?\s*", "", s)
        s2 = re.sub(r"\s*```$", "", s2)
        try:
            return json.loads(s2)
        except Exception:
            pass

    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    raise ValueError(f"annotation is not valid JSON: {path}")


def parse_json_from_text(text: str):
    def _clean(raw):
        return (raw or "").strip().replace("\ufeff", "")

    def _extract_code_block(s):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
        return m.group(1).strip() if m else s

    def _extract_balanced_json_obj(s):
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(s)):
            ch = s[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : idx + 1]
        return None

    def _json_or_literal_eval(s):
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            return obj if isinstance(obj, (dict, list)) else None
        except Exception:
            return None

    text = _clean(text)
    candidates = [text, _extract_code_block(text)]

    balanced = _extract_balanced_json_obj(text)
    if balanced:
        candidates.append(balanced)

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidates.append(m.group(0))

    for cand in candidates:
        if not cand:
            continue
        parsed = _json_or_literal_eval(_clean(cand))
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("LLM output is not valid JSON")


def repair_json_with_llm(client, model, bad_text):
    repair_prompt = f"""
请把下面文本改写成严格 JSON 对象，仅输出 JSON，不要解释。
要求：
- 保持原有字段语义；
- 若缺字段可补空值；
- 使用双引号；
- 不要 markdown 代码块。

原文本：
{(bad_text or '')[:12000]}
""".strip()

    resp = create_chat_completion(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You are a JSON repair tool. Output only valid JSON object."},
            {"role": "user", "content": repair_prompt},
        ],
        temperature=0.0,
        max_tokens=1400,
    )
    return (resp.choices[0].message.content or "").strip()


def extract_usage(resp):
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    def _get(v, key):
        if isinstance(v, dict):
            return v.get(key)
        return getattr(v, key, None)

    return {
        "prompt_tokens": _get(usage, "prompt_tokens"),
        "completion_tokens": _get(usage, "completion_tokens"),
        "total_tokens": _get(usage, "total_tokens"),
    }


def make_unprocessed_judge_prompt(trace_id: str, raw_text: str):
    return f"""
你是一个严格的 Agent 轨迹失误分析员。
你会收到一条 trace 的原始文本（未做结构化摘要）。
请直接基于原文本做根因分析，定位导致错误答案/不可靠结论的问题链路。

trace_id={trace_id}

原始 trace 文本：
{raw_text}

只输出 1 个 JSON 对象，不要 Markdown，不要解释文字。
JSON 结构必须如下：
{{
  "trace_id": "{trace_id}",
  "source": "unprocessed_trace",
  "errors": [
    {{
      "category": "Formatting Errors|Tool-related|Goal Deviation|Poor Information Retrieval|Hallucination|Reasoning Error|Resource Abuse|Other",
      "module": "planning/tool_calling/search/final_answer/unknown",
      "location": "span_id/node_id/unknown",
      "evidence": "来自原文本的短证据",
      "description": "错误机理描述",
      "impact": "LOW|MEDIUM|HIGH"
    }}
  ],
  "scores": [
    {{
      "reliability_score": 1-5,
      "reliability_reasoning": "简要理由",
      "security_score": 1-5,
      "security_reasoning": "简要理由",
      "instruction_adherence_score": 1-5,
      "instruction_adherence_reasoning": "简要理由",
      "plan_opt_score": 1-5,
      "plan_opt_reasoning": "简要理由",
      "overall": 1-5
    }}
  ],
  "root_causes": [
    {{
      "module": "模块名",
      "reason": "根因",
      "impact": "LOW|MEDIUM|HIGH",
      "confidence": 0-1
    }}
  ],
  "final_diagnosis": "一句话总结最关键问题"
}}

要求：
1) evidence 必须来自输入原文可见内容，不得捏造。
2) errors/scores/root_causes 必须是数组。
3) 输出前自检 JSON 可解析、字段类型正确。
""".strip()


def is_context_length_error(exc: Exception):
    msg = str(exc or "")
    needles = [
        "maximum context length",
        "requested",
        "invalid_request_error",
        "Please reduce the length of the messages or completion",
    ]
    return all(part in msg for part in needles)


def estimate_text_tokens(text: str):
    if not text:
        return 0

    cjk_chars = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
    ascii_chars = len(re.findall(r"[\x00-\x7f]", text))
    other_chars = max(0, len(text) - cjk_chars - ascii_chars)

    ascii_tokens = (ascii_chars + 2) // 3
    other_tokens = (other_chars + 1) // 2
    return cjk_chars + ascii_tokens + other_tokens


def estimate_message_tokens(system_text: str, user_text: str):
    message_overhead = 32
    return estimate_text_tokens(system_text) + estimate_text_tokens(user_text) + message_overhead


def truncate_text_middle(text: str, max_chars: int):
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False

    marker = "\n\n[... middle content truncated to fit model context ...]\n\n"
    if max_chars <= len(marker) + 32:
        return text[:max_chars], True

    head_chars = max(1, ((max_chars - len(marker)) * 2) // 3)
    tail_chars = max(1, max_chars - len(marker) - head_chars)
    return text[:head_chars] + marker + text[-tail_chars:], True


def truncate_text_middle_by_tokens(text: str, max_tokens: int):
    estimated_tokens = estimate_text_tokens(text)
    if max_tokens <= 0 or estimated_tokens <= max_tokens:
        return text, False, estimated_tokens

    marker = "\n\n[... middle content truncated to fit model token budget ...]\n\n"
    if len(text) <= len(marker) + 32:
        clipped = text[: max(1, len(text) // 2)]
        return clipped + marker, True, estimate_text_tokens(clipped + marker)

    low = 1
    high = len(text)
    best_text = marker
    best_tokens = estimate_text_tokens(best_text)

    while low <= high:
        keep_chars = (low + high) // 2
        if keep_chars <= len(marker) + 32:
            candidate = text[:keep_chars]
        else:
            head_chars = max(1, ((keep_chars - len(marker)) * 2) // 3)
            tail_chars = max(1, keep_chars - len(marker) - head_chars)
            candidate = text[:head_chars] + marker + text[-tail_chars:]

        candidate_tokens = estimate_text_tokens(candidate)
        if candidate_tokens <= max_tokens:
            best_text = candidate
            best_tokens = candidate_tokens
            low = keep_chars + 1
        else:
            high = keep_chars - 1

    return best_text, True, best_tokens


def apply_input_limits(text: str, max_tokens: int, max_chars: int):
    limited_text, truncated_by_tokens, estimated_tokens = truncate_text_middle_by_tokens(text, max_tokens)

    if max_chars and max_chars > 0:
        limited_text, truncated_by_chars = truncate_text_middle(limited_text, max_chars)
    else:
        truncated_by_chars = False

    return {
        "text": limited_text,
        "truncated": truncated_by_tokens or truncated_by_chars,
        "estimated_input_tokens": estimate_text_tokens(limited_text),
        "truncated_by_tokens": truncated_by_tokens,
        "truncated_by_chars": truncated_by_chars,
        "token_limit": max_tokens,
        "char_limit": max_chars,
        "source_estimated_tokens": estimate_text_tokens(text),
        "pre_char_limit_estimated_tokens": estimated_tokens,
    }


def resolve_input_token_limit(
    trace_id: str,
    requested_max_input_tokens: int,
    model_context_tokens: int,
    max_output_tokens: int,
    token_safety_buffer: int,
):
    system_text = "You are a strict evaluator. Output only valid JSON."
    prompt_shell = make_unprocessed_judge_prompt(trace_id, "")
    reserved_tokens = estimate_message_tokens(system_text, prompt_shell)
    available_tokens = model_context_tokens - max_output_tokens - token_safety_buffer - reserved_tokens
    safe_limit = max(512, available_tokens)

    if requested_max_input_tokens and requested_max_input_tokens > 0:
        return min(requested_max_input_tokens, safe_limit), reserved_tokens, safe_limit
    return safe_limit, reserved_tokens, safe_limit


def build_retry_token_limits(initial_limit: int, source_estimated_tokens: int, max_attempts: int):
    if source_estimated_tokens <= 0:
        return [0]

    initial_limit = max(512, min(initial_limit, source_estimated_tokens))

    limits = [initial_limit]
    next_limit = initial_limit
    floor_limit = min(source_estimated_tokens, MIN_RETRY_INPUT_TOKENS)

    while len(limits) < max(1, int(max_attempts or 1)):
        if next_limit <= floor_limit:
            break
        next_limit = max(floor_limit, next_limit // 2)
        if next_limit >= limits[-1]:
            break
        limits.append(next_limit)

    return limits


def call_unprocessed_judge(
    client,
    model: str,
    trace_id: str,
    raw_text: str,
    max_input_tokens: int,
    max_input_chars: int,
    model_context_tokens: int,
    max_output_tokens: int,
    token_safety_buffer: int,
    max_attempts: int = 3,
):
    last_err = None
    attempts = max(1, int(max_attempts or 1))
    source_estimated_tokens = estimate_text_tokens(raw_text)
    initial_token_limit, prompt_shell_tokens, safe_token_limit = resolve_input_token_limit(
        trace_id,
        max_input_tokens,
        model_context_tokens,
        max_output_tokens,
        token_safety_buffer,
    )
    retry_limits = build_retry_token_limits(initial_token_limit, source_estimated_tokens, attempts)
    system_text = "You are a strict evaluator. Output only valid JSON."

    for token_limit in retry_limits:
        limited = apply_input_limits(raw_text, token_limit, max_input_chars)
        sent_text = limited["text"]
        prompt = make_unprocessed_judge_prompt(trace_id, sent_text)
        estimated_prompt_tokens = estimate_message_tokens(system_text, prompt)

        try:
            resp = create_chat_completion(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_output_tokens,
            )
        except Exception as e:
            last_err = e
            if is_context_length_error(e):
                continue
            raise

        text = (resp.choices[0].message.content or "").strip()

        try:
            parsed = parse_json_from_text(text)
            usage = extract_usage(resp)
            return parsed, usage, {
                "input_chars": len(sent_text),
                "source_chars": len(raw_text),
                "input_estimated_tokens": limited["estimated_input_tokens"],
                "source_estimated_tokens": source_estimated_tokens,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "prompt_shell_estimated_tokens": prompt_shell_tokens,
                "truncated": limited["truncated"],
                "truncated_by_tokens": limited["truncated_by_tokens"],
                "truncated_by_chars": limited["truncated_by_chars"],
                "requested_max_input_tokens": max_input_tokens,
                "effective_max_input_tokens": token_limit,
                "requested_max_input_chars": max_input_chars,
                "effective_max_input_chars": max_input_chars if max_input_chars and max_input_chars > 0 else None,
                "model_context_tokens": model_context_tokens,
                "max_output_tokens": max_output_tokens,
                "token_safety_buffer": token_safety_buffer,
                "safe_input_token_limit": safe_token_limit,
            }
        except Exception as e1:
            try:
                repaired = repair_json_with_llm(client, model, text)
                parsed = parse_json_from_text(repaired)
                usage = extract_usage(resp)
                return parsed, usage, {
                    "input_chars": len(sent_text),
                    "source_chars": len(raw_text),
                    "input_estimated_tokens": limited["estimated_input_tokens"],
                    "source_estimated_tokens": source_estimated_tokens,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "prompt_shell_estimated_tokens": prompt_shell_tokens,
                    "truncated": limited["truncated"],
                    "truncated_by_tokens": limited["truncated_by_tokens"],
                    "truncated_by_chars": limited["truncated_by_chars"],
                    "requested_max_input_tokens": max_input_tokens,
                    "effective_max_input_tokens": token_limit,
                    "requested_max_input_chars": max_input_chars,
                    "effective_max_input_chars": max_input_chars if max_input_chars and max_input_chars > 0 else None,
                    "model_context_tokens": model_context_tokens,
                    "max_output_tokens": max_output_tokens,
                    "token_safety_buffer": token_safety_buffer,
                    "safe_input_token_limit": safe_token_limit,
                }
            except Exception as e2:
                last_err = e2 or e1

    raise last_err or RuntimeError("unprocessed judge failed")


def make_compare_prompt(trace_id: str, human_ann: dict, ds_judge: dict):
    return f"""
你是一个严格的“自动标注 vs 人工标注”覆盖率评估器。

请比较以下两份关于同一个 trace 的错误根因标注：
- trace_id: {trace_id}
- 自动标注来源: unprocessed_trace

【人工标注】
{json.dumps(human_ann, ensure_ascii=False, indent=2)}

【自动标注】
{json.dumps(ds_judge, ensure_ascii=False, indent=2)}

请输出严格 JSON（不要 markdown 代码块），按两轮评估：
{{
  "trace_id": "{trace_id}",
  "source": "unprocessed_trace",
  "round1_root_cause_coverage": {{
    "human_total": 整数,
    "matched": 整数,
    "coverage": 0-1,
    "coverage_percent": 0-100
  }},
  "round2_impact_coverage": {{
    "LOW": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
    "MEDIUM": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
    "HIGH": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
        "total_human_total": 整数,
        "total_matched": 整数,
        "total_coverage": 0-1,
        "total_coverage_percent": 0-100
  }},
  "strict_round1_root_cause_coverage": {{
    "human_total": 整数,
    "matched": 整数,
    "coverage": 0-1,
    "coverage_percent": 0-100
  }},
  "strict_round2_impact_coverage": {{
    "LOW": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
    "MEDIUM": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
    "HIGH": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
        "total_human_total": 整数,
        "total_matched": 整数,
        "total_coverage": 0-1,
        "total_coverage_percent": 0-100
  }},
  "matched_pairs": [
    {{
      "human_location": "span/node id 或 unknown",
      "auto_location": "span/node id 或 unknown",
      "human_impact": "LOW|MEDIUM|HIGH|UNKNOWN",
      "auto_impact": "LOW|MEDIUM|HIGH|UNKNOWN",
      "result_match": true,
      "location_match": true,
      "impact_match": true,
      "strict_match": true,
      "match_reason": "语义或位置匹配理由"
    }}
  ],
    "missed_human_root_causes": [
        {{
            "location": "span_id|node_id|unknown",
            "category": "类别名或 unknown",
            "impact": "LOW|MEDIUM|HIGH|UNKNOWN",
            "description": "人工根因描述",
            "evidence": "简短证据"
        }}
    ],
    "extra_auto_root_causes": [
        {{
            "location": "span_id|node_id|unknown",
            "category": "类别名或 unknown",
            "impact": "LOW|MEDIUM|HIGH|UNKNOWN",
            "description": "自动侧多报或偏移项描述",
            "evidence": "简短证据"
        }}
    ],
  "analysis": "简要分析"
}}

评估标准：
1) 第一轮忽略影响等级，仅看根因匹配。
2) 第二轮先分别给出 LOW/MEDIUM/HIGH 的 matched 和 human_total。
3) 第二轮总分只按 total_matched / total_human_total 计算，不使用宏平均。
4) 若 total_human_total=0，则 total_coverage=1。
5) `missed_human_root_causes` 和 `extra_auto_root_causes` 必须是对象数组，不要只输出字符串。
6) 若能定位 span_id/node_id，location 不允许填 unknown。

严格匹配策略（新增字段，不要替代上面的当前策略）：
1) `round1_root_cause_coverage` 和 `round2_impact_coverage` 继续使用当前宽松策略。
2) 对每个 `matched_pairs` 同时判断：
   - `result_match`: 自动根因/结果含义是否与人工根因一致，不考虑位置。
   - `location_match`: 自动 location 是否与人工 location 指向同一 span_id/node_id 或同一明确 trace 片段；unknown、仅相邻、仅上下游都不算位置匹配。
   - `impact_match`: 自动 impact 是否与人工 impact 一致；UNKNOWN 不算一致，除非两侧都 UNKNOWN。
   - `strict_match`: 必须等于 `result_match && location_match`。
3) `strict_round1_root_cause_coverage.matched` 只统计 `result_match && location_match` 都为 true 的人工根因。
4) `strict_round2_impact_coverage` 在严格匹配基础上再按 LOW/MEDIUM/HIGH 统计，若自动 impact 与人工 impact 不一致，不计入该档 matched。
""".strip()


def _extract_location_from_text(text: str) -> str:
    if not isinstance(text, str):
        return "unknown"
    m = ID_IN_TEXT_RE.search(text)
    return m.group(0) if m else "unknown"


def _normalize_root_cause_items(items, source_hint: str):
    normalized = []
    if not isinstance(items, list):
        return normalized

    for item in items:
        if isinstance(item, dict):
            description = str(
                item.get("description")
                or item.get("reason")
                or item.get("item")
                or item.get("text")
                or ""
            ).strip()
            evidence = str(item.get("evidence") or "").strip()
            location = str(item.get("location") or "").strip() or _extract_location_from_text(
                f"{description} {evidence}"
            )
            category = str(item.get("category") or "unknown").strip() or "unknown"
            impact = str(item.get("impact") or "UNKNOWN").upper().strip() or "UNKNOWN"
        else:
            description = str(item or "").strip()
            evidence = ""
            location = _extract_location_from_text(description)
            category = "unknown"
            impact = "UNKNOWN"

        if impact not in {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}:
            impact = "UNKNOWN"

        normalized.append(
            {
                "source": source_hint,
                "location": location,
                "category": category,
                "impact": impact,
                "description": description,
                "evidence": evidence,
            }
        )
    return normalized


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "matched", "match", "same", "exact", "equivalent", "是", "正确", "匹配"}:
            return True
        if text in {"false", "no", "n", "0", "unmatched", "mismatch", "different", "否", "错误", "不匹配"}:
            return False
    return default


def _normalize_impact(value) -> str:
    impact = str(value or "UNKNOWN").upper().strip() or "UNKNOWN"
    return impact if impact in {"LOW", "MEDIUM", "HIGH", "UNKNOWN"} else "UNKNOWN"


def _location_ids(value) -> set:
    if not isinstance(value, str):
        return set()
    return {m.group(0).lower() for m in ID_IN_TEXT_RE.finditer(value)}


def _location_matches(human_location: str, auto_location: str) -> bool:
    human_text = str(human_location or "").strip()
    auto_text = str(auto_location or "").strip()
    if not human_text or not auto_text:
        return False
    if human_text.lower() == "unknown" or auto_text.lower() == "unknown":
        return False

    human_ids = _location_ids(human_text)
    auto_ids = _location_ids(auto_text)
    if human_ids or auto_ids:
        return bool(human_ids & auto_ids)

    return human_text.casefold() == auto_text.casefold()


def _normalize_matched_pairs(obj: dict) -> list:
    pairs = obj.get("matched_pairs")
    if not isinstance(pairs, list):
        obj["matched_pairs"] = []
        return []

    normalized = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue

        human_location = str(pair.get("human_location") or "unknown").strip() or "unknown"
        auto_location = str(pair.get("auto_location") or "unknown").strip() or "unknown"
        human_impact = _normalize_impact(pair.get("human_impact"))
        auto_impact = _normalize_impact(pair.get("auto_impact"))

        if "result_match" in pair:
            result_match = _safe_bool(pair.get("result_match"), default=False)
        else:
            # Backward compatibility: old matched_pairs already represented a result-level match.
            result_match = True

        if "location_match" in pair:
            location_match = _safe_bool(pair.get("location_match"), default=False)
        else:
            location_match = _location_matches(human_location, auto_location)

        if "impact_match" in pair:
            impact_match = _safe_bool(pair.get("impact_match"), default=False)
        else:
            impact_match = human_impact == auto_impact

        pair["human_location"] = human_location
        pair["auto_location"] = auto_location
        pair["human_impact"] = human_impact
        pair["auto_impact"] = auto_impact
        pair["result_match"] = bool(result_match)
        pair["location_match"] = bool(location_match)
        pair["impact_match"] = bool(impact_match)
        pair["strict_match"] = bool(result_match and location_match)
        normalized.append(pair)

    obj["matched_pairs"] = normalized
    obj["strict_matched_pairs"] = [pair for pair in normalized if pair.get("strict_match")]
    return normalized


def _count_strict_pairs_by_impact(pairs: list) -> dict:
    counts = {level: 0 for level in IMPACT_LEVELS}
    for pair in pairs:
        if not pair.get("strict_match") or not pair.get("impact_match"):
            continue
        human_impact = _normalize_impact(pair.get("human_impact"))
        if human_impact in counts:
            counts[human_impact] += 1
    return counts


def _normalize_coverage_block(block, default_human_total: int, default_matched: int, empty_coverage: float = 0.0) -> dict:
    if not isinstance(block, dict):
        block = {}

    human_total = max(0, _safe_int(block.get("human_total"), default_human_total))
    matched = max(0, _safe_int(block.get("matched"), default_matched))
    matched = min(matched, human_total)
    coverage = empty_coverage if human_total == 0 else round(matched / human_total, 4)

    block["human_total"] = human_total
    block["matched"] = matched
    block["coverage"] = coverage
    block["coverage_percent"] = round(coverage * 100, 2)
    return block


def _normalize_strict_coverage(obj: dict) -> None:
    pairs = _normalize_matched_pairs(obj)

    r1 = obj.get("round1_root_cause_coverage", {})
    base_human_total = max(0, _safe_int(r1.get("human_total"), 0))
    derived_strict_matched = sum(1 for pair in pairs if pair.get("strict_match"))
    strict_r1 = obj.get("strict_round1_root_cause_coverage")
    if not isinstance(strict_r1, dict):
        strict_r1 = {}
    strict_r1["human_total"] = base_human_total
    if pairs:
        if "matched" in strict_r1:
            strict_r1["matched"] = min(_safe_int(strict_r1.get("matched"), 0), derived_strict_matched)
        else:
            strict_r1["matched"] = derived_strict_matched

    obj["strict_round1_root_cause_coverage"] = _normalize_coverage_block(
        strict_r1,
        default_human_total=base_human_total,
        default_matched=derived_strict_matched,
        empty_coverage=0.0,
    )

    base_r2 = obj.get("round2_impact_coverage", {})
    strict_r2 = obj.get("strict_round2_impact_coverage")
    if not isinstance(strict_r2, dict):
        strict_r2 = {}

    derived_by_impact = _count_strict_pairs_by_impact(pairs)
    total_human = 0
    total_matched = 0
    for level in IMPACT_LEVELS:
        base_level = base_r2.get(level, {}) if isinstance(base_r2, dict) else {}
        strict_level = strict_r2.get(level, {}) if isinstance(strict_r2.get(level), dict) else {}
        strict_level["human_total"] = max(0, _safe_int(base_level.get("human_total"), 0))
        if pairs:
            if "matched" in strict_level:
                strict_level["matched"] = min(_safe_int(strict_level.get("matched"), 0), derived_by_impact[level])
            else:
                strict_level["matched"] = derived_by_impact[level]
        strict_r2[level] = _normalize_coverage_block(
            strict_level,
            default_human_total=strict_level["human_total"],
            default_matched=derived_by_impact[level],
            empty_coverage=1.0,
        )
        total_human += strict_r2[level]["human_total"]
        total_matched += strict_r2[level]["matched"]

    total_coverage = 1 if total_human == 0 else round(total_matched / total_human, 4)
    strict_r2["total_human_total"] = total_human
    strict_r2["total_matched"] = total_matched
    strict_r2["total_coverage"] = total_coverage
    strict_r2["total_coverage_percent"] = round(total_coverage * 100, 2)
    strict_r2["macro_coverage"] = total_coverage
    strict_r2["macro_coverage_percent"] = round(total_coverage * 100, 2)
    obj["strict_round2_impact_coverage"] = strict_r2


def call_compare(client, model: str, trace_id: str, human_ann: dict, auto_judge: dict, max_attempts: int = 3):
    prompt = make_compare_prompt(trace_id, human_ann, auto_judge)
    last_err = None

    for _ in range(max(1, int(max_attempts or 1))):
        resp = create_chat_completion(
            client,
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1400,
        )
        text = (resp.choices[0].message.content or "").strip()

        try:
            return parse_json_from_text(text)
        except Exception as e1:
            try:
                repaired = repair_json_with_llm(client, model, text)
                return parse_json_from_text(repaired)
            except Exception as e2:
                last_err = e2 or e1

    raise last_err or RuntimeError("alignment compare failed")


def normalize_judge_result(obj: dict, trace_id: str):
    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("trace_id", trace_id)
    obj.setdefault("source", "unprocessed_trace")
    if not isinstance(obj.get("errors"), list):
        obj["errors"] = []
    if not isinstance(obj.get("scores"), list):
        obj["scores"] = []
    if not isinstance(obj.get("root_causes"), list):
        obj["root_causes"] = []
    obj.setdefault("final_diagnosis", "")
    return obj


def normalize_compare_result(obj: dict, trace_id: str):
    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("trace_id", trace_id)
    obj.setdefault("source", "unprocessed_trace")

    if not isinstance(obj.get("round1_root_cause_coverage"), dict):
        obj["round1_root_cause_coverage"] = {}
    r1 = obj["round1_root_cause_coverage"]
    r1.setdefault("human_total", 0)
    r1.setdefault("matched", 0)
    r1.setdefault("coverage", 0)
    r1.setdefault("coverage_percent", 0)

    if not isinstance(obj.get("round2_impact_coverage"), dict):
        obj["round2_impact_coverage"] = {}
    r2 = obj["round2_impact_coverage"]
    total_human = 0
    total_matched = 0
    for level in IMPACT_LEVELS:
        if not isinstance(r2.get(level), dict):
            r2[level] = {}
        r2[level].setdefault("human_total", 0)
        r2[level].setdefault("matched", 0)
        r2[level].setdefault("coverage", 0)
        r2[level].setdefault("coverage_percent", 0)
        try:
            human_total = int(r2[level].get("human_total") or 0)
        except Exception:
            human_total = 0
        try:
            matched = int(r2[level].get("matched") or 0)
        except Exception:
            matched = 0

        human_total = max(0, human_total)
        matched = max(0, min(matched, human_total))
        coverage = 1 if human_total == 0 else round(matched / human_total, 4)

        r2[level]["human_total"] = human_total
        r2[level]["matched"] = matched
        r2[level]["coverage"] = coverage
        r2[level]["coverage_percent"] = round(coverage * 100, 2)

        total_human += human_total
        total_matched += matched

    total_coverage = 1 if total_human == 0 else round(total_matched / total_human, 4)
    r2["total_human_total"] = total_human
    r2["total_matched"] = total_matched
    r2["total_coverage"] = total_coverage
    r2["total_coverage_percent"] = round(total_coverage * 100, 2)

    # Backward-compatible fields retained, but now aligned to total coverage.
    r2["macro_coverage"] = total_coverage
    r2["macro_coverage_percent"] = round(total_coverage * 100, 2)

    _normalize_strict_coverage(obj)

    obj["missed_human_root_causes"] = _normalize_root_cause_items(
        obj.get("missed_human_root_causes", []), "human"
    )
    obj["extra_auto_root_causes"] = _normalize_root_cause_items(
        obj.get("extra_auto_root_causes", []), "auto"
    )
    obj.setdefault("analysis", "")
    return obj


def get_thread_client(api_key: str, base_url: str):
    client = getattr(_thread_local, "client", None)
    if client is None:
        _thread_local.client = OpenAI(api_key=api_key, base_url=base_url)
        client = _thread_local.client
    return client


def resolve_max_workers(requested_workers: int, total_files: int):
    if total_files <= 0:
        return 1
    if requested_workers and requested_workers > 0:
        return min(requested_workers, total_files)

    cpu_count = os.cpu_count() or 4
    auto_workers = max(4, cpu_count * 4)
    return min(32, auto_workers, total_files)


def process_trace_file(
    gaia_file: Path,
    ann_dir: Path,
    judge_dir: Path,
    align_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
    max_input_tokens: int,
    max_input_chars: int,
    model_context_tokens: int,
    max_output_tokens: int,
    token_safety_buffer: int,
):
    trace_id = gaia_file.stem
    row = {"trace_id": trace_id, "status": "ok", "issues": []}

    try:
        client = get_thread_client(api_key, base_url)
        raw_text = gaia_file.read_text(encoding="utf-8")
        judged, usage, judge_meta = call_unprocessed_judge(
            client,
            model,
            trace_id,
            raw_text,
            max_input_tokens=max_input_tokens,
            max_input_chars=max_input_chars,
            model_context_tokens=model_context_tokens,
            max_output_tokens=max_output_tokens,
            token_safety_buffer=token_safety_buffer,
        )
        judged = normalize_judge_result(judged, trace_id)
        judged["meta"] = {"gaia_file": str(gaia_file), **judge_meta}

        (judge_dir / f"{trace_id}.json").write_text(
            json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        ann_file = ann_dir / f"{trace_id}.json"
        if not ann_file.exists():
            raise RuntimeError(f"人工标注不存在: {ann_file}")

        human = load_json_lenient(ann_file)
        compared = call_compare(client, model, trace_id, human, judged)
        compared = normalize_compare_result(compared, trace_id)

        (align_dir / f"{trace_id}.json").write_text(
            json.dumps(compared, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        row.update(
            {
                "round1_coverage_percent": compared.get("round1_root_cause_coverage", {}).get("coverage_percent"),
                "round1_total_matched": compared.get("round1_root_cause_coverage", {}).get("matched"),
                "round1_total_human_total": compared.get("round1_root_cause_coverage", {}).get("human_total"),
                "round2_coverage_percent": compared.get("round2_impact_coverage", {}).get("total_coverage_percent"),
                "round2_total_matched": compared.get("round2_impact_coverage", {}).get("total_matched"),
                "round2_total_human_total": compared.get("round2_impact_coverage", {}).get("total_human_total"),
                "round2_macro_coverage_percent": compared.get("round2_impact_coverage", {}).get("total_coverage_percent"),
                "strict_round1_coverage_percent": compared.get("strict_round1_root_cause_coverage", {}).get("coverage_percent"),
                "strict_round1_total_matched": compared.get("strict_round1_root_cause_coverage", {}).get("matched"),
                "strict_round1_total_human_total": compared.get("strict_round1_root_cause_coverage", {}).get("human_total"),
                "strict_round2_coverage_percent": compared.get("strict_round2_impact_coverage", {}).get("total_coverage_percent"),
                "strict_round2_total_matched": compared.get("strict_round2_impact_coverage", {}).get("total_matched"),
                "strict_round2_total_human_total": compared.get("strict_round2_impact_coverage", {}).get("total_human_total"),
                "strict_round2_macro_coverage_percent": compared.get("strict_round2_impact_coverage", {}).get("total_coverage_percent"),
                "token_usage": usage,
                "truncated": judge_meta.get("truncated", False),
                "input_chars": judge_meta.get("input_chars"),
                "input_estimated_tokens": judge_meta.get("input_estimated_tokens"),
                "effective_max_input_tokens": judge_meta.get("effective_max_input_tokens"),
                "effective_max_input_chars": judge_meta.get("effective_max_input_chars"),
            }
        )
    except Exception as e:
        row["status"] = "failed"
        row["issues"].append(str(e))

    return row


def run(args):
    global DEEPSEEK_THINKING_MODE
    if OpenAI is None:
        raise RuntimeError("未安装 openai 包，请先安装：pip install openai")

    DEEPSEEK_THINKING_MODE = args.thinking_mode

    api_key = args.api_key or DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 API Key。请通过 --api-key 或环境变量 DEEPSEEK_API_KEY 提供。")

    gaia_dir = Path(args.gaia_dir)
    ann_dir = Path(args.annotations_dir)
    out_dir = Path(args.out_dir)
    judge_dir = out_dir / "unprocessed_judgements"
    align_dir = out_dir / "unprocessed_vs_human"
    judge_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(gaia_dir.glob("*.json"))
    if args.trace_id:
        files = [p for p in files if p.stem == args.trace_id]
    if args.max_samples and args.max_samples > 0:
        files = files[: args.max_samples]

    if not files:
        raise RuntimeError("未找到可处理的 GAIA trace 文件。")

    max_workers = resolve_max_workers(args.max_workers, len(files))
    print(f"并发 worker 数: {max_workers}")

    indexed_rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_trace_file,
                f,
                ann_dir,
                judge_dir,
                align_dir,
                api_key,
                args.base_url,
                args.model,
                args.max_input_tokens,
                args.max_input_chars,
                args.model_context_tokens,
                args.max_output_tokens,
                args.token_safety_buffer,
            ): idx
            for idx, f in enumerate(files, start=1)
        }

        done = 0
        total = len(files)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            row = future.result()
            indexed_rows.append((idx, row))
            done += 1
            print(f"[{done}/{total}] {row['trace_id']}: {row['status']}")

    indexed_rows.sort(key=lambda x: x[0])
    rows = [r for _, r in indexed_rows]

    valid_rows = [r for r in rows if r.get("status") == "ok"]

    def _avg(key):
        vals = [r.get(key) for r in valid_rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    round1_total_matched = sum((r.get("round1_total_matched") or 0) for r in valid_rows)
    round1_total_human_total = sum((r.get("round1_total_human_total") or 0) for r in valid_rows)
    round1_total_coverage_percent = (
        round(round1_total_matched / round1_total_human_total * 100, 4)
        if round1_total_human_total
        else None
    )
    round2_total_matched = sum((r.get("round2_total_matched") or 0) for r in valid_rows)
    round2_total_human_total = sum((r.get("round2_total_human_total") or 0) for r in valid_rows)
    round2_total_coverage_percent = (
        round(round2_total_matched / round2_total_human_total * 100, 4)
        if round2_total_human_total
        else None
    )
    strict_round1_total_matched = sum((r.get("strict_round1_total_matched") or 0) for r in valid_rows)
    strict_round1_total_human_total = sum((r.get("strict_round1_total_human_total") or 0) for r in valid_rows)
    strict_round1_total_coverage_percent = (
        round(strict_round1_total_matched / strict_round1_total_human_total * 100, 4)
        if strict_round1_total_human_total
        else None
    )
    strict_round2_total_matched = sum((r.get("strict_round2_total_matched") or 0) for r in valid_rows)
    strict_round2_total_human_total = sum((r.get("strict_round2_total_human_total") or 0) for r in valid_rows)
    strict_round2_total_coverage_percent = (
        round(strict_round2_total_matched / strict_round2_total_human_total * 100, 4)
        if strict_round2_total_human_total
        else None
    )

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "processed": len(valid_rows),
        "failed": len(rows) - len(valid_rows),
        "aggregates": {
            "avg_round1_coverage_percent": _avg("round1_coverage_percent"),
            "total_round1_matched": round1_total_matched,
            "total_round1_human_total": round1_total_human_total,
            "total_round1_coverage_percent": round1_total_coverage_percent,
            "avg_round2_coverage_percent": _avg("round2_coverage_percent"),
            "total_round2_matched": round2_total_matched,
            "total_round2_human_total": round2_total_human_total,
            "total_round2_coverage_percent": round2_total_coverage_percent,
            "avg_round2_macro_coverage_percent": _avg("round2_coverage_percent"),
            "avg_strict_round1_coverage_percent": _avg("strict_round1_coverage_percent"),
            "total_strict_round1_matched": strict_round1_total_matched,
            "total_strict_round1_human_total": strict_round1_total_human_total,
            "total_strict_round1_coverage_percent": strict_round1_total_coverage_percent,
            "avg_strict_round2_coverage_percent": _avg("strict_round2_coverage_percent"),
            "total_strict_round2_matched": strict_round2_total_matched,
            "total_strict_round2_human_total": strict_round2_total_human_total,
            "total_strict_round2_coverage_percent": strict_round2_total_coverage_percent,
            "avg_strict_round2_macro_coverage_percent": _avg("strict_round2_coverage_percent"),
        },
        "rows": rows,
    }

    summary_path = out_dir / "unprocessed_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n完成：")
    print(f"- processed: {summary['processed']}")
    print(f"- failed: {summary['failed']}")
    print(f"- summary: {summary_path}")


def build_cli():
    p = argparse.ArgumentParser(
        description="Judge unprocessed GAIA trace text directly and align against human annotations"
    )
    p.add_argument("--gaia-dir", default="trail_data/GAIA", help="Directory of raw GAIA trace json files")
    p.add_argument("--annotations-dir", default="trail_data/processed_annotations_gaia", help="Directory of human annotations")
    p.add_argument("--out-dir", default="output_graphs/deepseek_unprocessed", help="Output directory")
    p.add_argument("--trace-id", default=None, help="Run one trace_id only")
    p.add_argument("--max-samples", type=int, default=0, help="Max samples to process, 0 means all")
    p.add_argument("--model", default=DEFAULT_MODEL, help="DeepSeek model")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="DeepSeek base URL")
    p.add_argument("--thinking-mode", choices=["disabled", "enabled"], default="disabled", help="DeepSeek V4 thinking mode")
    p.add_argument("--api-key", default=None, help="DeepSeek API key (or use DEEPSEEK_API_KEY / OPENAI_API_KEY env)")
    p.add_argument("--max-workers", type=int, default=0, help="Parallel workers, 0 means auto")
    p.add_argument(
        "--model-context-tokens",
        type=int,
        default=DEFAULT_MODEL_CONTEXT_TOKENS,
        help="Model context window used to compute safe prompt budget",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help="Reserved completion tokens for the judge response",
    )
    p.add_argument(
        "--token-safety-buffer",
        type=int,
        default=DEFAULT_TOKEN_SAFETY_BUFFER,
        help="Extra prompt-token safety margin kept below the model context limit",
    )
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help=(
            "Max estimated input tokens from raw trace text; "
            "0 means auto-safe token limit derived from model context"
        ),
    )
    p.add_argument(
        "--max-input-chars",
        type=int,
        default=0,
        help=(
            "Legacy secondary char cap applied after token truncation; "
            "0 means no extra char cap"
        ),
    )
    return p


if __name__ == "__main__":
    args = build_cli().parse_args()
    run(args)
