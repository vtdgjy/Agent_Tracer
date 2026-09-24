import argparse
import json
import os
import re
import time
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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
DEEPSEEK_THINKING_MODE = "disabled"
DEFAULT_COMPARE_MAX_TOKENS = 0

IMPACT_LEVELS = ["LOW", "MEDIUM", "HIGH"]
ID_IN_TEXT_RE = re.compile(r"\b(?:[0-9a-f]{16}|[A-Z]_[0-9]+)\b", re.IGNORECASE)
LOCATION_RELATIONS = {
    "same_location",
    "same_logical_chain_adjacent",
    "same_logical_chain_upstream",
    "same_logical_chain_downstream",
    "same_local_cluster",
    "different_chain",
    "unknown",
}
CHAIN_AWARE_LOCATION_RELATIONS = {
    "same_location",
    "same_logical_chain_adjacent",
    "same_logical_chain_upstream",
    "same_logical_chain_downstream",
    "same_local_cluster",
}

def load_json(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", text))


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_balanced_json_object(text: str):
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _cleanup_json_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def parse_json_from_text(text: str):
    stripped = _strip_code_fences(text)
    candidates = []
    if stripped:
        candidates.append(stripped)

    balanced = _extract_balanced_json_object(stripped)
    if balanced and balanced not in candidates:
        candidates.append(balanced)

    regex_match = re.search(r"\{[\s\S]*\}", stripped)
    if regex_match:
        maybe = regex_match.group(0)
        if maybe not in candidates:
            candidates.append(maybe)

    for candidate in list(candidates):
        cleaned = _cleanup_json_text(candidate)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    last_error = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"LLM did not return valid JSON: {last_error}")


def make_compare_prompt(trace_id: str, human_ann: dict, ds_judge: dict, source_name: str):
    return f"""
你是一个严格的“自动标注 vs 人工标注”覆盖率评估器。

请比较以下两份关于同一个 trace 的错误根因标注：
- trace_id: {trace_id}
- 自动标注来源: {source_name}

【人工标注】
{json.dumps(human_ann, ensure_ascii=False, indent=2)}

【自动标注】
{json.dumps(ds_judge, ensure_ascii=False, indent=2)}

Chain-aware note:
If the automatic annotation contains `causal_chains`, use `fault_origin`, `fault_usage`,
`final_failure`, and `causal_path` as strong evidence for same-chain matching. A human
location at the origin and an automatic location at the usage node should count as
chain-aware matched when both describe the same propagated failure.
If `root_causes[].location_disambiguation.policy` is `type_conditioned_human_anchor`,
treat the current `location` as the human-preferred exact anchor; use
`causal_origin_location`, `original_location`, and `causal_chains` as same-chain context.
For older `origin_first` outputs, treat the current `location` as the preferred exact
location and `original_location` as downstream context.
Exact-location matching should compare the human location against the automatic
human-preferred `location` / `human_preferred_location`. Chain-aware matching should
also accept `causal_origin_location`, `original_location`, and any node/span on
`causal_chains[].causal_path` when they describe the same propagated failure.

请输出严格 JSON（不要 markdown 代码块），按两轮评估：
{{
  "trace_id": "{trace_id}",
  "source": "{source_name}",
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
        "macro_coverage": 0-1,
        "macro_coverage_percent": 0-100
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
        "macro_coverage": 0-1,
        "macro_coverage_percent": 0-100
    }},
    "exact_strict_round1_root_cause_coverage": {{
        "human_total": 整数,
        "matched": 整数,
        "coverage": 0-1,
        "coverage_percent": 0-100
    }},
    "exact_strict_round2_impact_coverage": {{
        "LOW": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
        "MEDIUM": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
        "HIGH": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
        "macro_coverage": 0-1,
        "macro_coverage_percent": 0-100
    }},
    "chain_aware_round1_root_cause_coverage": {{
        "human_total": 整数,
        "matched": 整数,
        "coverage": 0-1,
        "coverage_percent": 0-100
    }},
    "chain_aware_round2_impact_coverage": {{
        "LOW": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
        "MEDIUM": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
        "HIGH": {{"human_total": 整数, "matched": 整数, "coverage": 0-1, "coverage_percent": 0-100}},
        "macro_coverage": 0-1,
        "macro_coverage_percent": 0-100
    }},
    "matched_pairs": [
        {{
            "human_location": "span/node id 或 unknown",
            "auto_location": "span/node id 或 unknown",
            "human_impact": "LOW|MEDIUM|HIGH|UNKNOWN",
            "auto_impact": "LOW|MEDIUM|HIGH|UNKNOWN",
            "result_match": true,
            "exact_location_match": true,
            "location_match": true,
            "chain_aware_location_match": true,
            "location_relation": "same_location|same_logical_chain_adjacent|same_logical_chain_upstream|same_logical_chain_downstream|same_local_cluster|different_chain|unknown",
            "impact_match": true,
            "exact_strict_match": true,
            "chain_aware_strict_match": true,
            "strict_match": true,
            "match_reason": "语义或位置匹配理由",
            "location_match_reason": "说明 location 相同、相邻链路等价或不匹配的判断理由"
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
第一轮（忽略影响等级）：
1) 只看根因语义和定位是否可对应（location 相同或语义同义都算匹配）。
2) 覆盖率 = matched / human_total。

第二轮（考虑影响等级）：
1) 在已匹配根因基础上，按 LOW/MEDIUM/HIGH 分别统计覆盖率。
2) 每档覆盖率 = 该档 matched / 该档 human_total；如果该档 human_total=0，coverage=1。
3) macro_coverage 为 LOW/MEDIUM/HIGH 三档 coverage 的平均值。

严格匹配策略（新增字段，不要替代上面的当前策略）：
1) `round1_root_cause_coverage` 和 `round2_impact_coverage` 继续使用当前宽松策略。
2) 对每个 `matched_pairs` 同时判断：
   - `result_match`: 自动根因/结果含义是否与人工根因一致，不考虑位置。
   - `location_relation`: 两个 location 的关系，必须取以下枚举之一：
     `same_location`、`same_logical_chain_adjacent`、`same_logical_chain_upstream`、`same_logical_chain_downstream`、`same_local_cluster`、`different_chain`、`unknown`。
   - `location_match`: 自动 location 是否可视为定位正确。以下情况算 true：
     a) 两者指向同一 span_id/node_id 或同一明确 trace 片段；
     b) 两者不是同一 location，但语义根因相同或高度相似，并且处在同一逻辑因果链的相邻/紧邻上下游位置；
     c) 人工标注偏向“错误搜索结果/错误观察/错误证据产生处”，自动标注偏向“使用该错误搜索结果/错误证据进行行动、推理或最终回答处”，或反过来；只要二者描述的是同一错误链上的原因-使用关系，就算 location_match=true；
     d) 人工标注偏向“工具调用失败/检索失败”，自动标注偏向“未处理该失败而继续计划/行动”，或反过来；只要是同一失败传播链上的相邻节点，就算 location_match=true。
     以下情况必须算 false：仅主题相似但不在同一任务链、相距很远且没有因果传播关系、自动定位到另一个独立错误、任一侧 location 为 unknown 且无法从 evidence/reason 中确定链路关系。
   - `impact_match`: 自动 impact 是否与人工 impact 一致；UNKNOWN 不算一致，除非两侧都 UNKNOWN。
   - `exact_location_match`: 只在两侧 location 指向同一 span_id/node_id 或同一明确 trace 片段时为 true。
   - `chain_aware_location_match`: 等同旧 `location_match`，允许同一错误传播链上的相邻/紧邻上下游位置视为定位可接受。
   - `exact_strict_match`: 必须等于 `result_match && exact_location_match`。
   - `chain_aware_strict_match`: 必须等于 `result_match && chain_aware_location_match`。
   - `strict_match`: 作为兼容字段，必须等于 `chain_aware_strict_match`。
3) `strict_round1_root_cause_coverage.matched` 只统计 `result_match && location_match` 都为 true 的人工根因。
4) `strict_round2_impact_coverage` 在严格匹配基础上再按 LOW/MEDIUM/HIGH 统计，若自动 impact 与人工 impact 不一致，不计入该档 matched。
5) `exact_strict_*` 统计 exact 严格口径；`chain_aware_*` 与旧 `strict_*` 统计同链容错严格口径。
6) 必须在 `location_match_reason` 中简短说明为什么位置可等价或不可等价，尤其要说明是否属于“错误信息产生处 vs 错误信息使用处”的相邻链路偏移。

注意：
- 匹配时允许文字不同，但要求“含义一致”。
- location 可作为强信号，但不是唯一信号；当两个位置处于同一错误传播链的相邻节点时，应视为定位可接受，而不是简单判错。
- `missed_human_root_causes` 和 `extra_auto_root_causes` 必须输出为对象数组，不要只输出字符串。
- 若能从人工/自动标注中定位到 span_id/node_id，location 不允许填 unknown。
- 输出必须是合法 JSON。
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


def _normalize_location_relation(value, exact_location_match: bool, chain_aware_location_match: bool) -> str:
    relation = str(value or "").strip()
    if exact_location_match:
        return "same_location"
    if chain_aware_location_match and relation in CHAIN_AWARE_LOCATION_RELATIONS:
        return relation
    if chain_aware_location_match:
        return "same_logical_chain_adjacent"
    if relation in LOCATION_RELATIONS:
        return relation
    return "different_chain"


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
        exact_location_match = _location_matches(human_location, auto_location)

        if "result_match" in pair:
            result_match = _safe_bool(pair.get("result_match"), default=False)
        else:
            # Backward compatibility: old matched_pairs already represented a result-level match.
            result_match = True

        if "chain_aware_location_match" in pair:
            chain_aware_location_match = _safe_bool(pair.get("chain_aware_location_match"), default=False)
        elif "location_match" in pair:
            chain_aware_location_match = _safe_bool(pair.get("location_match"), default=False)
        else:
            chain_aware_location_match = exact_location_match

        if "impact_match" in pair:
            impact_match = _safe_bool(pair.get("impact_match"), default=False)
        else:
            impact_match = human_impact == auto_impact

        exact_strict_match = bool(result_match and exact_location_match)
        chain_aware_strict_match = bool(result_match and chain_aware_location_match)

        pair["human_location"] = human_location
        pair["auto_location"] = auto_location
        pair["human_impact"] = human_impact
        pair["auto_impact"] = auto_impact
        pair["result_match"] = bool(result_match)
        pair["exact_location_match"] = bool(exact_location_match)
        pair["chain_aware_location_match"] = bool(chain_aware_location_match)
        # Legacy aliases keep old readers working. They now mean chain-aware strict matching.
        pair["location_match"] = bool(chain_aware_location_match)
        pair["location_relation"] = _normalize_location_relation(
            pair.get("location_relation"),
            exact_location_match,
            chain_aware_location_match,
        )
        pair["location_match_reason"] = str(pair.get("location_match_reason") or "").strip()
        pair["impact_match"] = bool(impact_match)
        pair["exact_strict_match"] = exact_strict_match
        pair["chain_aware_strict_match"] = chain_aware_strict_match
        pair["strict_match"] = chain_aware_strict_match
        normalized.append(pair)

    obj["matched_pairs"] = normalized
    obj["exact_strict_matched_pairs"] = [pair for pair in normalized if pair.get("exact_strict_match")]
    obj["chain_aware_strict_matched_pairs"] = [
        pair for pair in normalized if pair.get("chain_aware_strict_match")
    ]
    # Legacy alias: strict now follows the chain-aware policy.
    obj["strict_matched_pairs"] = obj["chain_aware_strict_matched_pairs"]
    return normalized


def _count_pairs_by_impact(pairs: list, match_key: str) -> dict:
    counts = {level: 0 for level in IMPACT_LEVELS}
    for pair in pairs:
        if not pair.get(match_key) or not pair.get("impact_match"):
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


def _build_policy_round1(obj: dict, field_name: str, human_total: int, matched: int, pairs: list) -> dict:
    block = obj.get(field_name)
    if not isinstance(block, dict):
        block = {}
    block["human_total"] = human_total
    if "matched" in block:
        block["matched"] = min(_safe_int(block.get("matched"), 0), matched)
    else:
        block["matched"] = matched
    return _normalize_coverage_block(
        block,
        default_human_total=human_total,
        default_matched=matched,
        empty_coverage=0.0,
    )


def _build_policy_round2(obj: dict, field_name: str, base_r2: dict, pairs: list, match_key: str) -> dict:
    block = obj.get(field_name)
    if not isinstance(block, dict):
        block = {}

    derived_by_impact = _count_pairs_by_impact(pairs, match_key)
    coverage_values = []
    for level in IMPACT_LEVELS:
        base_level = base_r2.get(level, {}) if isinstance(base_r2, dict) else {}
        level_block = block.get(level, {}) if isinstance(block.get(level), dict) else {}
        level_block["human_total"] = max(0, _safe_int(base_level.get("human_total"), 0))
        if "matched" in level_block:
            level_block["matched"] = min(
                _safe_int(level_block.get("matched"), 0),
                derived_by_impact[level],
            )
        else:
            level_block["matched"] = derived_by_impact[level]
        block[level] = _normalize_coverage_block(
            level_block,
            default_human_total=level_block["human_total"],
            default_matched=derived_by_impact[level],
            empty_coverage=1.0,
        )
        coverage_values.append(block[level]["coverage"])

    macro_coverage = round(sum(coverage_values) / len(coverage_values), 4) if coverage_values else 0.0
    block["macro_coverage"] = macro_coverage
    block["macro_coverage_percent"] = round(macro_coverage * 100, 2)
    return block


def _normalize_strict_coverage(obj: dict) -> None:
    pairs = _normalize_matched_pairs(obj)

    r1 = obj.get("round1_root_cause_coverage", {})
    base_human_total = max(0, _safe_int(r1.get("human_total"), 0))
    base_r2 = obj.get("round2_impact_coverage", {})

    exact_matched = sum(1 for pair in pairs if pair.get("exact_strict_match"))
    chain_aware_matched = sum(1 for pair in pairs if pair.get("chain_aware_strict_match"))

    obj["exact_strict_round1_root_cause_coverage"] = _build_policy_round1(
        obj,
        "exact_strict_round1_root_cause_coverage",
        base_human_total,
        exact_matched,
        pairs,
    )
    obj["exact_strict_round2_impact_coverage"] = _build_policy_round2(
        obj,
        "exact_strict_round2_impact_coverage",
        base_r2,
        pairs,
        "exact_strict_match",
    )

    obj["chain_aware_round1_root_cause_coverage"] = _build_policy_round1(
        obj,
        "chain_aware_round1_root_cause_coverage",
        base_human_total,
        chain_aware_matched,
        pairs,
    )
    obj["chain_aware_round2_impact_coverage"] = _build_policy_round2(
        obj,
        "chain_aware_round2_impact_coverage",
        base_r2,
        pairs,
        "chain_aware_strict_match",
    )

    # Backward compatibility: old strict fields now point to the chain-aware strict policy.
    obj["strict_round1_root_cause_coverage"] = obj["chain_aware_round1_root_cause_coverage"]
    obj["strict_round2_impact_coverage"] = obj["chain_aware_round2_impact_coverage"]


def make_json_repair_prompt(raw_text: str):
    return f"""
Repair the following malformed JSON and output exactly one valid JSON object.

Rules:
- Preserve the original field names and values as much as possible.
- Do not add markdown fences or any explanations.
- Escape any unescaped quotes inside string values.
- Remove trailing commas if present.

Malformed text:
{raw_text}
""".strip()


def _chat_json_once(client, model: str, system_prompt: str, user_prompt: str, max_tokens: int = DEFAULT_COMPARE_MAX_TOKENS):
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }
    if max_tokens and max_tokens > 0:
        request["max_tokens"] = max_tokens
    base_url = str(getattr(client, "base_url", "") or "")
    if "api.deepseek.com" in base_url and model.startswith("deepseek"):
        request["extra_body"] = {"thinking": {"type": DEEPSEEK_THINKING_MODE}}

    resp = client.chat.completions.create(**request)
    return (resp.choices[0].message.content or "").strip()


def _repair_json_with_model(client, model: str, raw_text: str, max_tokens: int = DEFAULT_COMPARE_MAX_TOKENS):
    repaired_text = _chat_json_once(
        client,
        model,
        "You repair malformed JSON. Output only one valid JSON object.",
        make_json_repair_prompt(raw_text),
        max_tokens=max_tokens,
    )
    return parse_json_from_text(repaired_text)


def call_deepseek_compare(
    client,
    model: str,
    trace_id: str,
    human_ann: dict,
    ds_judge: dict,
    source_name: str,
    max_retries: int = 3,
    compare_max_tokens: int = DEFAULT_COMPARE_MAX_TOKENS,
):
    prompt = make_compare_prompt(trace_id, human_ann, ds_judge, source_name)
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            text = _chat_json_once(
                client,
                model,
                "You are a strict evaluator. Output only valid JSON.",
                prompt,
                max_tokens=compare_max_tokens,
            )
            try:
                return parse_json_from_text(text)
            except Exception:
                try:
                    return _repair_json_with_model(client, model, text, max_tokens=compare_max_tokens)
                except Exception as repair_exc:
                    last_error = repair_exc
        except Exception as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(min(6, attempt * 2))

    raise last_error or RuntimeError("DeepSeek compare call failed without a concrete error")


def normalize_compare_result(obj: dict, trace_id: str, source_name: str):
    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("trace_id", trace_id)
    obj.setdefault("source", source_name)
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
    for level in ["LOW", "MEDIUM", "HIGH"]:
        if not isinstance(r2.get(level), dict):
            r2[level] = {}
        r2[level].setdefault("human_total", 0)
        r2[level].setdefault("matched", 0)
        r2[level].setdefault("coverage", 0)
        r2[level].setdefault("coverage_percent", 0)
    r2.setdefault("macro_coverage", 0)
    r2.setdefault("macro_coverage_percent", 0)

    _normalize_strict_coverage(obj)

    obj["missed_human_root_causes"] = _normalize_root_cause_items(
        obj.get("missed_human_root_causes", []), "human"
    )
    obj["extra_auto_root_causes"] = _normalize_root_cause_items(
        obj.get("extra_auto_root_causes", []), "auto"
    )
    obj.setdefault("analysis", "")
    return obj


def extract_hit_metrics(compare_result: dict, strict: bool = False, policy: str = "semantic"):
    if strict and policy == "semantic":
        policy = "chain_aware"

    if policy == "exact_strict":
        round1 = compare_result.get("exact_strict_round1_root_cause_coverage", {})
        round2 = compare_result.get("exact_strict_round2_impact_coverage", {})
    elif policy in {"chain_aware", "strict"}:
        round1 = (
            compare_result.get("chain_aware_round1_root_cause_coverage")
            or compare_result.get("strict_round1_root_cause_coverage", {})
        )
        round2 = (
            compare_result.get("chain_aware_round2_impact_coverage")
            or compare_result.get("strict_round2_impact_coverage", {})
        )
    else:
        round1 = compare_result.get("round1_root_cause_coverage", {})
        round2 = compare_result.get("round2_impact_coverage", {})

    metrics = {
        "overall": {
            "human_total": int(round1.get("human_total", 0) or 0),
            "matched": int(round1.get("matched", 0) or 0),
        },
        "by_human_impact": {},
    }

    for level in IMPACT_LEVELS:
        level_obj = round2.get(level, {})
        metrics["by_human_impact"][level] = {
            "human_total": int(level_obj.get("human_total", 0) or 0),
            "matched": int(level_obj.get("matched", 0) or 0),
        }

    return metrics


def finalize_hit_rate(stats: dict):
    human_total = stats.get("human_total", 0)
    matched = stats.get("matched", 0)
    coverage = (matched / human_total) if human_total else 0.0
    stats["coverage"] = round(coverage, 6)
    stats["coverage_percent"] = round(coverage * 100, 4)
    return stats


def init_source_hit_summary():
    return {
        "overall": {"human_total": 0, "matched": 0},
        "by_human_impact": {
            level: {"human_total": 0, "matched": 0}
            for level in IMPACT_LEVELS
        },
    }


def accumulate_hit_summary(target: dict, metrics: dict):
    target["overall"]["human_total"] += metrics["overall"]["human_total"]
    target["overall"]["matched"] += metrics["overall"]["matched"]

    for level in IMPACT_LEVELS:
        target_level = target["by_human_impact"][level]
        metrics_level = metrics["by_human_impact"][level]
        target_level["human_total"] += metrics_level["human_total"]
        target_level["matched"] += metrics_level["matched"]


def finalize_source_hit_summary(summary: dict):
    finalize_hit_rate(summary["overall"])
    for level in IMPACT_LEVELS:
        finalize_hit_rate(summary["by_human_impact"][level])
    return summary


def _init_location_source_summary():
    return {
        "total_pairs": 0,
        "result_match_pairs": 0,
        "exact_location_match_pairs": 0,
        "chain_aware_location_match_pairs": 0,
        "exact_strict_pairs": 0,
        "chain_aware_strict_pairs": 0,
        "accepted_chain_offset_pairs": 0,
        "rejected_location_mismatch_pairs": 0,
        "unknown_location_pairs": 0,
        "relation_counts": {relation: 0 for relation in sorted(LOCATION_RELATIONS)},
    }


def init_location_audit_bundle():
    return {
        "summary": {
            "raw_trace": _init_location_source_summary(),
            "trace_graph": _init_location_source_summary(),
        },
        "accepted_chain_offsets": [],
        "rejected_location_mismatches": [],
        "unknown_location_pairs": [],
    }


def _is_unknown_location(value) -> bool:
    text = str(value or "").strip()
    return not text or text.casefold() == "unknown"


def _location_audit_record(trace_id: str, source_name: str, pair: dict) -> dict:
    return {
        "trace_id": trace_id,
        "source": source_name,
        "human_location": pair.get("human_location", "unknown"),
        "auto_location": pair.get("auto_location", "unknown"),
        "human_impact": pair.get("human_impact", "UNKNOWN"),
        "auto_impact": pair.get("auto_impact", "UNKNOWN"),
        "result_match": bool(pair.get("result_match")),
        "exact_location_match": bool(pair.get("exact_location_match")),
        "chain_aware_location_match": bool(
            pair.get("chain_aware_location_match", pair.get("location_match"))
        ),
        "exact_strict_match": bool(pair.get("exact_strict_match")),
        "chain_aware_strict_match": bool(
            pair.get("chain_aware_strict_match", pair.get("strict_match"))
        ),
        "impact_match": bool(pair.get("impact_match")),
        "location_relation": pair.get("location_relation", "unknown"),
        "match_reason": pair.get("match_reason", ""),
        "location_match_reason": pair.get("location_match_reason", ""),
    }


def accumulate_location_audit(bundle: dict, compare_result: dict, trace_id: str, source_name: str):
    source_summary = bundle["summary"][source_name]
    for pair in compare_result.get("matched_pairs", []):
        if not isinstance(pair, dict):
            continue

        relation = str(pair.get("location_relation") or "unknown").strip()
        if relation not in LOCATION_RELATIONS:
            relation = "unknown"

        result_match = bool(pair.get("result_match"))
        exact_location_match = bool(pair.get("exact_location_match"))
        chain_aware_location_match = bool(
            pair.get("chain_aware_location_match", pair.get("location_match"))
        )
        exact_strict_match = bool(pair.get("exact_strict_match"))
        chain_aware_strict_match = bool(
            pair.get("chain_aware_strict_match", pair.get("strict_match"))
        )
        unknown_location = _is_unknown_location(pair.get("human_location")) or _is_unknown_location(
            pair.get("auto_location")
        )

        source_summary["total_pairs"] += 1
        source_summary["relation_counts"][relation] += 1
        if result_match:
            source_summary["result_match_pairs"] += 1
        if exact_location_match:
            source_summary["exact_location_match_pairs"] += 1
        if chain_aware_location_match:
            source_summary["chain_aware_location_match_pairs"] += 1
        if exact_strict_match:
            source_summary["exact_strict_pairs"] += 1
        if chain_aware_strict_match:
            source_summary["chain_aware_strict_pairs"] += 1
        if unknown_location:
            source_summary["unknown_location_pairs"] += 1

        record = _location_audit_record(trace_id, source_name, pair)
        if result_match and chain_aware_location_match and not exact_location_match:
            source_summary["accepted_chain_offset_pairs"] += 1
            bundle["accepted_chain_offsets"].append(record)
        if result_match and not chain_aware_location_match:
            source_summary["rejected_location_mismatch_pairs"] += 1
            bundle["rejected_location_mismatches"].append(record)
        if unknown_location:
            bundle["unknown_location_pairs"].append(record)


def finalize_location_audit_summary(summary: dict):
    for source_summary in summary.values():
        total = source_summary.get("total_pairs", 0)
        for key in [
            "result_match_pairs",
            "exact_location_match_pairs",
            "chain_aware_location_match_pairs",
            "exact_strict_pairs",
            "chain_aware_strict_pairs",
            "accepted_chain_offset_pairs",
            "rejected_location_mismatch_pairs",
            "unknown_location_pairs",
        ]:
            rate_key = key.replace("_pairs", "_rate_percent")
            source_summary[rate_key] = (
                round(source_summary[key] / total * 100, 4) if total else 0.0
            )
    return summary


def collect_location_audits(raw_cmp_dir: Path, graph_cmp_dir: Path, rows: list) -> dict:
    bundle = init_location_audit_bundle()
    for row in rows:
        trace_id = row.get("trace_id")
        if not trace_id:
            continue

        for source_name, cmp_dir in [("raw_trace", raw_cmp_dir), ("trace_graph", graph_cmp_dir)]:
            cmp_file = cmp_dir / f"{trace_id}.json"
            if not cmp_file.exists():
                continue
            cmp_obj = normalize_compare_result(load_json(cmp_file), trace_id, source_name)
            accumulate_location_audit(bundle, cmp_obj, trace_id, source_name)

    finalize_location_audit_summary(bundle["summary"])
    return bundle


def write_location_audit_files(out_dir: Path, bundle: dict) -> dict:
    audit_dir = out_dir / "location_audit"
    files = {
        "location_relation_summary": audit_dir / "location_relation_summary.json",
        "accepted_chain_offsets": audit_dir / "accepted_chain_offsets.json",
        "rejected_location_mismatches": audit_dir / "rejected_location_mismatches.json",
        "unknown_location_pairs": audit_dir / "unknown_location_pairs.json",
    }
    dump_json(files["location_relation_summary"], bundle["summary"])
    dump_json(
        files["accepted_chain_offsets"],
        {
            "count": len(bundle["accepted_chain_offsets"]),
            "items": bundle["accepted_chain_offsets"],
        },
    )
    dump_json(
        files["rejected_location_mismatches"],
        {
            "count": len(bundle["rejected_location_mismatches"]),
            "items": bundle["rejected_location_mismatches"],
        },
    )
    dump_json(
        files["unknown_location_pairs"],
        {
            "count": len(bundle["unknown_location_pairs"]),
            "items": bundle["unknown_location_pairs"],
        },
    )
    return {name: str(path) for name, path in files.items()}


def process_merged_file(
    client,
    model: str,
    merged_file: Path,
    ann_dir: Path,
    raw_cmp_dir: Path,
    graph_cmp_dir: Path,
    compare_max_tokens: int,
    resume: bool = False,
):
    trace_id = merged_file.stem
    row = {"trace_id": trace_id, "status": "ok", "issues": []}

    try:
        merged = load_json(merged_file)
        ann_file = ann_dir / f"{trace_id}.json"
        if not ann_file.exists():
            raise RuntimeError(f"人工标注不存在: {ann_file}")
        human = load_json(ann_file)

        raw_j = merged.get("raw_trace_judgement", {})
        graph_j = merged.get("graph_judgement", {})

        raw_cmp_file = raw_cmp_dir / f"{trace_id}.json"
        graph_cmp_file = graph_cmp_dir / f"{trace_id}.json"

        if resume and raw_cmp_file.exists():
            raw_cmp = normalize_compare_result(load_json(raw_cmp_file), trace_id, "raw_trace")
            raw_run_mode = "cached"
        else:
            raw_cmp = call_deepseek_compare(
                client,
                model,
                trace_id,
                human,
                raw_j,
                "raw_trace",
                compare_max_tokens=compare_max_tokens,
            )
            raw_cmp = normalize_compare_result(raw_cmp, trace_id, "raw_trace")
            raw_run_mode = "api"

        if resume and graph_cmp_file.exists():
            graph_cmp = normalize_compare_result(load_json(graph_cmp_file), trace_id, "trace_graph")
            graph_run_mode = "cached"
        else:
            graph_cmp = call_deepseek_compare(
                client,
                model,
                trace_id,
                human,
                graph_j,
                "trace_graph",
                compare_max_tokens=compare_max_tokens,
            )
            graph_cmp = normalize_compare_result(graph_cmp, trace_id, "trace_graph")
            graph_run_mode = "api"

        raw_hit_metrics = extract_hit_metrics(raw_cmp)
        graph_hit_metrics = extract_hit_metrics(graph_cmp)
        raw_exact_strict_hit_metrics = extract_hit_metrics(raw_cmp, policy="exact_strict")
        graph_exact_strict_hit_metrics = extract_hit_metrics(graph_cmp, policy="exact_strict")
        raw_chain_aware_hit_metrics = extract_hit_metrics(raw_cmp, policy="chain_aware")
        graph_chain_aware_hit_metrics = extract_hit_metrics(graph_cmp, policy="chain_aware")
        # Legacy strict metrics follow the chain-aware strict policy.
        raw_strict_hit_metrics = raw_chain_aware_hit_metrics
        graph_strict_hit_metrics = graph_chain_aware_hit_metrics

        dump_json(raw_cmp_file, raw_cmp)
        dump_json(graph_cmp_file, graph_cmp)

        raw_r1 = raw_cmp.get("round1_root_cause_coverage", {}).get("coverage_percent")
        graph_r1 = graph_cmp.get("round1_root_cause_coverage", {}).get("coverage_percent")
        raw_r2 = raw_cmp.get("round2_impact_coverage", {}).get("macro_coverage_percent")
        graph_r2 = graph_cmp.get("round2_impact_coverage", {}).get("macro_coverage_percent")
        raw_exact_strict_r1 = raw_cmp.get("exact_strict_round1_root_cause_coverage", {}).get("coverage_percent")
        graph_exact_strict_r1 = graph_cmp.get("exact_strict_round1_root_cause_coverage", {}).get("coverage_percent")
        raw_exact_strict_r2 = raw_cmp.get("exact_strict_round2_impact_coverage", {}).get("macro_coverage_percent")
        graph_exact_strict_r2 = graph_cmp.get("exact_strict_round2_impact_coverage", {}).get("macro_coverage_percent")
        raw_chain_aware_r1 = raw_cmp.get("chain_aware_round1_root_cause_coverage", {}).get("coverage_percent")
        graph_chain_aware_r1 = graph_cmp.get("chain_aware_round1_root_cause_coverage", {}).get("coverage_percent")
        raw_chain_aware_r2 = raw_cmp.get("chain_aware_round2_impact_coverage", {}).get("macro_coverage_percent")
        graph_chain_aware_r2 = graph_cmp.get("chain_aware_round2_impact_coverage", {}).get("macro_coverage_percent")
        # Legacy strict fields are kept as aliases for downstream scripts.
        raw_strict_r1 = raw_chain_aware_r1
        graph_strict_r1 = graph_chain_aware_r1
        raw_strict_r2 = raw_chain_aware_r2
        graph_strict_r2 = graph_chain_aware_r2

        raw_primary = raw_r1 if isinstance(raw_r1, (int, float)) else None
        graph_primary = graph_r1 if isinstance(graph_r1, (int, float)) else None
        better = "trace_graph"
        if isinstance(raw_primary, (int, float)) and isinstance(graph_primary, (int, float)):
            better = "raw_trace" if raw_primary >= graph_primary else "trace_graph"

        raw_strict_primary = raw_strict_r1 if isinstance(raw_strict_r1, (int, float)) else None
        graph_strict_primary = graph_strict_r1 if isinstance(graph_strict_r1, (int, float)) else None
        strict_better = "trace_graph"
        if isinstance(raw_strict_primary, (int, float)) and isinstance(graph_strict_primary, (int, float)):
            strict_better = "raw_trace" if raw_strict_primary >= graph_strict_primary else "trace_graph"
        chain_aware_better = strict_better

        raw_exact_primary = raw_exact_strict_r1 if isinstance(raw_exact_strict_r1, (int, float)) else None
        graph_exact_primary = graph_exact_strict_r1 if isinstance(graph_exact_strict_r1, (int, float)) else None
        exact_strict_better = "trace_graph"
        if isinstance(raw_exact_primary, (int, float)) and isinstance(graph_exact_primary, (int, float)):
            exact_strict_better = "raw_trace" if raw_exact_primary >= graph_exact_primary else "trace_graph"

        row.update(
            {
                "raw_round1_coverage": raw_r1,
                "graph_round1_coverage": graph_r1,
                "raw_round2_impact_macro": raw_r2,
                "graph_round2_impact_macro": graph_r2,
                "raw_exact_strict_round1_coverage": raw_exact_strict_r1,
                "graph_exact_strict_round1_coverage": graph_exact_strict_r1,
                "raw_exact_strict_round2_impact_macro": raw_exact_strict_r2,
                "graph_exact_strict_round2_impact_macro": graph_exact_strict_r2,
                "raw_chain_aware_round1_coverage": raw_chain_aware_r1,
                "graph_chain_aware_round1_coverage": graph_chain_aware_r1,
                "raw_chain_aware_round2_impact_macro": raw_chain_aware_r2,
                "graph_chain_aware_round2_impact_macro": graph_chain_aware_r2,
                "raw_strict_round1_coverage": raw_strict_r1,
                "graph_strict_round1_coverage": graph_strict_r1,
                "raw_strict_round2_impact_macro": raw_strict_r2,
                "graph_strict_round2_impact_macro": graph_strict_r2,
                "raw_hit_metrics": raw_hit_metrics,
                "graph_hit_metrics": graph_hit_metrics,
                "raw_exact_strict_hit_metrics": raw_exact_strict_hit_metrics,
                "graph_exact_strict_hit_metrics": graph_exact_strict_hit_metrics,
                "raw_chain_aware_hit_metrics": raw_chain_aware_hit_metrics,
                "graph_chain_aware_hit_metrics": graph_chain_aware_hit_metrics,
                "raw_strict_hit_metrics": raw_strict_hit_metrics,
                "graph_strict_hit_metrics": graph_strict_hit_metrics,
                "raw_run_mode": raw_run_mode,
                "graph_run_mode": graph_run_mode,
                "better": better,
                "exact_strict_better": exact_strict_better,
                "chain_aware_better": chain_aware_better,
                "strict_better": strict_better,
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

    merged_dir = Path(args.merged_dir)
    ann_dir = Path(args.annotations_dir)
    out_dir = Path(args.out_dir)
    raw_cmp_dir = out_dir / "raw_vs_human"
    graph_cmp_dir = out_dir / "graph_vs_human"
    raw_cmp_dir.mkdir(parents=True, exist_ok=True)
    graph_cmp_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    merged_files = sorted(merged_dir.glob("*.json"))
    if args.trace_id:
        merged_files = [p for p in merged_files if p.stem == args.trace_id]
    if args.max_samples and args.max_samples > 0:
        merged_files = merged_files[: args.max_samples]

    if not merged_files:
        raise RuntimeError("未找到 merged 判因文件，请先运行 deepseek_judge_trace.py")

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "thinking_mode": args.thinking_mode,
        "processed": 0,
        "failed": 0,
        "aggregates": {
            "avg_raw_round1_coverage": None,
            "avg_graph_round1_coverage": None,
            "avg_raw_round2_impact_macro": None,
            "avg_graph_round2_impact_macro": None,
            "avg_raw_exact_strict_round1_coverage": None,
            "avg_graph_exact_strict_round1_coverage": None,
            "avg_raw_exact_strict_round2_impact_macro": None,
            "avg_graph_exact_strict_round2_impact_macro": None,
            "avg_raw_chain_aware_round1_coverage": None,
            "avg_graph_chain_aware_round1_coverage": None,
            "avg_raw_chain_aware_round2_impact_macro": None,
            "avg_graph_chain_aware_round2_impact_macro": None,
            "avg_raw_strict_round1_coverage": None,
            "avg_graph_strict_round1_coverage": None,
            "avg_raw_strict_round2_impact_macro": None,
            "avg_graph_strict_round2_impact_macro": None,
            "global_hit_rate": {
                "raw_trace": init_source_hit_summary(),
                "trace_graph": init_source_hit_summary(),
            },
            "exact_strict_global_hit_rate": {
                "raw_trace": init_source_hit_summary(),
                "trace_graph": init_source_hit_summary(),
            },
            "chain_aware_global_hit_rate": {
                "raw_trace": init_source_hit_summary(),
                "trace_graph": init_source_hit_summary(),
            },
            "strict_global_hit_rate": {
                "raw_trace": init_source_hit_summary(),
                "trace_graph": init_source_hit_summary(),
            },
            "location_relation_summary": None,
            "location_audit_files": None,
            "run_modes": {
                "raw_trace": {"cached": 0, "api": 0},
                "trace_graph": {"cached": 0, "api": 0},
            },
        },
        "rows": [],
    }

    indexed_rows = []
    completed = 0
    max_workers = max(1, args.max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {
            executor.submit(
                process_merged_file,
                client,
                args.model,
                mfile,
                ann_dir,
                raw_cmp_dir,
                graph_cmp_dir,
                args.compare_max_tokens,
                args.resume,
            ): (idx, mfile.stem)
            for idx, mfile in enumerate(merged_files, start=1)
        }

        for future in as_completed(future_to_meta):
            idx, trace_id = future_to_meta[future]
            row = future.result()
            indexed_rows.append((idx, row))
            completed += 1
            print(f"[{completed}/{len(merged_files)}] {trace_id}: {row['status']}")

    indexed_rows.sort(key=lambda item: item[0])
    summary["rows"] = [row for _, row in indexed_rows]
    summary["processed"] = sum(1 for row in summary["rows"] if row.get("status") == "ok")
    summary["failed"] = sum(1 for row in summary["rows"] if row.get("status") == "failed")

    valid_rows = [r for r in summary["rows"] if r.get("status") == "ok"]
    if valid_rows:
        def _avg(key):
            vals = [r.get(key) for r in valid_rows if isinstance(r.get(key), (int, float))]
            return round(sum(vals) / len(vals), 4) if vals else None

        summary["aggregates"]["avg_raw_round1_coverage"] = _avg("raw_round1_coverage")
        summary["aggregates"]["avg_graph_round1_coverage"] = _avg("graph_round1_coverage")
        summary["aggregates"]["avg_raw_round2_impact_macro"] = _avg("raw_round2_impact_macro")
        summary["aggregates"]["avg_graph_round2_impact_macro"] = _avg("graph_round2_impact_macro")
        summary["aggregates"]["avg_raw_exact_strict_round1_coverage"] = _avg("raw_exact_strict_round1_coverage")
        summary["aggregates"]["avg_graph_exact_strict_round1_coverage"] = _avg("graph_exact_strict_round1_coverage")
        summary["aggregates"]["avg_raw_exact_strict_round2_impact_macro"] = _avg("raw_exact_strict_round2_impact_macro")
        summary["aggregates"]["avg_graph_exact_strict_round2_impact_macro"] = _avg("graph_exact_strict_round2_impact_macro")
        summary["aggregates"]["avg_raw_chain_aware_round1_coverage"] = _avg("raw_chain_aware_round1_coverage")
        summary["aggregates"]["avg_graph_chain_aware_round1_coverage"] = _avg("graph_chain_aware_round1_coverage")
        summary["aggregates"]["avg_raw_chain_aware_round2_impact_macro"] = _avg("raw_chain_aware_round2_impact_macro")
        summary["aggregates"]["avg_graph_chain_aware_round2_impact_macro"] = _avg("graph_chain_aware_round2_impact_macro")
        summary["aggregates"]["avg_raw_strict_round1_coverage"] = _avg("raw_strict_round1_coverage")
        summary["aggregates"]["avg_graph_strict_round1_coverage"] = _avg("graph_strict_round1_coverage")
        summary["aggregates"]["avg_raw_strict_round2_impact_macro"] = _avg("raw_strict_round2_impact_macro")
        summary["aggregates"]["avg_graph_strict_round2_impact_macro"] = _avg("graph_strict_round2_impact_macro")

        global_hit_rate = summary["aggregates"]["global_hit_rate"]
        exact_strict_global_hit_rate = summary["aggregates"]["exact_strict_global_hit_rate"]
        chain_aware_global_hit_rate = summary["aggregates"]["chain_aware_global_hit_rate"]
        strict_global_hit_rate = summary["aggregates"]["strict_global_hit_rate"]
        for row in valid_rows:
            raw_run_mode = row.get("raw_run_mode")
            if raw_run_mode in summary["aggregates"]["run_modes"]["raw_trace"]:
                summary["aggregates"]["run_modes"]["raw_trace"][raw_run_mode] += 1

            graph_run_mode = row.get("graph_run_mode")
            if graph_run_mode in summary["aggregates"]["run_modes"]["trace_graph"]:
                summary["aggregates"]["run_modes"]["trace_graph"][graph_run_mode] += 1

            raw_hit_metrics = row.get("raw_hit_metrics")
            if isinstance(raw_hit_metrics, dict):
                accumulate_hit_summary(global_hit_rate["raw_trace"], raw_hit_metrics)

            graph_hit_metrics = row.get("graph_hit_metrics")
            if isinstance(graph_hit_metrics, dict):
                accumulate_hit_summary(global_hit_rate["trace_graph"], graph_hit_metrics)

            raw_exact_strict_hit_metrics = row.get("raw_exact_strict_hit_metrics")
            if isinstance(raw_exact_strict_hit_metrics, dict):
                accumulate_hit_summary(
                    exact_strict_global_hit_rate["raw_trace"],
                    raw_exact_strict_hit_metrics,
                )

            graph_exact_strict_hit_metrics = row.get("graph_exact_strict_hit_metrics")
            if isinstance(graph_exact_strict_hit_metrics, dict):
                accumulate_hit_summary(
                    exact_strict_global_hit_rate["trace_graph"],
                    graph_exact_strict_hit_metrics,
                )

            raw_chain_aware_hit_metrics = row.get("raw_chain_aware_hit_metrics")
            if isinstance(raw_chain_aware_hit_metrics, dict):
                accumulate_hit_summary(chain_aware_global_hit_rate["raw_trace"], raw_chain_aware_hit_metrics)

            graph_chain_aware_hit_metrics = row.get("graph_chain_aware_hit_metrics")
            if isinstance(graph_chain_aware_hit_metrics, dict):
                accumulate_hit_summary(
                    chain_aware_global_hit_rate["trace_graph"],
                    graph_chain_aware_hit_metrics,
                )

            raw_strict_hit_metrics = row.get("raw_strict_hit_metrics")
            if isinstance(raw_strict_hit_metrics, dict):
                accumulate_hit_summary(strict_global_hit_rate["raw_trace"], raw_strict_hit_metrics)

            graph_strict_hit_metrics = row.get("graph_strict_hit_metrics")
            if isinstance(graph_strict_hit_metrics, dict):
                accumulate_hit_summary(strict_global_hit_rate["trace_graph"], graph_strict_hit_metrics)

        finalize_source_hit_summary(global_hit_rate["raw_trace"])
        finalize_source_hit_summary(global_hit_rate["trace_graph"])
        finalize_source_hit_summary(exact_strict_global_hit_rate["raw_trace"])
        finalize_source_hit_summary(exact_strict_global_hit_rate["trace_graph"])
        finalize_source_hit_summary(chain_aware_global_hit_rate["raw_trace"])
        finalize_source_hit_summary(chain_aware_global_hit_rate["trace_graph"])
        finalize_source_hit_summary(strict_global_hit_rate["raw_trace"])
        finalize_source_hit_summary(strict_global_hit_rate["trace_graph"])

        location_audit_bundle = collect_location_audits(raw_cmp_dir, graph_cmp_dir, valid_rows)
        summary["aggregates"]["location_relation_summary"] = location_audit_bundle["summary"]
        summary["aggregates"]["location_audit_files"] = write_location_audit_files(
            out_dir,
            location_audit_bundle,
        )

    dump_json(out_dir / "alignment_summary.json", summary)
    print("\n完成：")
    print(f"- processed: {summary['processed']}")
    print(f"- failed: {summary['failed']}")
    print(f"- summary: {out_dir / 'alignment_summary.json'}")


def build_cli():
    p = argparse.ArgumentParser(description="Compare DeepSeek judge outputs with human annotations using DeepSeek API")
    p.add_argument("--merged-dir", default="output_graphs/deepseek_judge/merged", help="Directory from deepseek_judge_trace merged outputs")
    p.add_argument("--annotations-dir", default="trail_data/processed_annotations_gaia", help="Directory of human annotations")
    p.add_argument("--out-dir", default="output_graphs/deepseek_alignment", help="Output directory for alignment results")
    p.add_argument("--trace-id", default=None, help="Evaluate one trace_id only")
    p.add_argument("--max-samples", type=int, default=0, help="Max samples to evaluate, 0 means all")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Alignment model name (OpenAI-compatible)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Alignment API base URL")
    p.add_argument("--thinking-mode", choices=["disabled", "enabled"], default="disabled", help="DeepSeek V4 thinking mode")
    p.add_argument("--api-key", default=None, help="Alignment API key (or use DEEPSEEK_API_KEY / OPENAI_API_KEY env)")
    p.add_argument("--max-workers", type=int, default=4, help="Maximum number of concurrent trace comparisons")
    p.add_argument(
        "--compare-max-tokens",
        type=int,
        default=DEFAULT_COMPARE_MAX_TOKENS,
        help="Max completion tokens for each compare call; 0 means do not send an explicit max_tokens cap",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing raw_vs_human/graph_vs_human JSON files and only call the API for missing files",
    )
    return p


if __name__ == "__main__":
    args = build_cli().parse_args()
    run(args)
