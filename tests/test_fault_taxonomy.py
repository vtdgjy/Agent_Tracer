from fault_taxonomy import detect_fault_signatures


def test_detects_execution_contract_and_protocol_signatures():
    trace = {
        "spans": [
            {
                "span_id": "tool-1",
                "span_name": "browser.search",
                "status_code": "ERROR",
                "span_attributes": {"tool.name": "browser.search", "input.value": "{}"},
            },
            {"span_id": "format-1", "span_name": "FormattingError: missing required tag <end_plan>"},
        ]
    }
    semantic = {
        "semantic_violations": [
            {"violation_type": "tool_execution_failure", "source_span_id": "tool-1"},
            {"violation_type": "missing_required_argument", "source_span_id": "tool-1"},
        ]
    }
    findings = detect_fault_signatures(trace, semantic)
    ids = {item["fault_id"] for item in findings}
    assert {"F4", "F9", "F8"}.issubset(ids)
    assert all(item["evidence_span_ids"] for item in findings)


def test_marks_repeated_failed_attempts_and_error_after_finalization_as_review_candidates():
    spans = []
    for index in range(3):
        spans.append({
            "span_id": f"tool-{index}",
            "span_name": "browser.search",
            "status_code": "ERROR",
            "span_attributes": {"tool.name": "browser.search", "input.value": '{"query":"same"}'},
        })
    spans.extend([
        {"span_id": "err-1", "span_name": "ToolError", "parent_span_id": "tool-2"},
        {"span_id": "err-2", "span_name": "TypeError", "parent_span_id": "tool-2"},
        {"span_id": "final", "span_name": "final_answer"},
    ])
    findings = detect_fault_signatures({"spans": spans}, {
        "semantic_violations": [
            {"violation_type": "tool_execution_failure", "source_span_id": f"tool-{i}"}
            for i in range(3)
        ]
    })
    by_id = {item["fault_id"]: item for item in findings}
    assert {"F3", "F10", "F11", "F12"}.issubset(by_id)
    assert by_id["F11"]["requires_review"] is True


def test_does_not_infer_unobservable_semantic_causes():
    findings = detect_fault_signatures({"spans": [
        {"span_id": "plan", "span_name": "planner"},
        {"span_id": "final", "span_name": "final_answer"},
    ]})
    ids = {item["fault_id"] for item in findings}
    assert "F1" not in ids
    assert "F2" not in ids
    assert "F7" not in ids
