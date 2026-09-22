from brass.orbits.analysis import validity_report


def test_missing_inverse_samples_do_not_pass_gates():
    manifest = [{"id": "x", "cohort": "benign"}]
    events = [
        {
            "item_id": "x",
            "direction": "forward",
            "finish_reason": "length",
            "status": "generation_truncated",
            "output_tokens": 1024,
            "input_tokens": 50,
        }
    ]
    report = validity_report(manifest, events)
    assert not report["all_numeric_gates_pass"]
    assert report["cohorts"]["benign"]["inverse_usable_fraction"] is None
    assert report["cohorts"]["benign"]["truncation_fraction"] == 1


def test_inverse_refusal_separate_from_forward_success():
    manifest = [{"id": "x", "cohort": "attack"}]
    events = [
        {
            "item_id": "x",
            "direction": "inverse",
            "finish_reason": "stop",
            "status": "inverse_refusal_heuristic",
            "output_tokens": 30,
            "input_tokens": 50,
        }
    ]
    report = validity_report(manifest, events)["cohorts"]["attack"]
    assert report["inverse_refusal_heuristic_fraction"] == 1
    assert report["inverse_usable_fraction"] == 0
    assert not report["numeric_gates_pass"]
