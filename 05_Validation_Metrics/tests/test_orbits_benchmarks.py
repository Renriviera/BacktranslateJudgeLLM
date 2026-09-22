from brass.orbits.benchmarks import extract_python, qa_score, score_basic


def test_math_uses_final_answer_not_intermediate_number():
    score = score_basic(
        {"kind": "numeric", "answer": "10"}, "Four bills give 80 dollars.\nFinal answer: $10."
    )
    assert score["correct"]


def test_wrong_choice_not_credited_for_mentioning_right_one():
    score = score_basic(
        {"kind": "choice", "answer": "B", "choices": {"label": ["A", "B", "C"]}},
        "B is a tempting option.\nAnswer: C.",
    )
    assert score["correct"] is False


def test_qa_official_normalization():
    assert qa_score("The Eiffel Tower.", ["Eiffel Tower"]) == {"exact_match": 1.0, "f1": 1.0}


def test_python_extraction_does_not_execute_code():
    assert (
        extract_python("Here:\n```python\ndef f():\n    return 1\n```")
        == "def f():\n    return 1\n"
    )


def test_unscored_rubric_remains_unknown():
    assert score_basic({"kind": "rubric"}, "A creative answer")["correct"] is None


def test_word_sorting_does_not_drop_articles():
    assert score_basic({"kind": "bbh", "answer": "a an the"}, "a an the")["correct"]
    assert not score_basic({"kind": "bbh", "answer": "a an the"}, "")["correct"]
