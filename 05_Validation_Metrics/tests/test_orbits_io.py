import pytest

from brass.orbits.io import append_jsonl, read_jsonl


def test_unicode_line_separators_inside_json_are_not_record_boundaries(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = [{"text": "before\u2028after\u2029next\x85end"}, {"text": "two\nlines"}]
    append_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_partial_record_still_fails_loudly(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"text":"unfinished')
    with pytest.raises(ValueError):
        read_jsonl(path)
