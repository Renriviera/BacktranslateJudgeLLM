from brass.orbits.grouping import grouped_main


def test_near_duplicate_of_pilot_is_excluded_with_its_relatives():
    text = " ".join(f"word{i}" for i in range(100))
    pilot = [{"id": "pilot", "group_id": "old", "prompt": text}]
    main = [
        {"id": "new", "group_id": "newgroup", "prompt": text + " extra"},
        {"id": "relative", "group_id": "newgroup", "prompt": "framed sibling"},
        {"id": "independent", "group_id": "independent", "prompt": "A wholly different task"},
    ]
    kept, audit = grouped_main(main, pilot)
    assert [r["id"] for r in kept] == ["independent"]
    assert set(audit["removed_main_ids"]) == {"new", "relative"}


def test_near_groups_keep_source_ids_for_archive_joins():
    main = [
        {"id": "a", "group_id": "source:a", "prompt": "Task text"},
        {"id": "b", "group_id": "source:b", "prompt": "TASK TEXT!"},
    ]
    kept, _ = grouped_main(main, [])
    assert {r["group_id"] for r in kept} == {"source:a", "source:b"}
    assert len({r["analysis_group_id"] for r in kept}) == 1
