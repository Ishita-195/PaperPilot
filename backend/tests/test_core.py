"""Fast, dependency-light tests that run in CI without API keys or models."""
from backend import evaluation
from backend.kb import DOCUMENTS, categories


def test_kb_integrity():
    ids = [d["id"] for d in DOCUMENTS]
    assert len(ids) == len(set(ids)), "duplicate document ids"
    for d in DOCUMENTS:
        assert {"id", "topic", "category", "text"} <= d.keys()
        assert d["text"].strip()


def test_categories_nonempty():
    assert len(categories()) >= 5


def test_parse_score_clamps_and_extracts():
    assert evaluation._parse_score("0.85") == 0.85
    assert evaluation._parse_score("Score: 0.9 out of 1") == 0.9
    assert evaluation._parse_score("1.7") == 1.0
    assert evaluation._parse_score("-0.3") == 0.0
    assert evaluation._parse_score("garbage", default=0.5) == 0.5


def test_eval_dataset_shape():
    assert len(evaluation.EVAL_DATA) >= 5
    for item in evaluation.EVAL_DATA:
        assert item["question"] and item["ground_truth"]
