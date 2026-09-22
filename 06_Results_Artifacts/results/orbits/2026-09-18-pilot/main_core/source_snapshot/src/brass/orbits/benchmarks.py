"""Transparent answer extraction. Program execution lives in a separate container."""

from __future__ import annotations

import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation


def answer_span(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    explicit = re.findall(r"(?:####|(?:final\s+)?answer\s*(?:is|:))\s*(.+)", text, re.I)
    if explicit:
        return explicit[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_qa(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_score(prediction, answers):
    def score(answer):
        p, a = normalize_qa(prediction).split(), normalize_qa(answer).split()
        common = sum((Counter(p) & Counter(a)).values())
        return 2 * common / (len(p) + len(a)) if p or a else 1.0

    return {
        "exact_match": float(any(normalize_qa(prediction) == normalize_qa(a) for a in answers)),
        "f1": max(score(a) for a in answers),
    }


def score_basic(evaluation, text):
    kind = evaluation["kind"]
    span = answer_span(text)
    result = {"kind": kind, "answer_span": span, "correct": None}
    if kind == "numeric":
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", span)
        if numbers:
            try:
                result["correct"] = Decimal(numbers[-1].replace(",", "")) == Decimal(
                    evaluation["answer"].replace(",", "")
                )
            except InvalidOperation:
                pass
        result["extractor"] = "last_number_in_final_span"
    elif kind == "choice":
        labels = evaluation["choices"]["label"]
        matches = re.findall(r"(?<!\w)[(\s*]*([A-E1-5])[).:\s*]*(?!\w)", span)
        matches = [m for m in matches if m in labels]
        if matches:
            result["correct"] = matches[-1] == evaluation["answer"]
        result["extractor"] = "last_explicit_option_in_final_span"
    elif kind == "bbh":
        target = evaluation["answer"]
        if target.startswith("("):
            matches = re.findall(r"\(([A-Z])\)", span)
            result["correct"] = bool(matches) and f"({matches[-1]})" == target
        elif target in ("True", "False"):
            matches = re.findall(r"\b(True|False)\b", span, re.I)
            result["correct"] = bool(matches) and matches[-1].lower() == target.lower()
        elif re.fullmatch(r"\d+", target):
            numbers = re.findall(r"\b\d+\b", span)
            result["correct"] = bool(numbers) and numbers[-1] == target
        else:
            # Word sorting must retain articles: 'a', 'an', and 'the' are task data.
            def normalize_words(s):
                return " ".join(
                    s.lower().translate(str.maketrans("", "", string.punctuation)).split()
                )

            result["correct"] = normalize_words(span) == normalize_words(target)
        result["extractor"] = "task_specific_final_span"
    elif kind == "qa":
        result.update(qa_score(span, evaluation["answers"]))
        result["correct"] = bool(result["exact_match"])
        result["extractor"] = "final_span_SQuAD_normalization"
    return result


def extract_python(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S | re.I)
    return "\n\n".join(blocks) if blocks else text
