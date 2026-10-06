"""Short-answer QA evaluation (NQ, TriviaQA): EM, token-level F1, and SoftEM.

SoftEM scores a prediction as correct if any gold answer appears as a normalized
substring of the normalized prediction, so that correct answers embedded in longer
responses are not penalized.

Usage:
  python evaluation/short_answer_eval.py --input outputs/triviaqa_llama.json
"""

import argparse
import json
import re
import string
from typing import List


def normalize_answer(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("_", " ").lower()
    exclude = set(string.punctuation + "‘’´`")
    text = "".join(ch if ch not in exclude else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).strip()


def get_gold_answers(record: dict) -> List[str]:
    for key in ("correct_answers", "answers"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value]
    return []


def soft_em(prediction: str, answers: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return 0.0
    for answer in answers:
        gold_norm = normalize_answer(answer)
        if gold_norm and gold_norm in pred_norm:
            return 1.0
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="EM / F1 / SoftEM for short-answer QA")
    parser.add_argument("--input", required=True,
                        help="results JSON written by generate.py (nq or triviaqa)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)["results"]

    em, f1, soft = [], [], []
    for record in records:
        answers = get_gold_answers(record)
        if not answers:
            continue
        em.append(float(record["em"]))
        f1.append(float(record["f1"]))
        soft.append(soft_em(str(record.get("prediction", "")), answers))

    n = max(len(em), 1)
    print(f"Samples : {len(em)}")
    print(f"EM      : {sum(em) / n * 100:.2f}")
    print(f"F1      : {sum(f1) / n * 100:.2f}")
    print(f"SoftEM  : {sum(soft) / n * 100:.2f}")


if __name__ == "__main__":
    main()
