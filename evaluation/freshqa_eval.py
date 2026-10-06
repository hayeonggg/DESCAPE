#!/usr/bin/env python3

"""
FreshQA LLM-as-judge evaluator (FreshEval-compatible)

This script evaluates FreshQA model outputs using the official FreshEval prompts
from the FreshQA repository notebooks:
  - fresheval_strict.ipynb
  - fresheval_relaxed.ipynb

It reproduces the official prompt construction pattern:
  prefix + few-shot demonstrations + target question block

Usage example:
  git clone https://github.com/freshllms/freshqa
  export OPENAI_API_KEY=...
  python evaluation/freshqa_eval.py \
    --input outputs/freshqa_llama.json \
    --output outputs/freshqa_llama_strict.json \
    --freshqa-dir freshqa \
    --mode strict
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from tqdm import tqdm

try:
    from openai import OpenAI as OpenAIClient

    _OPENAI_V1 = True
except Exception:
    import openai as openai_legacy

    OpenAIClient = None
    _OPENAI_V1 = False


OFFICIAL_NOTEBOOKS = {
    "strict": "fresheval_strict.ipynb",
    "relaxed": "fresheval_relaxed.ipynb",
}


@dataclass
class FreshEvalArtifacts:
    mode: str
    notebook_path: str
    current_date: str
    prefix: str
    demo_questions: List[str]
    demo_evaluations: List[str]
    evaluation_template: str


def _current_date_pst() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%B %d, %Y")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_official_fresheval_artifacts(mode: str, freshqa_dir: Path) -> FreshEvalArtifacts:
    if mode not in OFFICIAL_NOTEBOOKS:
        raise ValueError(f"Unsupported mode: {mode}")

    notebook_path = freshqa_dir / OFFICIAL_NOTEBOOKS[mode]
    if not notebook_path.exists():
        raise FileNotFoundError(f"Official FreshEval notebook not found: {notebook_path}")

    notebook = json.loads(_read_text(notebook_path))
    cells = notebook.get("cells", [])

    prompt_cell_source = None
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source_text = "\n".join(source)
        if "prefix = (" in source_text and "demo_examples = [" in source_text and "evaluation_template = (" in source_text:
            prompt_cell_source = source_text
            break

    if prompt_cell_source is None:
        raise RuntimeError(f"Could not locate official FreshEval prompt cell in {notebook_path}")

    local_vars: Dict[str, Any] = {"current_date": _current_date_pst()}
    exec(prompt_cell_source, {}, local_vars)

    required = ["prefix", "demo_questions", "demo_evaluations", "evaluation_template"]
    missing = [k for k in required if k not in local_vars]
    if missing:
        raise RuntimeError(f"Failed to load FreshEval prompt artifacts. Missing keys: {missing}")

    return FreshEvalArtifacts(
        mode=mode,
        notebook_path=str(notebook_path),
        current_date=local_vars["current_date"],
        prefix=local_vars["prefix"],
        demo_questions=local_vars["demo_questions"],
        demo_evaluations=local_vars["demo_evaluations"],
        evaluation_template=local_vars["evaluation_template"],
    )


def build_fresheval_prompt(
    artifacts: FreshEvalArtifacts,
    question: str,
    response: str,
    correct_answers: List[str],
) -> str:
    formatted_eval = artifacts.evaluation_template.format(
        correct_answers=" | ".join(correct_answers),
        response=response,
    )

    demo_prompts = []
    for question_demo, evaluation_demo in zip(artifacts.demo_questions, artifacts.demo_evaluations):
        demo_prompts.append(f"\n\n\nquestion: {question_demo}{evaluation_demo}")

    fresheval_demo = "".join(demo_prompts).strip()
    fresheval_question = f"\n\n\nquestion: {question}{formatted_eval}"
    return artifacts.prefix + "\n\n\n" + fresheval_demo + fresheval_question


def extract_rating(response: str) -> Tuple[bool, Optional[str]]:
    for line in response.split("\n"):
        match = re.search(r"evaluation:\s*(correct|incorrect)\b", line.strip(), flags=re.IGNORECASE)
        if match:
            label = match.group(1).lower()
            return True, "TRUE" if label == "correct" else "FALSE"

    if "Thus, the response is credited." in response:
        return True, "TRUE"
    if "Thus, the response is not credited." in response:
        return True, "FALSE"
    return False, None


def load_input_records(input_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(_read_text(input_path))
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError("Input JSON must be either a list or an object with a 'results' list")


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    question = (
        record.get("question")
        or record.get("query")
        or record.get("prompt")
        or ""
    )
    response = (
        record.get("prediction")
        or record.get("output")
        or record.get("response")
        or record.get("model_response")
        or record.get("answer")
        or ""
    )

    correct_answers = _to_list(record.get("correct_answers"))

    if not correct_answers:
        for i in range(10):
            key = f"answer_{i}"
            if key in record and str(record[key]).strip():
                correct_answers.append(str(record[key]).strip())

    return {
        "idx": record.get("idx"),
        "question": str(question).strip(),
        "response": str(response).strip(),
        "correct_answers": correct_answers,
    }


def resolve_api_key() -> str:
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    raise ValueError("OPENAI_API_KEY environment variable is not set.")


def call_judge(
    client: Any,
    judge_model: str,
    prompt: str,
    current_date: str,
    max_tokens: int = 256,
    max_retries: int = 5,
) -> str:
    wait_sec = 1.0
    last_error = None

    for _ in range(max_retries):
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Respond as concisely as"
                        f" possible. Knowledge cutoff: {current_date}."
                    ),
                },
                {"role": "user", "content": "What's today's date?"},
                {
                    "role": "assistant",
                    "content": f"Today is {current_date} in Pacific Standard Time.",
                },
                {"role": "user", "content": prompt},
            ]

            if _OPENAI_V1:
                completion = client.chat.completions.create(
                    model=judge_model,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                return (completion.choices[0].message.content or "").strip()

            completion = client.ChatCompletion.create(
                model=judge_model,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=messages,
            )
            return (completion["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:
            last_error = exc
            time.sleep(wait_sec)
            wait_sec = min(wait_sec * 2, 30.0)

    raise RuntimeError(f"Judge API call failed after retries: {last_error}")


def evaluate_freshqa(
    input_records: List[Dict[str, Any]],
    artifacts: FreshEvalArtifacts,
    client: Optional[Any],
    judge_model: str,
    parse_retries: int,
    max_samples: Optional[int],
    dry_run: bool,
) -> Dict[str, Any]:
    normalized = [normalize_record(record) for record in input_records]
    if max_samples is not None:
        normalized = normalized[: max(0, max_samples)]

    evaluated_rows = []
    true_count = 0
    false_count = 0
    invalid_count = 0

    iterator = tqdm(normalized, desc=f"FreshEval-{artifacts.mode}")
    for idx, row in enumerate(iterator):
        question = row["question"]
        response = row["response"]
        correct_answers = row["correct_answers"]

        prompt = build_fresheval_prompt(
            artifacts=artifacts,
            question=question,
            response=response,
            correct_answers=correct_answers,
        )

        if dry_run:
            rating = None
            explanation = "[DRY_RUN]"
            is_valid = False
        else:
            if client is None:
                raise RuntimeError("OpenAI client is required when dry_run is False")

            is_valid = False
            rating = None
            explanation = ""
            for _ in range(max(1, parse_retries)):
                explanation = call_judge(
                    client=client,
                    judge_model=judge_model,
                    prompt=prompt,
                    current_date=artifacts.current_date,
                )
                is_valid, rating = extract_rating(explanation)
                if is_valid:
                    break

        if rating == "TRUE":
            true_count += 1
        elif rating == "FALSE":
            false_count += 1
        else:
            invalid_count += 1

        evaluated_rows.append(
            {
                "idx": row["idx"] if row["idx"] is not None else idx,
                "question": question,
                "response": response,
                "correct_answers": correct_answers,
                "rating": rating,
                "is_valid": is_valid,
                "judge_output": explanation,
                "judge_prompt": prompt if dry_run else None,
            }
        )

    total = len(evaluated_rows)
    valid = true_count + false_count
    accuracy_on_total = (true_count / total) if total else 0.0
    accuracy_on_valid = (true_count / valid) if valid else 0.0

    return {
        "summary": {
            "mode": artifacts.mode,
            "total": total,
            "valid_ratings": valid,
            "true_count": true_count,
            "false_count": false_count,
            "invalid_count": invalid_count,
            "accuracy_on_total": round(accuracy_on_total, 6),
            "accuracy_on_valid": round(accuracy_on_valid, 6),
        },
        "results": evaluated_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FreshQA LLM-as-judge evaluator (official FreshEval prompt)")
    parser.add_argument("--input", required=True, help="Path to FreshQA predictions JSON")
    parser.add_argument("--output", required=True, help="Path to save evaluation JSON")
    parser.add_argument("--freshqa-dir", default="freshqa",
                        help="Path to a clone of https://github.com/freshllms/freshqa")
    parser.add_argument("--mode", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--parse-retries", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Build prompts only without API calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    artifacts = load_official_fresheval_artifacts(args.mode, Path(args.freshqa_dir))
    records = load_input_records(input_path)

    client = None
    if not args.dry_run:
        api_key = resolve_api_key()
        if _OPENAI_V1:
            client = OpenAIClient(api_key=api_key)
        else:
            openai_legacy.api_key = api_key
            client = openai_legacy

    evaluated = evaluate_freshqa(
        input_records=records,
        artifacts=artifacts,
        client=client,
        judge_model=args.judge_model,
        parse_retries=args.parse_retries,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
    )

    payload = {
        "config": {
            "input": str(input_path),
            "mode": args.mode,
            "judge_model": args.judge_model,
            "parse_retries": args.parse_retries,
            "max_samples": args.max_samples,
            "dry_run": args.dry_run,
            "official_prompt_notebook": artifacts.notebook_path,
            "official_prompt_current_date": artifacts.current_date,
            "num_demo_examples": len(artifacts.demo_questions),
        },
        **evaluated,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = payload["summary"]
    print(
        "FreshEval done | "
        f"mode={summary['mode']} "
        f"total={summary['total']} "
        f"TRUE={summary['true_count']} "
        f"FALSE={summary['false_count']} "
        f"invalid={summary['invalid_count']} "
        f"acc_total={summary['accuracy_on_total']:.4f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
