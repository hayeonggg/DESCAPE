"""Benchmark definitions: data loading, prompt templates, and decoding lengths."""

import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

CONCISE_SYSTEM = "You are a helpful assistant. Answer the question concisely in only one sentence."


# =========================================================================== #
# EM / token-level F1
# =========================================================================== #

def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


def exact_match(prediction: str, answers: List[str]) -> float:
    pred_norm = normalize_answer(prediction)
    return float(any(pred_norm == normalize_answer(a) for a in answers))


def token_f1(prediction: str, answers: List[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for ans in answers:
        gold_tokens = normalize_answer(ans).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred_tokens)
        recall = n_common / len(gold_tokens)
        best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return best_f1


# =========================================================================== #
# Data loaders: each returns a list of records with a 'question' (or 'entity') field
# =========================================================================== #

def load_truthfulqa(args) -> List[Dict]:
    from datasets import load_dataset
    dataset = load_dataset("truthful_qa", "generation")['validation']
    if args.num_sample and args.num_sample < len(dataset):
        dataset = dataset.select(range(args.num_sample))
    return [{
        'question': s['question'],
        'correct_answers': s.get('correct_answers', []),
        'incorrect_answers': s.get('incorrect_answers', []),
    } for s in dataset]


def load_freshqa(args) -> List[Dict]:
    import pandas as pd
    if not args.data_path:
        raise ValueError("FreshQA requires --data_path (path to the FreshQA csv)")
    df = pd.read_csv(args.data_path)
    if args.num_sample and args.num_sample < len(df):
        df = df.iloc[:args.num_sample]
    return [{
        'question': str(row['question']),
        'correct_answers': [
            str(row[f'answer_{i}'])
            for i in range(10)
            if f'answer_{i}' in row and pd.notna(row[f'answer_{i}'])
        ],
    } for _, row in df.iterrows()]


def load_factscore(args) -> List[Dict]:
    if not args.data_path:
        raise ValueError(
            "FActScore requires --data_path (prompt_entities.txt from the FActScore data)"
        )
    with open(args.data_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    random.seed(args.seed)
    random.shuffle(lines)
    return [{'entity': e} for e in lines[:args.num_sample]]


def load_nq(args) -> List[Dict]:
    from datasets import load_dataset
    dataset = load_dataset("nq_open", split="validation")
    if args.num_sample and args.num_sample < len(dataset):
        dataset = dataset.select(range(args.num_sample))
    return [{'question': s['question'], 'answers': s['answer']} for s in dataset]


def load_triviaqa(args) -> List[Dict]:
    from datasets import load_dataset
    dataset = load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='validation')
    if args.num_sample is not None and args.num_sample < len(dataset):
        random.seed(args.seed)
        indices = random.sample(range(len(dataset)), args.num_sample)
        dataset = dataset.select(indices)
    return [{
        'question': str(s['question']),
        'correct_answers': s['answer']['aliases'],
    } for s in dataset]


# =========================================================================== #
# Tasks
# =========================================================================== #

@dataclass
class Task:
    system_prompt: str
    user_prompt: Callable[[Dict], str]
    load: Callable
    max_length: int
    min_length: int
    num_sample: Optional[int] = None        # default number of samples (None = all)
    answer_key: Optional[str] = None        # gold answers for EM / F1 (short-answer QA)
    # Whether prompt tokens are also subject to the repetition penalty.
    penalize_prompt_tokens: bool = True


TASKS: Dict[str, Task] = {
    'truthfulqa': Task(
        system_prompt=CONCISE_SYSTEM,
        user_prompt=lambda r: f"{r['question']}\nProvide a truthful, direct answer in one sentence.",
        load=load_truthfulqa,
        max_length=256, min_length=12,
    ),
    'freshqa': Task(
        system_prompt=CONCISE_SYSTEM,
        user_prompt=lambda r: (
            f"Answer the following question briefly and factually.\n"
            f"If the question contains a false premise, explicitly point it out and correct it.\n"
            f"Do not add unnecessary details.\n\n"
            f"Q: {r['question']}\nA:"
        ),
        load=load_freshqa,
        max_length=256, min_length=20,
    ),
    'factscore': Task(
        system_prompt=(
            "You are a knowledgeable assistant. Write a concise, factually accurate "
            "biography in 3-5 sentences. Include only verified facts."
        ),
        user_prompt=lambda r: f"Question: Tell me a bio of {r['entity']}.",
        load=load_factscore,
        max_length=256, min_length=20, num_sample=500,
    ),
    'nq': Task(
        system_prompt=CONCISE_SYSTEM,
        user_prompt=lambda r: f"{r['question']}\nAnswer with a short, factual phrase or name.",
        load=load_nq,
        max_length=128, min_length=4, answer_key='answers',
    ),
    'triviaqa': Task(
        system_prompt=(
            "You are a helpful assistant. Answer the question directly and concisely "
            "in a few words or a short phrase."
        ),
        user_prompt=lambda r: f"Question: {r['question']}\nAnswer:",
        load=load_triviaqa,
        max_length=256, min_length=1, num_sample=3000, answer_key='correct_answers',
        penalize_prompt_tokens=False,
    ),
}
