"""Identify the factual-salient layer span with sliding-window MLP ablation.

For each window w of contiguous layers, the MLP outputs in w are zeroed out and the
Factual Attribution Score is computed on the answer tokens a:

    Delta(w) = log P_S(a) - log P_W(a)

where P_S is the original model (strong view) and P_W the ablated model (weak view).
The window with the highest mean Delta(w) is the factual-salient layer span.
Only TriviaQA samples that the model answers correctly under greedy decoding are used.

Usage:
  python find_layer_span.py --model meta-llama/Llama-3.1-8B-Instruct \
      --output_path outputs/layer_span_llama.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class LayerSpanAnalyzer:

    def __init__(self, model_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        print(f"Loading model from {model_path}...")
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded with {self.num_layers} layers")

        self.hooks = []
        self._correct_cache: Dict[str, bool] = {}

    # ── Ablation hooks (MLP zeroing) ───────────────────────────────────

    def _register_ablation_hooks(self, layer_range: Tuple[int, int]):
        self._remove_hooks()
        start_layer, end_layer = layer_range

        def ablation_hook(_module, _input, output):
            return torch.zeros_like(output)

        for layer_idx in range(start_layer, end_layer + 1):
            if layer_idx < self.num_layers:
                mlp = self.model.model.layers[layer_idx].mlp
                self.hooks.append(mlp.register_forward_hook(ablation_hook))

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    # ── Sample selection ───────────────────────────────────────────────

    def _is_answered_correctly(self, question: str, answer: str,
                               answer_aliases: List[str]) -> bool:
        """Greedy-decode a short answer and check for a partial match with the gold answer."""
        if question in self._correct_cache:
            return self._correct_cache[question]

        messages = [
            {"role": "system",
             "content": "You are a helpful assistant. Answer the question directly and concisely."},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            self._remove_hooks()
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip().lower()

        candidates = [answer.lower()] + [alias.lower() for alias in answer_aliases]

        # bidirectional substring inclusion
        correct = any(c in generated_text or generated_text in c for c in candidates)

        # word-level overlap within the first three words (words longer than 2 characters)
        if not correct:
            generated_words = generated_text.split()[:3]
            for c in candidates:
                candidate_words = c.split()[:3]
                if any(gw in candidate_words for gw in generated_words if len(gw) > 2):
                    correct = True
                    break

        self._correct_cache[question] = correct
        return correct

    # ── Factual Attribution Score ──────────────────────────────────────

    def compute_delta(self, question: str, answer: str, answer_aliases: List[str],
                      layer_range: Tuple[int, int]) -> Optional[float]:
        """Mean Delta(w) over the answer tokens, or None if the sample is not answered correctly."""
        if not self._is_answered_correctly(question, answer, answer_aliases):
            return None

        messages = [
            {"role": "user",
             "content": f"Answer the question directly and concisely.\n\n{question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if len(answer_ids) == 0:
            return None

        with torch.no_grad():
            # Strong view: original model
            self._remove_hooks()
            logits_strong = self.model(input_ids).logits[:, -1, :]

            # Weak view: MLPs in the window zeroed out
            self._register_ablation_hooks(layer_range)
            logits_weak = self.model(input_ids).logits[:, -1, :]
            self._remove_hooks()

        prob_strong = F.softmax(logits_strong, dim=-1)
        prob_weak = F.softmax(logits_weak, dim=-1)

        deltas = []
        for token_id in answer_ids:
            p_strong = prob_strong[0, token_id].item()
            p_weak = prob_weak[0, token_id].item()
            deltas.append(np.log(p_strong + 1e-10) - np.log(p_weak + 1e-10))
        return float(np.mean(deltas))

    def analyze(self, dataset: List[Dict], layer_ranges: List[Tuple[int, int]]) -> Dict:
        results = {'layer_ranges': [], 'avg_delta': [], 'std_delta': [], 'correct_count': []}

        for layer_range in layer_ranges:
            range_str = f"{layer_range[0]}-{layer_range[1]}"
            scores = []
            total = 0

            pbar = tqdm(dataset, desc=f"Window [{range_str}]")
            for sample in pbar:
                question = sample['question']
                answer = sample['answer'].get('value', '') or sample['answer'].get('normalized_value', '')
                aliases = sample['answer'].get('aliases', [])
                if not answer:
                    continue
                total += 1

                delta = self.compute_delta(question, answer, aliases, layer_range)
                if delta is not None:
                    scores.append(delta)
                if scores:
                    pbar.set_postfix({'Delta': f'{np.mean(scores):.3f}', 'N': len(scores)})

            results['layer_ranges'].append(range_str)
            results['avg_delta'].append(float(np.mean(scores)) if scores else 0.0)
            results['std_delta'].append(float(np.std(scores)) if scores else 0.0)
            results['correct_count'].append(len(scores))
            print(f"  [{range_str}]  Delta(w) = {results['avg_delta'][-1]:.4f}  "
                  f"(correct: {len(scores)}/{total})")

        return results


def parse_layer_ranges(text: str) -> List[Tuple[int, int]]:
    ranges = []
    for part in text.split(','):
        start, end = part.strip().split('-')
        ranges.append((int(start), int(end)))
    return ranges


def main():
    parser = argparse.ArgumentParser(description="Sliding-window MLP ablation")
    parser.add_argument('--model', type=str, required=True,
                        help='Hugging Face model id or local path')
    parser.add_argument('--output_path', type=str, default="outputs/layer_span.json")
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of TriviaQA samples')
    parser.add_argument('--layer_ranges', type=str,
                        default="8-14,12-18,16-22,20-26,24-30",
                        help='Candidate windows (inclusive), window size 7 and stride 4 by default')
    args = parser.parse_args()

    analyzer = LayerSpanAnalyzer(args.model)

    print("\nLoading TriviaQA dataset...")
    dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")
    samples = [{'question': s['question'], 'answer': s['answer']}
               for s in dataset.select(range(min(args.num_samples, len(dataset))))]

    layer_ranges = parse_layer_ranges(args.layer_ranges)
    results = analyzer.analyze(samples, layer_ranges)

    best = int(np.argmax(results['avg_delta']))
    results['factual_salient_span'] = results['layer_ranges'][best]
    results['config'] = {
        'model': args.model,
        'num_samples': args.num_samples,
        'num_layers': analyzer.num_layers,
    }

    print("\n" + "=" * 60)
    print(f"{'Window':^12} | {'Delta(w)':^12} | {'N correct':^10}")
    print("-" * 60)
    for i, (r, d, n) in enumerate(zip(results['layer_ranges'], results['avg_delta'],
                                      results['correct_count'])):
        marker = "*" if i == best else " "
        print(f"{marker} {r:^10} | {d:^12.4f} | {n:^10}")
    print("-" * 60)
    print(f"Factual-salient layer span: [{results['factual_salient_span']}]")
    print("=" * 60)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or '.', exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {args.output_path}")


if __name__ == "__main__":
    main()
