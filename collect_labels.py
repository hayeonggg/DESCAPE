"""Collect supervision targets for the probe.

Runs greedy autoregressive generation on Databricks Dolly-15k while keeping two KV
caches in parallel: one for the original model (strong view) and one for the model
whose MLPs in the factual-salient layer span are zeroed out (weak view). At every
decoding step this gives the real attribution signal of each top-K candidate token v,

    delta_real(v) = log p(v) - log p_abl(v),

without extra forward passes. The hidden state at the last layer of the span is stored
as the probe input.

Usage:
  python collect_labels.py --model meta-llama/Llama-3.1-8B-Instruct \
      --layer_span 12-18 --num_sample 1000 --output_path labels/labels_llama.pt
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class LayerHiddenCapture:
    """Forward hook that keeps the last-position hidden state of a decoder layer."""

    def __init__(self):
        self.captured: Optional[torch.Tensor] = None

    def hook(self, module, inp, out):
        if isinstance(out, tuple):
            self.captured = out[0][:, -1, :].detach()   # (1, D)
        else:
            self.captured = out[:, -1, :].detach()

    def clear(self):
        self.captured = None


class LabelCollector:
    """
    Parameters
    ----------
    model_path    : base LLM
    factual_start : first layer of the factual-salient span
    factual_end   : last layer of the span (inclusive); the hidden state is captured here
    top_k         : number of candidate tokens recorded per step
    max_length    : maximum number of generated tokens
    tau_entropy   : steps with entropy above this value are marked as triggered (for analysis)
    """

    def __init__(
        self,
        model_path: str,
        factual_start: int,
        factual_end:   int,
        top_k:         int   = 10,
        max_length:    int   = 128,
        tau_entropy:   float = 2.0,
        min_length:    int   = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        print(f"Loading LLM from {model_path} ...")
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size
        self.factual_layers = list(range(factual_start, factual_end + 1))
        self.probe_layer = factual_end

        print(f"Factual layers : {self.factual_layers}")
        print(f"Probe layer    : {self.probe_layer}")

        self._ablation_active = False
        self._register_ablation_hooks()

        self._hs_capture = LayerHiddenCapture()
        self.model.model.layers[self.probe_layer].register_forward_hook(
            self._hs_capture.hook
        )

        self.top_k       = top_k
        self.max_length  = max_length
        self.min_length  = min_length
        self.tau_entropy = tau_entropy

    # ── Ablation hooks (MLP zeroing) ───────────────────────────────────

    def _register_ablation_hooks(self):
        def hook(module, inp, out):
            if self._ablation_active:
                if isinstance(out, tuple):
                    return (torch.zeros_like(out[0]),) + out[1:]
                return torch.zeros_like(out)
        for layer_idx in self.factual_layers:
            self.model.model.layers[layer_idx].mlp.register_forward_hook(hook)

    # ── Forward helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def _forward_normal(
        self, cur_ids: torch.Tensor, past_kv
    ) -> Tuple[torch.Tensor, object, torch.Tensor]:
        """One-token forward of the original model. Returns (logits (1,V), past_kv, h_t (1,D))."""
        self._hs_capture.clear()
        self._ablation_active = False
        out = self.model(cur_ids, past_key_values=past_kv, use_cache=True)
        logits = out.logits[:, -1, :].float()
        h_t    = self._hs_capture.captured.float()
        return logits, out.past_key_values, h_t

    @torch.no_grad()
    def _forward_ablated(
        self, cur_ids: torch.Tensor, past_kv_a
    ) -> Tuple[torch.Tensor, object]:
        """One-token forward with the factual MLPs zeroed. Returns (logits (1,V), past_kv)."""
        self._ablation_active = True
        out = self.model(cur_ids, past_key_values=past_kv_a, use_cache=True)
        self._ablation_active = False
        return out.logits[:, -1, :].float(), out.past_key_values

    # ── Collect one sample ─────────────────────────────────────────────

    def collect_sample(self, prompt: str) -> List[Dict]:
        """Greedy generation for one prompt; returns one record per decoding step."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # ── Dual-KV prefill ──────────────────────────────────────────
        self._hs_capture.clear()
        self._ablation_active = False
        with torch.no_grad():
            out_n = self.model(input_ids, use_cache=True)
        past_kv_n = out_n.past_key_values
        logits_n  = out_n.logits[:, -1, :].float()
        h_t       = self._hs_capture.captured.float()  # (1, D)

        self._ablation_active = True
        with torch.no_grad():
            out_a = self.model(input_ids, use_cache=True)
            self._ablation_active = False
        past_kv_a = out_a.past_key_values
        logits_a  = out_a.logits[:, -1, :].float()

        records: List[Dict] = []
        generated_tokens: List[int] = []

        # ── Decode loop ──────────────────────────────────────────────
        while len(generated_tokens) < self.max_length:
            log_probs_n = F.log_softmax(logits_n[0], dim=-1)    # (V,)
            log_probs_a = F.log_softmax(logits_a[0], dim=-1)    # (V,)

            _, top_ids_tensor = torch.topk(logits_n[0], self.top_k)
            top_ids = top_ids_tensor.tolist()

            delta_real = torch.tensor(
                [log_probs_n[w].item() - log_probs_a[w].item() for w in top_ids],
                dtype=torch.float32,
            )   # (K,)

            top1_prob = float(log_probs_n.max().exp().item())
            entropy   = float(-(log_probs_n.exp() * log_probs_n).sum().item())
            triggered = entropy > self.tau_entropy

            next_token = top_ids[0]

            records.append({
                'h':         h_t[0].half().cpu(),          # (D,) float16
                'top_ids':   top_ids,                       # List[int] K
                'delta':     delta_real.cpu(),              # (K,) float32
                'entropy':   entropy,
                'top1_prob': top1_prob,
                'token_id':  next_token,
                'triggered': triggered,
            })

            generated_tokens.append(next_token)
            cur_ids = torch.tensor([[next_token]], device=self.device)

            if next_token == self.tokenizer.eos_token_id:
                if len(generated_tokens) >= self.min_length:
                    break

            logits_n, past_kv_n, h_t = self._forward_normal(cur_ids, past_kv_n)
            logits_a, past_kv_a      = self._forward_ablated(cur_ids, past_kv_a)

        return records

    # ── Build prompt ───────────────────────────────────────────────────

    def build_prompt(self, question: str, reference_answer: str = "") -> str:
        """The length instruction follows the length of the reference answer."""
        if reference_answer:
            n_words = len(reference_answer.split())
            if n_words <= 15:
                length_hint = "in one concise sentence"
            elif n_words <= 50:
                length_hint = "in 2-3 sentences"
            else:
                length_hint = "in a short paragraph"
        else:
            length_hint = "in 1-3 sentences"

        messages = [
            {"role": "system",
             "content": "You are a helpful and knowledgeable assistant."},
            {"role": "user",
             "content": f"{question}\nAnswer {length_hint}."},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def load_dolly(num_sample: Optional[int] = None) -> List[Dict]:
    """QA-oriented categories of Databricks Dolly-15k."""
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    qa_cats = {"closed_qa", "open_qa", "general_qa"}
    items = []
    for row in ds:
        if row.get("category", "") not in qa_cats:
            continue
        q = row.get("instruction", "").strip()
        a = row.get("response", "").strip()
        ctx = row.get("context", "").strip()
        if ctx:
            q = f"{q}\n\nContext: {ctx}"
        if q and a:
            items.append({"question": q, "reference_answer": a})

    print(f"  raw items: {len(items)}")
    if num_sample and num_sample < len(items):
        random.seed(42)
        items = random.sample(items, num_sample)
    print(f"  using    : {len(items)} samples")
    return items


def main():
    parser = argparse.ArgumentParser(description="Collect probe supervision targets")
    parser.add_argument('--model', type=str, required=True,
                        help='Hugging Face model id or local path')
    parser.add_argument('--layer_span', type=str, required=True,
                        help='factual-salient layer span, inclusive (e.g. 12-18)')
    parser.add_argument('--output_path', type=str, default="labels/labels.pt")
    parser.add_argument('--num_sample', type=int, default=3000)
    parser.add_argument('--top_k',       type=int,   default=10)
    parser.add_argument('--max_length',  type=int,   default=128)
    parser.add_argument('--tau_entropy', type=float, default=2.0)
    args = parser.parse_args()

    factual_start, factual_end = (int(x) for x in args.layer_span.split('-'))
    collector = LabelCollector(
        model_path    = args.model,
        factual_start = factual_start,
        factual_end   = factual_end,
        top_k         = args.top_k,
        max_length    = args.max_length,
        tau_entropy   = args.tau_entropy,
    )

    dataset = load_dolly(args.num_sample)

    all_records: List[Dict] = []
    n_steps_total = 0
    n_trig_total  = 0

    pbar = tqdm(dataset, desc="Collecting")
    for idx, item in enumerate(pbar):
        prompt = collector.build_prompt(item['question'], item.get('reference_answer', ''))

        try:
            records = collector.collect_sample(prompt)
        except Exception as e:
            print(f"\n[WARN] Sample {idx} failed: {e}")
            continue

        all_records.extend(records)
        n_steps_total += len(records)
        n_trig_total  += sum(r['triggered'] for r in records)
        pbar.set_postfix_str(f"steps={n_steps_total}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    meta = {
        'dataset':        'dolly',
        'num_sample':     len(dataset),
        'num_steps':      n_steps_total,
        'trig_rate':      n_trig_total / max(n_steps_total, 1),
        'top_k':          args.top_k,
        'probe_layer':    collector.probe_layer,
        'factual_layers': collector.factual_layers,
        'hidden_dim':     collector.hidden_dim,
        'model':          args.model,
    }
    torch.save({'records': all_records, 'meta': meta}, args.output_path)

    meta_path = os.path.splitext(args.output_path)[0] + '_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  steps collected : {n_steps_total}")
    print(f"  probe layer     : {collector.probe_layer}")
    print(f"  saved           : {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
