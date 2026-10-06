"""DESCAPE decoder: beam search guided by the probe-estimated factual attribution signal."""

import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .probe import load_probe


class DescapeDecoder:
    """
    DESCAPE: signal-integrated beam search.

    At every step, each candidate token v is scored with the probe-estimated
    factual attribution signal s = delta_hat_t(v):
      S_inc(v) = log p(v) - alpha * max(0, s - tau) + gamma * 1[tau_fact <= s < tau]
    The top-B beams are kept by cumulative S_inc, duplicate sequences are removed,
    and the final sequence is selected with length normalization.
    """

    def __init__(
        self,
        model_path:     str,
        probe_ckpt:     str,
        # Beam search
        beam_width:     int   = 5,
        candidates_per_beam: int = 12,
        # Probe-guided scoring
        alpha:          float = 0.5,
        tau:            float = 3.0,
        gamma:          float = 0.3,
        factual_lo:     float = 0.5,
        # Generation
        max_length:     int   = 128,
        min_length:     int   = 12,
        repetition_penalty: float = 1.2,
        length_penalty: float = 0.6,
        penalize_prompt_tokens: bool = True,
        device:         str   = 'cuda',
    ):
        self.device = device

        # ── Load LLM ─────────────────────────────────────────────────
        print(f"Loading LLM: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map='auto',
            attn_implementation="sdpa",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        # ── Load Probe ───────────────────────────────────────────────
        print(f"Loading probe: {probe_ckpt}")
        (self.probe, self.probe_layer, self.use_embed_concat,
         hidden_dim, best_rho) = load_probe(probe_ckpt, device)
        print(f"  probe_layer={self.probe_layer}  "
              f"{'h+embed' if self.use_embed_concat else 'h-only'}  "
              f"best_rho={best_rho:.4f}")

        if self.use_embed_concat:
            self.embed_weight = (
                self.model.get_input_embeddings().weight.detach().to(device)
            )  # keep in fp16 to save ~1GB VRAM

        # ── Hidden state hook ─────────────────────────────────────────
        self._h_captured: Optional[torch.Tensor] = None

        def _hs_hook(module, inp, out):
            if isinstance(out, tuple):
                self._h_captured = out[0][:, -1, :].detach().float()
            else:
                self._h_captured = out[:, -1, :].detach().float()

        self.model.model.layers[self.probe_layer].register_forward_hook(_hs_hook)

        # ── Parameters ────────────────────────────────────────────────
        self.beam_width = beam_width
        self.candidates_per_beam = max(candidates_per_beam, beam_width)
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.factual_lo = factual_lo
        self.max_length = max_length
        self.min_length = min_length
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.penalize_prompt_tokens = penalize_prompt_tokens

        self.stats = {
            'total_tokens': 0, 'total_time': 0.0, 'forward_passes': 0,
            'dedup_replacements': 0, 'early_stops': 0,
        }
        self.log_file = None

        print(f"Ready  |  B={beam_width}  K={self.candidates_per_beam}  "
              f"α={alpha}  τ={tau}  γ={gamma}  τ_fact={factual_lo}  "
              f"lp={length_penalty}  max_len={max_length}")

    # ── KV cache helpers ──────────────────────────────────────────────

    def _expand_kv(self, past_kv, n):
        """Expand KV cache from batch=1 to batch=n (DynamicCache-aware)."""
        if isinstance(past_kv, DynamicCache):
            new_cache = DynamicCache()
            for layer_idx, layer in enumerate(past_kv.layers):
                if layer.get_seq_length() == 0:
                    continue
                k = layer.keys.repeat(n, 1, 1, 1)
                v = layer.values.repeat(n, 1, 1, 1)
                new_cache.update(k, v, layer_idx)
            return new_cache
        return tuple(
            (k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1))
            for k, v in past_kv
        )

    def _reindex_kv(self, past_kv, indices):
        """Reindex KV cache by beam parent indices (in-place for DynamicCache)."""
        if isinstance(past_kv, DynamicCache):
            for layer in past_kv.layers:
                if layer.get_seq_length() == 0:
                    continue
                dev = layer.keys.device
                idx = indices.to(dev)
                layer.keys = layer.keys.index_select(0, idx)
                layer.values = layer.values.index_select(0, idx)
            return past_kv
        return tuple(
            (k.index_select(0, indices.to(k.device)),
             v.index_select(0, indices.to(v.device)))
            for k, v in past_kv
        )

    # ── Forward ────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _forward(self, cur_ids, past_kv):
        self._h_captured = None
        out = self.model(cur_ids, past_key_values=past_kv, use_cache=True)
        self.stats['forward_passes'] += 1
        logits = out.logits[:, -1, :].float()
        h_t = self._h_captured
        return logits, out.past_key_values, h_t

    # ── Probe ──────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _probe_batch(self, h_t, topk_ids):
        """
        Batched probe scoring.
        h_t: (B, D),  topk_ids: (B, K)
        Returns: (B, K) probe scores
        """
        if h_t is None:
            return torch.zeros(
                topk_ids.shape, dtype=torch.float, device=topk_ids.device,
            )
        B, K = topk_ids.shape
        h_exp = h_t.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

        if self.use_embed_concat:
            flat_ids = topk_ids.reshape(-1)
            e_w = self.embed_weight[flat_ids].float()
            x = torch.cat([h_exp, e_w], dim=-1)
        else:
            x = h_exp

        scores = self.probe(x)
        return scores.reshape(B, K)

    # ── Scoring helpers ────────────────────────────────────────────────

    def _compute_score_inc(self, logps, probe_s):
        """
        S_inc = logp − α·max(0, s_t − τ) + γ·𝟙[τ_fact ≤ s_t < τ]
        logps: (B, K), probe_s: (B, K) → (B, K)
        """
        risk_pen = self.alpha * torch.clamp(probe_s - self.tau, min=0)
        fact_mask = (probe_s >= self.factual_lo) & (probe_s < self.tau)
        fact_bon = self.gamma * fact_mask.float()
        return logps - risk_pen + fact_bon

    def _length_norm(self, length):
        if self.length_penalty > 0:
            return ((5.0 + length) ** self.length_penalty) / \
                   (6.0 ** self.length_penalty)
        return 1.0

    # ── Beam deduplication ──────────────────────────────────────────────

    def _select_topB_dedup(
        self,
        cand_scores: torch.Tensor,  # (B, K)
        topk_ids: torch.Tensor,     # (B, K)
        beam_tokens: List[List[int]],
        beam_done: List[bool],
        B: int,
        K: int,
        eos_id: int,
    ) -> Tuple[List[int], List[int], List[int], List[float], int]:
        """
        Select the top-B candidates while removing duplicate sequences:
        if the same token sequence is reached from several beams, keep only one.

        Returns:
            parent_b, token_k, new_toks, new_scores, n_dedup
        """
        flat = cand_scores.view(-1)
        # take a larger pool to make up for candidates dropped by deduplication
        n_pool = min(B * 2, flat.numel())
        vals, top_flat = flat.topk(n_pool)

        # Move to CPU once to avoid per-element CUDA syncs
        top_flat_list = top_flat.cpu().tolist()
        vals_list = vals.cpu().tolist()
        topk_ids_cpu = topk_ids.cpu()

        parent_b = []
        token_k = []
        new_toks = []
        new_scores = []
        seen_seqs = set()
        n_dedup = 0

        for idx in range(n_pool):
            if len(parent_b) >= B:
                break

            fi = top_flat_list[idx]
            pb = fi // K
            tk = fi % K
            nt = int(topk_ids_cpu[pb, tk])
            sc = vals_list[idx]

            if beam_done[pb] and nt == eos_id:
                new_seq = tuple(beam_tokens[pb])
            else:
                new_seq = tuple(beam_tokens[pb] + [nt])

            if new_seq in seen_seqs:
                n_dedup += 1
                continue

            seen_seqs.add(new_seq)
            parent_b.append(pb)
            token_k.append(tk)
            new_toks.append(nt)
            new_scores.append(sc)

        # if the pool runs out (very rare), fill the remaining slots with the last candidate
        while len(parent_b) < B:
            parent_b.append(parent_b[-1])
            token_k.append(token_k[-1])
            new_toks.append(new_toks[-1])
            new_scores.append(new_scores[-1])

        return parent_b, token_k, new_toks, new_scores, n_dedup

    # ── Logging ────────────────────────────────────────────────────────

    def set_log_file(self, path):
        self.log_file = open(path, 'w', encoding='utf-8')
        self._log("=" * 90)
        self._log("DESCAPE decoding log")
        self._log(
            f"B={self.beam_width}  K={self.candidates_per_beam}  "
            f"α={self.alpha}  τ={self.tau}  γ={self.gamma}  "
            f"τ_fact={self.factual_lo}  lp={self.length_penalty}  "
            f"rep_pen={self.repetition_penalty}  "
            f"min_len={self.min_length}  max_len={self.max_length}"
        )
        self._log("=" * 90 + "\n")

    def close_log_file(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def _log(self, msg):
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()

    def _log_probe_dist(self, probe_s, prefix=""):
        """Log a summary of the probe score distribution."""
        if not self.log_file:
            return
        flat = probe_s.view(-1)
        n_risk = (flat > self.tau).sum().item()
        n_fact = ((flat >= self.factual_lo) & (flat < self.tau)).sum().item()
        n_safe = (flat < self.factual_lo).sum().item()
        self._log(
            f"{prefix}probe dist: "
            f"min={flat.min():.2f}  max={flat.max():.2f}  "
            f"mean={flat.mean():.2f}  std={flat.std():.2f}  |  "
            f"risk(>{self.tau})={n_risk}  "
            f"fact([{self.factual_lo},{self.tau}))={n_fact}  "
            f"safe(<{self.factual_lo})={n_safe}"
        )

    def _log_beam_status(self, t, beam_tokens, beam_scores, beam_done, B):
        """Log the current state of all beams."""
        if not self.log_file:
            return
        self._log(f"  ── All beams at t={t} ──")
        for b in range(B):
            status = "DONE" if beam_done[b] else "active"
            text = self.tokenizer.decode(beam_tokens[b], skip_special_tokens=True)
            ln = len(beam_tokens[b])
            norm_score = beam_scores[b] / self._length_norm(ln) if ln > 0 else 0
            self._log(
                f"    beam {b} [{status}]  "
                f"raw={beam_scores[b]:.3f}  norm={norm_score:.3f}  "
                f"len={ln}  "
                f"full='{text[:70]}'"
            )

    # ── Generate ───────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(self, prompt, pbar=None):
        B = self.beam_width
        K = self.candidates_per_beam
        eos_id = self.tokenizer.eos_token_id

        input_ids = self.tokenizer.encode(
            prompt, return_tensors='pt', add_special_tokens=False,
        ).to(self.device)

        meta = {
            'total_tokens': 0, 'latency': 0.0,
            'finished_beams': 0, 'dedup_count': 0,
        }
        start = time.time()

        # ── Prefill ───────────────────────────────────────────────────
        self._h_captured = None
        out = self.model(input_ids, use_cache=True)
        self.stats['forward_passes'] += 1
        logits = out.logits[:, -1, :].float()       # (1, V)
        past_kv = out.past_key_values
        h_t = self._h_captured                       # (1, D)

        self._log(f"  [PREFILL] prompt_len={input_ids.shape[1]}  "
                  f"h_t={'OK' if h_t is not None else 'NONE'}")

        # ── Step 0: Initialize beams ──────────────────────────────────
        lp = F.log_softmax(logits, dim=-1)           # (1, V)
        lp[:, eos_id] = float('-inf')

        topk_logps, topk_ids = lp.topk(K, dim=-1)   # (1, K)
        probe_s = self._probe_batch(h_t, topk_ids)   # (1, K)
        cand_scores = self._compute_score_inc(topk_logps, probe_s)  # (1, K)

        vals, top_idx = cand_scores.view(-1).topk(min(B, K))

        beam_tokens = [[int(topk_ids[0, i].item())] for i in top_idx]
        beam_scores = vals.tolist()
        beam_done = [False] * B

        self._log("[INIT] Step 0 — top candidates:")
        self._log_probe_dist(probe_s, prefix="  ")
        self._log("  Selected beams:")
        for b in range(B):
            tok = beam_tokens[b][-1]
            ps = probe_s[0, top_idx[b]].item()
            lp_val = topk_logps[0, top_idx[b]].item()
            tok_text = self.tokenizer.decode([tok])
            risk_excess = max(0, ps - self.tau)
            fact = self.factual_lo <= ps < self.tau
            self._log(
                f"    beam {b}: '{tok_text}'  "
                f"logp={lp_val:.3f}  probe={ps:.3f}  "
                f"risk_excess={risk_excess:.3f}  fact={'Y' if fact else 'N'}  "
                f"score={beam_scores[b]:.3f}"
            )

        # Expand KV to B beams
        past_kv = self._expand_kv(past_kv, B)
        first_tok = torch.tensor(
            [[beam_tokens[b][-1]] for b in range(B)],
            device=self.device,
        )
        logits, past_kv, h_t = self._forward(first_tok, past_kv)

        finished = []
        total_dedup = 0
        prompt_ids_list = input_ids[0].tolist()

        # Pre-compute prompt token mask for batched repetition penalty
        V = logits.shape[-1]
        _prompt_rep_1d = torch.zeros(V, dtype=torch.bool, device=self.device)
        if self.penalize_prompt_tokens:
            _prompt_rep_1d[list(set(prompt_ids_list))] = True

        # Pre-allocate beam scores tensor (reused each step)
        _bs_t = torch.zeros(B, device=self.device)

        # ── Steps 1 … max_length ──────────────────────────────────────
        for t in range(1, self.max_length):
            if all(beam_done):
                self._log(f"  [t={t}] All beams done — stopping.")
                break

            # ── Early stopping ────────────────────────────────────
            # If best finished beam (length-normalized) already beats
            # all active beams' current normalized scores, stop.
            # Active beams only get worse (logp<0 dominates γ bonus).
            if finished:
                best_fin_norm = max(f['score'] for f in finished)
                all_active_worse = True
                for b in range(B):
                    if not beam_done[b]:
                        cur_len = len(beam_tokens[b])
                        cur_norm = beam_scores[b] / self._length_norm(cur_len)
                        if cur_norm >= best_fin_norm:
                            all_active_worse = False
                            break
                if all_active_worse:
                    self._log(
                        f"  [t={t}] Early stop: all active beams "
                        f"worse than best finished ({best_fin_norm:.4f})"
                    )
                    self.stats['early_stops'] += 1
                    break

            active_count = sum(not d for d in beam_done)

            # ── Repetition penalty (batched) ──────────────────────
            if self.repetition_penalty != 1.0:
                rep_mask = _prompt_rep_1d.unsqueeze(0).expand(B, -1).clone()
                for b in range(B):
                    if beam_done[b]:
                        rep_mask[b].zero_()
                    elif beam_tokens[b]:
                        rep_mask[b, beam_tokens[b]] = True
                pen = torch.where(logits < 0, self.repetition_penalty,
                                  1.0 / self.repetition_penalty)
                logits = torch.where(rep_mask, logits * pen, logits)

            # ── EOS suppression ───────────────────────────────────
            if t + 1 < self.min_length:
                logits[:, eos_id] = float('-inf')

            # ── Candidates ────────────────────────────────────────
            lp = F.log_softmax(logits, dim=-1)          # (B, V)
            topk_logps, topk_ids = lp.topk(K, dim=-1)   # (B, K)
            probe_s = self._probe_batch(h_t, topk_ids)   # (B, K)
            score_inc = self._compute_score_inc(topk_logps, probe_s)

            # ── Cumulative scores ─────────────────────────────────
            for b in range(B):
                _bs_t[b] = beam_scores[b]
            cand_scores = _bs_t.unsqueeze(1) + score_inc  # (B, K)

            for b in range(B):
                if beam_done[b]:
                    cand_scores[b, :] = float('-inf')
                    cand_scores[b, 0] = beam_scores[b]
                    topk_ids[b, 0] = eos_id

            # ── Select top B with deduplication ───────────────────
            parent_b, token_k, new_toks, new_scores, n_dedup = \
                self._select_topB_dedup(
                    cand_scores, topk_ids,
                    beam_tokens, beam_done,
                    B, K, eos_id,
                )
            total_dedup += n_dedup
            if n_dedup > 0:
                self.stats['dedup_replacements'] += n_dedup

            # ── Per-step logging ──────────────────────────────────
            is_detail_step = (self.log_file is not None and
                              ((t <= 3) or (t % 5 == 0) or (n_dedup > 0)))

            if is_detail_step:
                self._log(f"\n  [t={t}] active={active_count}/{B}  "
                          f"finished={len(finished)}")
                self._log_probe_dist(probe_s, prefix="    ")
                if n_dedup > 0:
                    self._log(f"    ⚠ DEDUP: {n_dedup} duplicate beam(s) removed")
                self._log("    Selected tokens:")
                for i in range(B):
                    pb = parent_b[i]
                    nt = new_toks[i]
                    tok_text = self.tokenizer.decode([nt])
                    ps_val = float(probe_s[pb, token_k[i]].item())
                    lp_val = float(topk_logps[pb, token_k[i]].item())
                    risk_ex = max(0, ps_val - self.tau)
                    fact = self.factual_lo <= ps_val < self.tau
                    done_now = (nt == eos_id)
                    self._log(
                        f"      beam {i}: parent={pb}  "
                        f"tok='{tok_text}'  "
                        f"logp={lp_val:.3f}  probe={ps_val:.3f}  "
                        f"risk_ex={risk_ex:.3f}  "
                        f"fact={'Y' if fact else 'N'}  "
                        f"cum={new_scores[i]:.3f}"
                        f"{'  [EOS]' if done_now else ''}"
                    )

            # ── Update beams ──────────────────────────────────────
            new_beam_tokens = []
            new_beam_done = []
            for i, (pb, nt) in enumerate(zip(parent_b, new_toks)):
                if beam_done[pb] and nt == eos_id:
                    new_beam_tokens.append(list(beam_tokens[pb]))
                    new_beam_done.append(True)
                else:
                    new_beam_tokens.append(beam_tokens[pb] + [nt])
                    done = (nt == eos_id)
                    new_beam_done.append(done)
                    if done:
                        ln = len(new_beam_tokens[-1])
                        norm = new_scores[i] / self._length_norm(ln)
                        finished.append({
                            'tokens': list(new_beam_tokens[-1]),
                            'score': norm,
                            'raw': new_scores[i],
                            'length': ln,
                        })
                        self._log(
                            f"    ✓ beam {i} FINISHED: raw={new_scores[i]:.3f}  "
                            f"norm={norm:.3f}  len={ln}"
                        )

            beam_tokens = new_beam_tokens
            beam_scores = new_scores
            beam_done = new_beam_done

            # ── Full beam status (first 3 steps, then every 10 steps) ────
            if self.log_file and (t <= 3 or t % 10 == 0):
                self._log_beam_status(t, beam_tokens, beam_scores, beam_done, B)

            # ── Diversity metric ──────────────────────────────────
            if is_detail_step:
                unique_parents = len(set(parent_b))
                last_toks = [bt[-1] for bt in beam_tokens]
                unique_last = len(set(last_toks))
                self._log(
                    f"    diversity: unique_parents={unique_parents}/{B}  "
                    f"unique_last_tok={unique_last}/{B}  "
                    f"dedup_total={total_dedup}"
                )

            # ── Reindex KV + forward ──────────────────────────────
            # Skip reindex when parent indices are identity (common)
            _is_identity = all(parent_b[i] == i for i in range(B))
            if not _is_identity:
                pidx = torch.tensor(
                    parent_b, dtype=torch.long, device=self.device,
                )
                past_kv = self._reindex_kv(past_kv, pidx)
            next_in = torch.tensor(
                [[nt] for nt in new_toks], device=self.device,
            )
            logits, past_kv, h_t = self._forward(next_in, past_kv)

            if pbar is not None:
                pbar.set_postfix_str(
                    f"t={t}  fin={len(finished)}  "
                    f"dup={total_dedup}  "
                    f"{time.time()-start:.0f}s"
                )

        # ── Finalize ──────────────────────────────────────────────────
        for b in range(B):
            if not beam_done[b]:
                ln = len(beam_tokens[b])
                norm = beam_scores[b] / self._length_norm(ln)
                finished.append({
                    'tokens': beam_tokens[b],
                    'score': norm,
                    'raw': beam_scores[b],
                    'length': ln,
                })

        if not finished:
            finished.append({
                'tokens': beam_tokens[0],
                'score': beam_scores[0],
                'raw': beam_scores[0],
                'length': len(beam_tokens[0]),
            })

        best = max(finished, key=lambda x: x['score'])
        text = self.tokenizer.decode(best['tokens'], skip_special_tokens=True)

        # ── Final log ─────────────────────────────────────────────────
        sorted_fin = sorted(finished, key=lambda x: -x['score'])
        self._log(f"\n  ── FINAL: {len(finished)} candidate beams ──")
        for i, fb in enumerate(sorted_fin[:8]):
            ts = self.tokenizer.decode(
                fb['tokens'], skip_special_tokens=True,
            )[:90]
            self._log(
                f"    #{i}: norm={fb['score']:.4f}  raw={fb['raw']:.3f}  "
                f"len={fb['length']}  '{ts}'"
            )
        if len(sorted_fin) > 1:
            gap = sorted_fin[0]['score'] - sorted_fin[1]['score']
            self._log(f"  score gap (1st-2nd): {gap:.4f}")
        self._log(
            f"  [SELECTED] norm={best['score']:.4f}  len={best['length']}  "
            f"dedup_total={total_dedup}"
        )
        self._log(f"  Text: {text}\n")

        elapsed = time.time() - start
        meta['total_tokens'] = best['length']
        meta['latency'] = elapsed
        meta['finished_beams'] = len(finished)
        meta['dedup_count'] = total_dedup

        self.stats['total_tokens'] += best['length']
        self.stats['total_time'] += elapsed

        return text, meta
