"""Train the probe f_phi on the collected attribution signals.

The probe is trained with a spike-weighted Huber loss, and the checkpoint with the
best Spearman correlation on the validation split is kept.

Usage:
  python train_probe.py --model meta-llama/Llama-3.1-8B-Instruct \
      --labels_path labels/labels_llama.pt --save_dir checkpoints/llama
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM

from descape.probe import DeltaProbeHead


class DeltaStepDataset(Dataset):
    """
    Builds (probe input, target) pairs from the collected records.
    Each record (one decoding step) yields K pairs, one per candidate token:
      x = [h_t ; e(v)]  (2D,),  y = delta_real(v)
    """

    def __init__(self, records: List[Dict], embed_weight: torch.Tensor):
        self.items: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for rec in records:
            h    = rec['h'].float()                      # (D,)
            tids = rec['top_ids']                        # List[int] K
            dy   = rec['delta']                          # (K,)

            e_w   = embed_weight[tids]                   # (K, D)
            h_rep = h.unsqueeze(0).expand(len(tids), -1) # (K, D)
            x     = torch.cat([h_rep, e_w], dim=-1)      # (K, 2D)

            for i in range(len(tids)):
                self.items.append((x[i], dy[i]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.items[idx]


class ProbeTrainer:

    def __init__(
        self,
        model_path:  str,
        hidden_dim:  int,
        probe_layer: int,
        factual_layers: List[int],
        mlp_dim:  int   = 256,
        dropout:  float = 0.1,
        lr:       float = 3e-4,
        huber_delta:  float = 1.0,
        spike_weight: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.hidden_dim     = hidden_dim
        self.probe_layer    = probe_layer
        self.factual_layers = factual_layers
        self.huber_delta    = huber_delta
        self.spike_weight   = spike_weight

        # Token embedding matrix of the frozen LLM
        print(f"Loading embedding weights from {model_path} ...")
        llm = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="cpu"
        )
        self.embed_weight = llm.get_input_embeddings().weight.detach().float().cpu()
        del llm
        print(f"  embed_weight: {self.embed_weight.shape}")

        self.head  = DeltaProbeHead(2 * hidden_dim, mlp_dim, dropout).to(device)
        self.optim = AdamW(self.head.parameters(), lr=lr)

        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []
        self.val_rhos:     List[float] = []

    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Huber loss with spike-sensitive weights that up-weight large positive targets."""
        pred = self.head(x)
        elem = F.huber_loss(pred, y, delta=self.huber_delta, reduction="none")
        if self.spike_weight > 0:
            w = F.softmax(self.spike_weight * y.clamp(min=0), dim=0) * len(y)
            return (elem * w).mean()
        return elem.mean()

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, Optional[float]]:
        """Returns (average loss, Spearman rho); rho is computed on validation only."""
        self.head.train(train)
        total_loss = 0.0
        n_batches  = 0
        all_pred: List[float] = []
        all_true: List[float] = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if train:
                self.optim.zero_grad()
                loss = self._loss(x_batch, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
                self.optim.step()
            else:
                with torch.no_grad():
                    pred = self.head(x_batch)
                    loss = F.huber_loss(pred, y_batch,
                                        delta=self.huber_delta, reduction="mean")
                    all_pred.extend(pred.cpu().tolist())
                    all_true.extend(y_batch.cpu().tolist())

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        rho = None
        if not train and len(all_pred) >= 2:
            rho = float(spearmanr(all_pred, all_true)[0])
        return avg_loss, rho

    def train(
        self,
        records:    List[Dict],
        num_epochs: int   = 30,
        batch_size: int   = 512,
        val_ratio:  float = 0.2,
        save_dir:   Optional[str] = None,
    ) -> float:
        full_ds = DeltaStepDataset(records, self.embed_weight)
        n_total = len(full_ds)
        n_val   = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"\n  Dataset: {n_total} pairs (steps x top_k)  ->  train={n_train}, val={n_val}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

        best_rho = -float('inf')
        for epoch in range(1, num_epochs + 1):
            train_loss, _ = self._run_epoch(train_loader, train=True)
            val_loss, rho = self._run_epoch(val_loader,   train=False)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_rhos.append(rho if rho is not None else float('nan'))

            print(f"Epoch {epoch:2d}/{num_epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  Spearman_rho={rho:.4f}")

            if save_dir and rho is not None and rho > best_rho:
                best_rho  = rho
                best_path = self.save(save_dir, tag="best")
                print(f"             best rho={best_rho:.4f}  saved -> {best_path}")

        return best_rho

    @torch.no_grad()
    def evaluate(self, records: List[Dict]) -> Dict:
        """Spearman rho over all steps and over triggered steps, plus MAE and bias."""
        self.head.eval()
        all_pred, all_true = [], []
        pred_trig, true_trig = [], []
        n_trig_steps = 0

        for rec in records:
            h    = rec['h'].float()
            tids = rec['top_ids']
            dy   = rec['delta']

            e_w   = self.embed_weight[tids]
            h_rep = h.unsqueeze(0).expand(len(tids), -1)
            x     = torch.cat([h_rep, e_w], dim=-1).to(self.device)
            pred  = self.head(x).cpu()

            all_pred.extend(pred.tolist())
            all_true.extend(dy.tolist())
            if rec.get('triggered', False):
                pred_trig.extend(pred.tolist())
                true_trig.extend(dy.tolist())
                n_trig_steps += 1

        rho_all,  _ = spearmanr(all_pred,  all_true)
        rho_trig, _ = spearmanr(pred_trig, true_trig) if pred_trig else (float('nan'), None)

        pred_arr = np.array(all_pred)
        true_arr = np.array(all_true)
        return {
            'rho_all':           rho_all,
            'rho_triggered':     rho_trig,
            'mae':               float(np.abs(pred_arr - true_arr).mean()),
            'bias':              float((pred_arr - true_arr).mean()),
            'n_pairs':           len(all_pred),
            'n_triggered_steps': n_trig_steps,
        }

    def save(self, save_dir: str, tag: str = "final") -> str:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"probe_{tag}.pt")
        torch.save({
            'head_state_dict': self.head.state_dict(),
            'val_rhos':        self.val_rhos,
            'config': {
                'hidden_dim':     self.hidden_dim,
                'probe_layer':    self.probe_layer,
                'factual_layers': self.factual_layers,
                'input_dim':      2 * self.hidden_dim,
            },
        }, path)
        return path


def main():
    parser = argparse.ArgumentParser(description="Train the DESCAPE probe")
    parser.add_argument('--model', type=str, required=True,
                        help='Hugging Face model id or local path (for the token embeddings)')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='.pt file created by collect_labels.py')
    parser.add_argument('--save_dir', type=str, default="checkpoints/probe")

    parser.add_argument('--num_epochs',  type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=512)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--val_ratio',   type=float, default=0.2)
    parser.add_argument('--mlp_dim',     type=int,   default=256)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--huber_delta',  type=float, default=1.0,
                        help='Huber threshold')
    parser.add_argument('--spike_weight', type=float, default=2.0,
                        help='Scaling factor of the spike-sensitive weights (0 = plain Huber loss)')
    args = parser.parse_args()

    print(f"Loading labels from {args.labels_path} ...")
    saved = torch.load(args.labels_path, map_location='cpu', weights_only=True)
    records = saved['records']
    meta    = saved['meta']
    print(f"  steps       : {len(records)}")
    print(f"  probe_layer : {meta['probe_layer']}")

    trainer = ProbeTrainer(
        model_path     = args.model,
        hidden_dim     = meta['hidden_dim'],
        probe_layer    = meta['probe_layer'],
        factual_layers = meta['factual_layers'],
        mlp_dim        = args.mlp_dim,
        dropout        = args.dropout,
        lr             = args.lr,
        huber_delta    = args.huber_delta,
        spike_weight   = args.spike_weight,
    )

    best_rho = trainer.train(
        records    = records,
        num_epochs = args.num_epochs,
        batch_size = args.batch_size,
        val_ratio  = args.val_ratio,
        save_dir   = args.save_dir,
    )
    final_path = trainer.save(args.save_dir, tag="final")
    print(f"\nBest validation Spearman rho = {best_rho:.4f}")
    print(f"Final checkpoint: {final_path}")

    eval_result = trainer.evaluate(records)
    print(f"\n{'='*60}")
    print(f"  Spearman rho (all steps)       : {eval_result['rho_all']:.4f}")
    print(f"  Spearman rho (triggered steps) : {eval_result['rho_triggered']:.4f}")
    print(f"  MAE                            : {eval_result['mae']:.4f}")
    print(f"{'='*60}")

    result_path = os.path.join(args.save_dir, "eval_result.json")
    with open(result_path, 'w') as f:
        json.dump({
            'eval': eval_result,
            'training_config': {
                'num_epochs':   args.num_epochs,
                'batch_size':   args.batch_size,
                'lr':           args.lr,
                'huber_delta':  args.huber_delta,
                'spike_weight': args.spike_weight,
                'val_ratio':    args.val_ratio,
            },
            'data_meta': meta,
            'best_val_rho': best_rho,
        }, f, indent=2)
    print(f"Saved -> {result_path}")


if __name__ == "__main__":
    main()
