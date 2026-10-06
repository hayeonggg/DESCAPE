"""Probe f_phi that approximates the factual attribution signal from a single forward pass."""

from pathlib import Path

import torch
import torch.nn as nn

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"

# Factual-salient layer span (inclusive) identified for each model in the paper.
LAYER_SPANS = {
    "llama":   (12, 18),   # Llama-3.1-8B-Instruct
    "mistral": (20, 26),   # Mistral-7B-Instruct-v0.3
    "qwen":    (12, 18),   # Qwen2.5-7B-Instruct
}


class DeltaProbeHead(nn.Module):
    """
    f_phi([h_t ; e(v)]) -> delta_hat_t(v)

    Input  : (2 * hidden_dim,)  hidden state at the last layer of the factual-salient
             span, concatenated with the embedding of candidate token v
    Output : scalar estimate of the factual attribution signal
    """

    def __init__(self, input_dim: int, mlp_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def resolve_probe_path(probe: str) -> str:
    """Map a released probe name (llama / mistral / qwen) to its checkpoint; otherwise return the path."""
    if probe in LAYER_SPANS:
        return str(CHECKPOINT_DIR / f"probe_{probe}.pt")
    return probe


def load_probe(ckpt_path: str, device: str = 'cuda'):
    ckpt = torch.load(resolve_probe_path(ckpt_path), map_location=device, weights_only=True)
    config = ckpt['config']
    state = ckpt['head_state_dict']

    input_dim = config['input_dim']
    probe_layer = config['probe_layer']
    hidden_dim = config.get('hidden_dim', input_dim // 2)
    use_embed_concat = (input_dim == 2 * hidden_dim)

    probe = DeltaProbeHead(input_dim, mlp_dim=state['net.0.weight'].shape[0]).to(device)
    probe.load_state_dict(state)
    probe.eval()

    best_rho = max(ckpt.get('val_rhos', [float('nan')]))
    return probe, probe_layer, use_embed_concat, hidden_dim, best_rho
