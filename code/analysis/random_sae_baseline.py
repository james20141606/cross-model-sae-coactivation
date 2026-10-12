"""Matched random-SAE baselines — the control the 2026 SAE literature now requires.

PROVENANCE: copied from a sibling analysis's random_sae_baseline.py (shared utility). This copy adds ONE circuit_tracing-specific fix: Qwen's
manually-loaded SAEs (code/analysis/qwen_sae_adapter.py, "qwenpt:" releases) store
`.W_enc` TRANSPOSED ([d_sae, d_model]) relative to the sae_lens convention
([d_in, d_sae]) that this shim otherwise assumes -- see the orientation-detection block
in RandomSAEShim.__init__ below.

Motivation
----------
`arXiv:2602.14111` ("Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random
Baselines?") reports random directions matching trained SAEs on interpretability
(0.87 vs 0.90), sparse probing (0.69 vs 0.72) and **causal editing (0.73 vs 0.72)**.
Reviewers in this cycle ask, of any SAE-based effect: *would random directions of
matched dimension and matched sparsity produce the same number?* An effect that
survives the random control is not an SAE finding.

This module supplies drop-in shims exposing the only interface the attribution
path needs — a callable `encode(h) -> sparse code` plus `.cfg` metadata — so an
existing pipeline can be re-run unchanged against a matched random control.

Modes
-----
``random-dir``
    Encoder directions replaced by random unit vectors (Gaussian, row-normalised).
    Destroys everything learned; keeps d_sae and L0. The standard "does it beat
    random" control.
``rotate``
    The *real* encoder weights composed with a random orthogonal rotation of the
    input space. Preserves the weight spectrum/geometry but destroys feature
    identity and any alignment with the residual basis. The **more conservative**
    control: an effect surviving this is about gross geometry, not about *which*
    features were learned.

Sparsity is matched by top-k on the pre-activation, with k calibrated to the real
SAE's mean L0 on a calibration batch (so the random code has the same number of
active features per state as the trained one).

Reusable across the SAE papers (several analyses):
the shim only assumes `real_sae.encode`, `real_sae.cfg.d_sae` and, if present,
`real_sae.cfg.d_in` / `cfg.metadata["hook_name"]`.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

log = logging.getLogger(__name__)

MODES = ("random-dir", "rotate")


def _cfg_get(cfg, name, default=None):
    """cfg may be a dataclass-like object or expose values under .metadata."""
    if hasattr(cfg, name):
        return getattr(cfg, name)
    meta = getattr(cfg, "metadata", None)
    if isinstance(meta, dict) and name in meta:
        return meta[name]
    return default


@torch.no_grad()
def calibrate_l0(real_sae, states, max_states: int = 64) -> int:
    """Mean number of active features of the *trained* SAE over calibration states."""
    counts = []
    for h in states[:max_states]:
        z = real_sae.encode(h.unsqueeze(0)).squeeze(0)
        counts.append(int((z != 0).sum().item()))
    if not counts:
        return 32
    k = max(1, int(round(sum(counts) / len(counts))))
    log.info(f"[random-sae] calibrated L0={k} over {len(counts)} states")
    return k


class RandomSAEShim:
    """Matched random control exposing `.encode` and `.cfg` like a sae_lens SAE."""

    def __init__(self, real_sae, mode: str, seed: int, k: int,
                 device=None, dtype=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.k = int(k)
        self.cfg = real_sae.cfg  # reuse hook_name / d_sae metadata verbatim

        W_enc = getattr(real_sae, "W_enc", None)
        if W_enc is None:
            raise AttributeError("real_sae has no W_enc; cannot build a matched control")
        device = device or W_enc.device
        dtype = dtype or W_enc.dtype
        W_enc_f = W_enc.detach()

        # ORIENTATION FIX (circuit_tracing-specific): sae_lens SAEs store W_enc as
        # [d_in, d_sae] (encode: h @ W_enc). qwen_sae_adapter.QwenSAE instead stores
        # W_enc as [d_sae, d_model] (its own encode uses a separately-cached
        # `_W_encT` = W_enc.T for the actual matmul; `.W_enc` is kept transposed only
        # as a dtype/shape probe -- see qwen_sae_adapter.py:26-28). If we don't correct
        # for this here, d_in/d_sae get swapped and the shim silently builds a
        # (d_sae x d_in)-shaped encoder instead of (d_in x d_sae), breaking both the
        # random-dir scale match and the rotate-mode matmul shapes.
        cfg_d_sae = _cfg_get(self.cfg, "d_sae", None)
        if (cfg_d_sae is not None and W_enc_f.shape[0] == int(cfg_d_sae)
                and W_enc_f.shape[1] != int(cfg_d_sae)):
            log.info("[random-sae] detected transposed W_enc (qwenpt-style "
                     f"[d_sae={W_enc_f.shape[0]}, d_in={W_enc_f.shape[1]}]); "
                     "transposing to sae_lens convention [d_in, d_sae]")
            W_enc_f = W_enc_f.T

        d_in, d_sae = W_enc_f.shape[0], W_enc_f.shape[1]
        self.d_sae = int(d_sae)

        g = torch.Generator(device="cpu").manual_seed(int(seed))
        if mode == "random-dir":
            W = torch.randn(d_in, d_sae, generator=g, dtype=torch.float32)
            W = W / W.norm(dim=0, keepdim=True).clamp_min(1e-6)
            # match the trained encoder's overall scale so top-k sees a comparable range
            # (.item(): W lives on CPU here while W_enc may be on cuda)
            W = W * W_enc_f.float().norm(dim=0, keepdim=True).mean().item()
            self.W = W.to(device=device, dtype=dtype)
        else:  # rotate — random orthogonal mixing of the input space
            A = torch.randn(d_in, d_in, generator=g, dtype=torch.float32)
            Q, _ = torch.linalg.qr(A)
            self.W = (Q.to(device=device, dtype=torch.float32)
                      @ W_enc_f.float()).to(dtype=dtype)
        log.info(f"[random-sae] mode={mode} seed={seed} d_in={d_in} "
                 f"d_sae={d_sae} k={self.k}")

    @torch.no_grad()
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """(..., d_in) -> (..., d_sae) with exactly k active entries per row."""
        pre = torch.relu(h.to(self.W.dtype) @ self.W)
        k = min(self.k, pre.shape[-1])
        z = torch.zeros_like(pre)
        vals, idx = torch.topk(pre, k, dim=-1)
        z.scatter_(-1, idx, vals)
        return z

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        return self.encode(h)


def wrap_pair(sae_src, sae_dst, mode: str, seed: int, k_src: int, k_dst: int):
    """Build matched random controls for a (src, dst) SAE pair.

    Distinct seeds per side so the two shims are independent.
    """
    return (
        RandomSAEShim(sae_src, mode, seed, k_src),
        RandomSAEShim(sae_dst, mode, seed + 10_000, k_dst),
    )
