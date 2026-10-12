#!/usr/bin/env python3
"""Manual loader for Qwen's released residual SAEs (not registered in sae-lens).

Repo format (e.g. Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50): per-layer state dicts
`layerN.sae.pt` with W_enc [d_sae, d_model], W_dec [d_model, d_sae], b_enc, b_dec.
Encode convention (from the repo's own app.py): pre = h @ W_enc.T + b_enc, then
TopK-ReLU with k = the repo's L0 (50 for *-L0_50). Hook point: resid_post of layer N
(the output of model.model.layers[N], i.e. hidden_states[N+1] in HF terms).

The adapter exposes the attribute surface our matching scripts use from sae-lens SAEs:
  .encode(h) -> acts   .W_enc (for dtype)   .W_dec [d_sae, d_model]   .cfg.{hook_layer,hook_name,d_sae}
"""
import re
import torch


class _Cfg:
    def __init__(self, hook_layer, d_sae):
        self.hook_layer = hook_layer
        self.hook_name = f"blocks.{hook_layer}.hook_resid_post"
        self.d_sae = d_sae


class QwenSAE:
    def __init__(self, sd, hook_layer, k, device):
        # stored: W_enc [d_sae, d_model], W_dec [d_model, d_sae]
        self._W_encT = sd["W_enc"].float().T.to(device)   # [d_model, d_sae]
        self.W_enc = sd["W_enc"].float().to(device)       # [d_sae, d_model] (dtype probe)
        self.W_dec = sd["W_dec"].float().T.contiguous().to(device)  # [d_sae, d_model] sae-lens convention
        self.b_enc = sd["b_enc"].float().to(device)
        self.k = k
        self.cfg = _Cfg(hook_layer, self.W_enc.shape[0])

    def to(self, device):
        return self

    @torch.no_grad()
    def encode(self, h):
        pre = h.float() @ self._W_encT + self.b_enc
        relu = torch.relu(pre)
        v, idx = torch.topk(relu, self.k, dim=-1)
        out = torch.zeros_like(relu)
        out.scatter_(-1, idx, v)
        return out


def load_qwen_sae(repo, sae_id, device):
    """sae_id: 'layer14' (or anything containing the layer int)."""
    from huggingface_hub import hf_hub_download
    layer = int(re.findall(r"\d+", str(sae_id))[0])
    m = re.search(r"L0[_-](\d+)", repo)
    k = int(m.group(1)) if m else 50
    p = hf_hub_download(repo, f"layer{layer}.sae.pt")
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    return QwenSAE(sd, layer, k, device)
