# Equivalence test for the --gemma-hook-offset switch (2026-09-24): with offset 0,
# resolve_hs_idx must give the same (hook_layer, hs_idx) as the rule every published receipt
# used (commit 6543990); offset 1 may move only gemma-scope dictionaries. Needs the SAEs cached.

import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_model_sae_match import resolve_hs_idx
from sae_lens import SAE
def old_rule(cfg, sae_id):
    hook_layer = getattr(cfg, "hook_layer", None)
    if hook_layer is None:
        for cand in (getattr(cfg, "hook_name", None), str(sae_id)):
            if not cand: continue
            nums = re.findall(r"\d+", str(cand))
            if nums: hook_layer = int(nums[0]); break
    hook_name = str(getattr(cfg, "hook_name", "")) + " " + str(sae_id)
    return hook_layer, (hook_layer + 1 if "resid_post" in hook_name else hook_layer)
cases = [("gemma-scope-9b-pt-res-canonical","layer_20/width_16k/canonical"),
         ("gemma-scope-2b-pt-res-canonical","layer_12/width_16k/canonical"),
         ("gemma-scope-2b-pt-res-canonical","layer_12/width_65k/canonical"),
         ("mistral-7b-res-wg","blocks.16.hook_resid_pre"),
         ("pythia-70m-deduped-res-sm","blocks.3.hook_resid_post"),
         ("gpt2-small-res-jb","blocks.6.hook_resid_pre")]
for rel, sid in cases:
    sae = SAE.from_pretrained(rel, sid, device="cpu")
    if isinstance(sae, tuple): sae = sae[0]
    cfg = sae.cfg
    o = old_rule(cfg, sid); n0 = resolve_hs_idx(cfg, sid, rel, 0); n1 = resolve_hs_idx(cfg, sid, rel, 1)
    print(rel, sid, "old", o, "off0", n0[:2], "off1", n1[:2], "meta", n0[2], "SAME" if o == n0[:2] else "DIFF")
    del sae
from qwen_sae_adapter import load_qwen_sae
q = load_qwen_sae("Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "layer14", "cpu")
print("qwen1.7b old", old_rule(q.cfg, "layer14"), "off0", resolve_hs_idx(q.cfg, "layer14", "qwenpt:Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", 0)[:2], "off1", resolve_hs_idx(q.cfg, "layer14", "qwenpt:x", 1)[:2])
