"""
共享工具函数 — 模型加载、归因图生成、特征分析、扰动实验

设计说明:
    circuit-tracer 库的 Graph 是张量表示 (adjacency_matrix + active_features),
    没有 .nodes/.edges 列表。本模块在其上构建轻量适配层 (AdaptedGraph 系列),
    向 01-05 实验脚本暴露节点-边语义,便于跨语言/跨模型的特征比对。

主要 API:
    - load_model_and_transcoders(cfg) -> ReplacementModel
    - generate_attribution_graph(model, prompt, ...) -> (AdaptedGraph, metadata)
    - batch_generate_graphs(model, prompts, output_dir) -> list[(AdaptedGraph, meta)]
    - extract_feature_ids(graph) -> set of (layer, feat_idx)
    - compute_feature_overlap(g_a, g_b) -> Jaccard 等
    - compute_graph_structure_features(g) -> 结构性指标
    - run_perturbation(model, prompt, feature_id, scale) -> 扰动结果
"""
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

from config import ModelConfig, RESULTS_DIR


# ============================================================
# 适配层 — 在张量 Graph 之上提供节点/边语义
# ============================================================

@dataclass
class AdaptedNode:
    """合成节点。kind ∈ {'feature', 'error', 'token', 'logit'}。"""
    node_id: int
    kind: str
    layer: int
    pos: int
    feat_idx: int = -1
    influence: float = 0.0
    activation: float = 0.0
    label: str = ""

    @property
    def feature_id(self):
        """(layer, pos, feat_idx) 三元组,只对 feature 节点有意义。其余返回 None。"""
        if self.kind == "feature":
            return (self.layer, self.pos, self.feat_idx)
        return None

    @property
    def direct_effect(self):
        """legacy alias — 与 cumulative influence score 对齐。"""
        return self.influence


@dataclass
class AdaptedEdge:
    source: int  # node_id
    target: int  # node_id
    weight: float


@dataclass
class AdaptedGraph:
    """对 circuit_tracer.graph.Graph 的轻量包装。"""
    raw_graph: object
    nodes: list
    edges: list
    target_token: str
    node_index: dict = field(default_factory=dict)


def _build_adapted_graph(raw_graph, node_mask, edge_mask, cumulative_scores, tokenizer):
    """把张量 Graph + prune 出的 mask 转成节点/边列表。"""
    n_features = len(raw_graph.selected_features)
    n_layers = raw_graph.cfg.n_layers
    error_end_idx = n_features + raw_graph.n_pos * n_layers
    token_end_idx = error_end_idx + len(raw_graph.input_tokens)

    nodes = []
    node_index = {}

    nz = node_mask.nonzero()
    if nz.dim() > 1:
        nz = nz.squeeze(-1)
    kept_indices = nz.tolist()
    if isinstance(kept_indices, int):
        kept_indices = [kept_indices]

    for node_idx in kept_indices:
        score = float(cumulative_scores[node_idx])

        if node_idx < n_features:
            layer, pos, feat_idx = raw_graph.active_features[
                raw_graph.selected_features[node_idx]
            ].tolist()
            activation = float(
                raw_graph.activation_values[raw_graph.selected_features[node_idx]]
            )
            n = AdaptedNode(
                node_id=node_idx, kind="feature",
                layer=int(layer), pos=int(pos), feat_idx=int(feat_idx),
                influence=score, activation=activation,
                label=f"L{layer}/P{pos}/F{feat_idx}",
            )
        elif node_idx < error_end_idx:
            layer, pos = divmod(node_idx - n_features, raw_graph.n_pos)
            n = AdaptedNode(
                node_id=node_idx, kind="error",
                layer=int(layer), pos=int(pos),
                influence=score,
                label=f"err/L{layer}/P{pos}",
            )
        elif node_idx < token_end_idx:
            pos = node_idx - error_end_idx
            tok_id = int(raw_graph.input_tokens[pos])
            tok_str = tokenizer.decode([tok_id]) if tokenizer is not None else str(tok_id)
            n = AdaptedNode(
                node_id=node_idx, kind="token",
                layer=-1, pos=int(pos),
                influence=score,
                label=f"tok[{pos}]={tok_str!r}",
            )
        else:
            tgt_pos = node_idx - token_end_idx
            tgt = raw_graph.logit_targets[tgt_pos]
            tgt_str = getattr(tgt, "token_str", str(tgt))
            n = AdaptedNode(
                node_id=node_idx, kind="logit",
                layer=int(n_layers), pos=int(raw_graph.n_pos - 1),
                influence=score,
                label=f"logit={tgt_str!r}",
            )
        nodes.append(n)
        node_index[node_idx] = n

    # edges
    edges = []
    em = edge_mask.detach().cpu().numpy() if hasattr(edge_mask, "cpu") else np.asarray(edge_mask)
    am_t = raw_graph.adjacency_matrix
    am = am_t.detach().cpu().numpy() if hasattr(am_t, "cpu") else np.asarray(am_t)
    dsts, srcs = em.nonzero()
    for src, dst in zip(srcs.tolist(), dsts.tolist()):
        if src in node_index and dst in node_index:
            edges.append(AdaptedEdge(source=src, target=dst, weight=float(am[dst, src])))

    target_str = ""
    if raw_graph.logit_targets:
        target_str = getattr(raw_graph.logit_targets[0], "token_str", "") or ""

    return AdaptedGraph(
        raw_graph=raw_graph, nodes=nodes, edges=edges,
        target_token=target_str, node_index=node_index,
    )


# ============================================================
# 模型加载
# ============================================================

def load_model_and_transcoders(cfg: ModelConfig):
    """
    加载模型 + transcoders, 返回 circuit-tracer 的 ReplacementModel。

    使用 ReplacementModel.from_pretrained classmethod (符合真实 API)。
    """
    from circuit_tracer import ReplacementModel

    print(f"Loading {cfg.name}...")
    print(f"  Model: {cfg.model_name}")
    print(f"  Transcoder: {cfg.transcoder_name}")
    print(f"  Backend: {cfg.backend}")
    print(f"  Device: {cfg.device}")

    t0 = time.time()
    device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device

    extra_kwargs = {}
    if cfg.cache_dir:
        extra_kwargs["cache_dir"] = cfg.cache_dir

    model = ReplacementModel.from_pretrained(
        model_name=cfg.model_name,
        transcoder_set=cfg.transcoder_name,
        backend=cfg.backend,
        device=device,
        dtype=getattr(torch, cfg.dtype),
        lazy_encoder=False,
        lazy_decoder=True,
        **extra_kwargs,
    )

    # 兼容旧 API: 01_hello_world/run.py 调 `model.run(prompt).logits`
    if not hasattr(model, "run"):
        def _run(prompt):
            logits, _ = model.feature_intervention(prompt, [])
            return SimpleNamespace(logits=logits)
        try:
            model.run = _run
        except AttributeError:
            # 若 model 是 nnsight LanguageModel 不允许动态属性, 退而绑定到包装器
            pass

    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model


# ============================================================
# 归因图生成
# ============================================================

def generate_attribution_graph(
    model,
    prompt: str,
    target_token: Optional[str] = None,
    threshold: Optional[float] = None,  # legacy, 保留入参但不使用
    max_nodes: Optional[int] = None,    # legacy, 保留入参但不使用
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    max_feature_nodes: int = 7500,
    batch_size: int = 256,
    offload: Optional[str] = None,
    verbose: bool = False,
):
    """
    对一个 prompt 生成归因图。

    Args:
        model: ReplacementModel 实例
        prompt: 输入文本
        target_token: 目标 token (None 时自动选择 top logits)
        node_threshold: 保留节点累积影响阈值 (0-1)
        edge_threshold: 保留边累积影响阈值 (0-1)
        max_feature_nodes: feature 节点数上限
        batch_size: 反向传播 batch 大小
        offload: 'cpu' / 'disk' / None
        verbose: 是否打印进度
        threshold, max_nodes: legacy kwargs, 接受但不使用 (避免破坏 03/04 调用)

    Returns:
        (AdaptedGraph, metadata dict)
    """
    from circuit_tracer import attribute
    from circuit_tracer.graph import prune_graph

    t0 = time.time()

    targets = [target_token] if target_token else None

    raw_graph = attribute(
        prompt=prompt,
        model=model,
        attribution_targets=targets,
        max_n_logits=10,
        desired_logit_prob=0.95,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose,
    )

    node_mask, edge_mask, cumulative_scores = prune_graph(
        raw_graph,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    tokenizer = getattr(model, "tokenizer", None)
    graph = _build_adapted_graph(raw_graph, node_mask, edge_mask, cumulative_scores, tokenizer)

    elapsed = time.time() - t0

    metadata = {
        "prompt": prompt,
        "target_token": target_token or graph.target_token,
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
        "num_feature_nodes": sum(1 for n in graph.nodes if n.kind == "feature"),
        "num_error_nodes": sum(1 for n in graph.nodes if n.kind == "error"),
        "node_threshold": node_threshold,
        "edge_threshold": edge_threshold,
        "elapsed_seconds": elapsed,
    }

    print(f"  Graph: {metadata['num_nodes']} nodes, {metadata['num_edges']} edges ({elapsed:.1f}s)")
    return graph, metadata


def batch_generate_graphs(
    model,
    prompts: list,
    output_dir,
    threshold: Optional[float] = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
):
    """批量生成归因图并保存。接口与旧版兼容 (threshold 仍可传, 已被忽略)。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, p in enumerate(tqdm(prompts, desc="Generating graphs")):
        prompt_text = p["text"]
        target = p.get("target", None)

        try:
            graph, meta = generate_attribution_graph(
                model, prompt_text, target,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
            meta.update({k: v for k, v in p.items() if k not in ("text", "target")})

            save_path = output_dir / f"graph_{i:04d}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump({"graph": graph, "metadata": meta}, f)

            results.append((graph, meta))

        except Exception as e:
            import traceback
            print(f"  ERROR on prompt {i}: {e}")
            traceback.print_exc()
            results.append((None, {"error": str(e), "prompt": prompt_text}))

    meta_path = output_dir / "metadata.json"
    all_meta = [r[1] for r in results]
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    return results


# ============================================================
# 特征分析
# ============================================================

def extract_feature_ids(graph) -> set:
    """
    提取所有 feature 节点的位置无关 ID, 用于跨语言/跨模型对比。
    返回 {(layer, feat_idx)} 集合 (忽略 pos, 因为不同 prompt 的 pos 不同)。
    """
    return {(n.layer, n.feat_idx) for n in graph.nodes if n.kind == "feature"}


def compute_feature_overlap(graph_a, graph_b) -> dict:
    """计算两图的 feature 重叠 (Jaccard)。"""
    feats_a = extract_feature_ids(graph_a)
    feats_b = extract_feature_ids(graph_b)

    shared = feats_a & feats_b
    union = feats_a | feats_b

    return {
        "jaccard": len(shared) / len(union) if union else 0.0,
        "overlap_count": len(shared),
        "total_a": len(feats_a),
        "total_b": len(feats_b),
        "shared_features": [list(t) for t in shared],
    }


def compute_graph_structure_features(graph) -> dict:
    """提取归因图结构指标 (节点/边数、密度、层跨度、度分布等)。"""
    nodes = graph.nodes
    edges = graph.edges

    n_nodes = len(nodes)
    n_edges = len(edges)

    if n_nodes == 0:
        return {"n_nodes": 0, "n_edges": 0}

    layers = [n.layer for n in nodes if n.layer >= 0]
    layer_min = min(layers) if layers else 0
    layer_max = max(layers) if layers else 0

    weights = [abs(e.weight) for e in edges]
    weight_mean = float(np.mean(weights)) if weights else 0.0
    weight_std = float(np.std(weights)) if weights else 0.0
    weight_max = float(max(weights)) if weights else 0.0

    in_degree = {}
    out_degree = {}
    for e in edges:
        out_degree[e.source] = out_degree.get(e.source, 0) + 1
        in_degree[e.target] = in_degree.get(e.target, 0) + 1
    in_degrees = list(in_degree.values()) or [0]
    out_degrees = list(out_degree.values()) or [0]

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_feature_nodes": sum(1 for n in nodes if n.kind == "feature"),
        "n_error_nodes": sum(1 for n in nodes if n.kind == "error"),
        "density": n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0,
        "layer_span": layer_max - layer_min,
        "layer_min": layer_min,
        "layer_max": layer_max,
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "weight_max": weight_max,
        "in_degree_mean": float(np.mean(in_degrees)),
        "in_degree_max": int(max(in_degrees)),
        "out_degree_mean": float(np.mean(out_degrees)),
        "out_degree_max": int(max(out_degrees)),
    }


# ============================================================
# Perturbation 实验
# ============================================================

def run_perturbation(
    model,
    prompt: str,
    feature_id,
    scale: float = 0.0,
    target_token: Optional[str] = None,
):
    """
    对指定 feature 做 intervention, 观察输出变化。

    Args:
        model: ReplacementModel
        prompt: 输入文本
        feature_id: (layer, pos, feat_idx) 三元组 (来自 AdaptedNode.feature_id)
                    或 (layer, feat_idx) 二元组 (此时 pos 默认 -1, slice 应用到所有位置)
        scale: 干预值 (0.0 = ablate; 任意 float = 替换激活)
        target_token: 观察该 token 的 logit 变化

    Returns:
        dict
    """
    if not isinstance(feature_id, (tuple, list)) or len(feature_id) not in (2, 3):
        raise ValueError(f"feature_id must be (layer, pos, feat_idx) or (layer, feat_idx); got {feature_id}")

    if len(feature_id) == 3:
        layer, pos, feat_idx = feature_id
    else:
        layer, feat_idx = feature_id
        pos = slice(0, None)  # 默认作用到所有位置

    # baseline (空干预)
    original_logits, _ = model.feature_intervention(prompt, [])

    # 实际干预
    intervention = [(int(layer), pos, int(feat_idx), float(scale))]
    new_logits, _ = model.feature_intervention(prompt, intervention)

    # 取最后位置的 logits
    orig_last = original_logits[0, -1] if original_logits.dim() == 3 else original_logits[-1]
    new_last = new_logits[0, -1] if new_logits.dim() == 3 else new_logits[-1]

    tokenizer = getattr(model, "tokenizer", None)
    top5_orig = torch.topk(orig_last, 5)
    top5_pert = torch.topk(new_last, 5)

    def decode_topk(topk):
        if tokenizer is None:
            return [(str(idx.item()), score.item())
                    for idx, score in zip(topk.indices, topk.values)]
        return [(tokenizer.decode([idx.item()]), score.item())
                for idx, score in zip(topk.indices, topk.values)]

    result = {
        "feature_id": list(feature_id),
        "scale": scale,
        "top5_original": decode_topk(top5_orig),
        "top5_perturbed": decode_topk(top5_pert),
    }

    if target_token and tokenizer is not None:
        target_id = tokenizer.encode(target_token, add_special_tokens=False)
        if target_id:
            tid = target_id[0]
            result["target_token"] = target_token
            result["original_logit"] = orig_last[tid].item()
            result["perturbed_logit"] = new_last[tid].item()
            result["delta_logit"] = result["perturbed_logit"] - result["original_logit"]

    return result


# ============================================================
# 保存 / 加载
# ============================================================

def save_results(data, path):
    """保存实验结果 (JSON for metadata, pickle for graphs)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    elif path.suffix == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    print(f"  Saved to {path}")


def load_results(path):
    path = Path(path)
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
