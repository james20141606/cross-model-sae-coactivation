"""
实验 03: 跨语言 Feature Sharing 分析
核心论文贡献点：Anthropic 只测了欧洲语言，我们测中-英-日-韩

用法:
    python run.py --model qwen3-4b --device cuda:0
    python run.py --model gemma3-4b --device cuda:1
    python run.py --model qwen3-14b --device cuda:0,1  # 多卡
"""
import argparse
import json
import sys
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    QWEN3_0_6B, QWEN3_4B, QWEN3_8B, QWEN3_14B,
    GEMMA3_4B, GEMMA3_12B, GEMMA3_27B,
    CROSS_LINGUAL_PROMPTS, CHINESE_REASONING_PROMPTS,
    RESULTS_DIR,
)
from utils import (
    load_model_and_transcoders,
    generate_attribution_graph,
    extract_feature_ids,
    compute_feature_overlap,
    compute_graph_structure_features,
    save_results,
)

MODEL_MAP = {
    "qwen3-0.6b": QWEN3_0_6B,
    "qwen3-4b": QWEN3_4B,
    "qwen3-8b": QWEN3_8B,
    "qwen3-14b": QWEN3_14B,
    "gemma3-4b": GEMMA3_4B,
    "gemma3-12b": GEMMA3_12B,
    "gemma3-27b": GEMMA3_27B,
}


def group_prompts_by_concept(prompts):
    """
    将 prompt 按概念分组。
    例如：所有语言的 "small 的反义词" 归为一组。
    """
    groups = defaultdict(list)

    for i, p in enumerate(prompts):
        # 用 target 来近似分组同一概念
        target = p.get("target", "").strip()
        lang = p.get("lang", "unknown")

        # 根据概念手动分组 (基于 target 的语义)
        # 反义词组
        if target in ["large", "大", "大きい", "크다"]:
            groups["antonym_small"].append((i, p))
        elif target in ["cold", "冷", "寒い"]:
            groups["antonym_hot"].append((i, p))
        elif target in ["slow", "慢"]:
            groups["antonym_fast"].append((i, p))
        # 首都组
        elif target in ["Paris", "巴黎", "パリ"]:
            groups["capital_france"].append((i, p))
        elif target in ["Beijing", "北京"]:
            groups["capital_china"].append((i, p))
        else:
            groups[f"other_{i}"].append((i, p))

    return dict(groups)


def compute_pairwise_overlap_matrix(graphs_by_lang, concept_groups):
    """
    计算每个概念组内，不同语言之间的 feature overlap 矩阵。

    Returns:
        dict[concept_name] -> {
            "languages": [lang1, lang2, ...],
            "jaccard_matrix": [[...], ...],
            "overlap_counts": [[...], ...],
        }
    """
    results = {}

    for concept_name, group_items in concept_groups.items():
        if len(group_items) < 2:
            continue

        langs = []
        feats = []
        for idx, prompt_info in group_items:
            lang = prompt_info.get("lang", "?")
            if idx in graphs_by_lang and graphs_by_lang[idx] is not None:
                langs.append(lang)
                feats.append(extract_feature_ids(graphs_by_lang[idx]))

        n = len(langs)
        if n < 2:
            continue

        jaccard_matrix = np.zeros((n, n))
        overlap_counts = np.zeros((n, n), dtype=int)

        for i, j in itertools.combinations(range(n), 2):
            shared = feats[i] & feats[j]
            union = feats[i] | feats[j]
            jacc = len(shared) / len(union) if union else 0.0
            jaccard_matrix[i][j] = jacc
            jaccard_matrix[j][i] = jacc
            overlap_counts[i][j] = len(shared)
            overlap_counts[j][i] = len(shared)

        np.fill_diagonal(jaccard_matrix, 1.0)
        for i in range(n):
            overlap_counts[i][i] = len(feats[i])

        results[concept_name] = {
            "languages": langs,
            "jaccard_matrix": jaccard_matrix.tolist(),
            "overlap_counts": overlap_counts.tolist(),
            "feature_counts": [len(f) for f in feats],
        }

        # 打印
        print(f"\n  概念: {concept_name}")
        print(f"    语言: {langs}")
        print(f"    Feature 数量: {[len(f) for f in feats]}")
        for i, j in itertools.combinations(range(n), 2):
            print(f"    {langs[i]} ↔ {langs[j]}: "
                  f"Jaccard={jaccard_matrix[i][j]:.4f}, "
                  f"Shared={overlap_counts[i][j]}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--include-chinese-reasoning", action="store_true",
                        help="也运行中文特色推理实验")
    args = parser.parse_args()

    cfg = MODEL_MAP[args.model]
    cfg.device = args.device
    output_dir = RESULTS_DIR / "03_cross_lingual" / cfg.name

    print("=" * 60)
    print(f"  跨语言 Feature Sharing 分析")
    print(f"  模型: {cfg.name}")
    print(f"  设备: {cfg.device}")
    print("=" * 60)

    # 加载模型
    model = load_model_and_transcoders(cfg)
    prompts = CROSS_LINGUAL_PROMPTS.prompts

    # 1. 生成所有归因图
    print("\n[1/3] 生成归因图...")
    graphs = {}  # idx -> graph
    all_metadata = []

    for i, p in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] [{p.get('lang', '?')}] {p['text']}")
        try:
            graph, meta = generate_attribution_graph(
                model, p["text"], p.get("target"), args.threshold
            )
            meta["lang"] = p.get("lang", "?")
            graphs[i] = graph
            all_metadata.append(meta)
        except Exception as e:
            print(f"    ERROR: {e}")
            graphs[i] = None
            all_metadata.append({"error": str(e), "prompt": p["text"]})

    # 2. 按概念分组，计算跨语言 overlap
    print("\n[2/3] 计算跨语言 Feature Overlap...")
    concept_groups = group_prompts_by_concept(prompts)
    overlap_results = compute_pairwise_overlap_matrix(graphs, concept_groups)

    # 3. 汇总统计
    print("\n[3/3] 汇总统计...")

    # 按语言对统计平均 Jaccard
    lang_pair_jaccards = defaultdict(list)
    for concept, data in overlap_results.items():
        langs = data["languages"]
        jm = data["jaccard_matrix"]
        for i, j in itertools.combinations(range(len(langs)), 2):
            pair = tuple(sorted([langs[i], langs[j]]))
            lang_pair_jaccards[pair].append(jm[i][j])

    summary = {}
    print("\n  语言对平均 Jaccard Similarity:")
    for pair, jaccards in sorted(lang_pair_jaccards.items()):
        mean_j = np.mean(jaccards)
        std_j = np.std(jaccards)
        pair_key = f"{pair[0]}-{pair[1]}"
        summary[pair_key] = {
            "mean_jaccard": float(mean_j),
            "std_jaccard": float(std_j),
            "n_concepts": len(jaccards),
        }
        print(f"    {pair_key}: {mean_j:.4f} ± {std_j:.4f} (n={len(jaccards)})")

    # 保存
    save_results(all_metadata, output_dir / "metadata.json")
    save_results(overlap_results, output_dir / "overlap_results.json")
    save_results(summary, output_dir / "summary.json")

    # 中文特色推理 (可选)
    if args.include_chinese_reasoning:
        print("\n\n" + "=" * 60)
        print("  中文特色推理实验")
        print("=" * 60)
        zh_prompts = CHINESE_REASONING_PROMPTS.prompts
        zh_dir = output_dir / "chinese_reasoning"

        for i, p in enumerate(zh_prompts):
            print(f"\n  [{i+1}/{len(zh_prompts)}] [{p.get('type', '?')}] {p['text']}")
            try:
                graph, meta = generate_attribution_graph(
                    model, p["text"], p.get("target"), args.threshold
                )
                meta["type"] = p.get("type", "?")
                sf = compute_graph_structure_features(graph)
                meta.update(sf)
                save_results(meta, zh_dir / f"prompt_{i:04d}.json")
            except Exception as e:
                print(f"    ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"  完成! 结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
