"""
实验 05: 跨模型族 Circuit Universality 分析
核心论文贡献：Qwen3 vs Gemma3 的电路对比 + Scaling 分析

用法:
    # 同规模对比
    python run.py --mode matched_pairs --device cuda:0

    # Qwen3 scaling
    python run.py --mode qwen_scaling --device cuda:0

    # Gemma3 scaling
    python run.py --mode gemma_scaling --device cuda:0

    # 全量 (所有模型、所有实验)
    python run.py --mode full --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    QWEN3_SCALING, GEMMA3_SCALING, MATCHED_PAIRS, DEV_MODELS,
    ANTONYM_PROMPTS, MULTISTEP_PROMPTS, CROSS_LINGUAL_PROMPTS,
    RESULTS_DIR,
)
from utils import (
    load_model_and_transcoders,
    generate_attribution_graph,
    extract_feature_ids,
    compute_graph_structure_features,
    save_results,
)


# 用于跨模型对比的 prompt (需要所有模型都能理解)
UNIVERSAL_PROMPTS = [
    {"text": "The opposite of small is", "target": " large", "type": "antonym"},
    {"text": "The opposite of hot is", "target": " cold", "type": "antonym"},
    {"text": "The capital of France is", "target": " Paris", "type": "factual"},
    {"text": "The capital of China is", "target": " Beijing", "type": "factual"},
    {"text": "2 + 3 =", "target": " 5", "type": "arithmetic"},
    {"text": "The CEO of Apple is", "target": " Tim", "type": "entity"},
]


def run_scaling_analysis(model_configs, prompts, output_dir, device="cuda:0"):
    """
    对一系列模型 (从小到大) 运行相同 prompt，分析电路随 scale 的变化。

    Returns:
        scaling_data: list of dicts，每个包含模型信息 + 所有 prompt 的结构特征
    """
    output_dir = Path(output_dir)
    scaling_data = []

    for cfg in model_configs:
        cfg.device = device
        print(f"\n{'='*60}")
        print(f"  模型: {cfg.name} ({cfg.params_b}B)")
        print(f"{'='*60}")

        model = load_model_and_transcoders(cfg)

        model_results = {
            "model": cfg.name,
            "family": cfg.family,
            "params_b": cfg.params_b,
            "num_layers": cfg.num_layers,
            "prompt_results": [],
        }

        for i, p in enumerate(prompts):
            print(f"\n  [{i+1}/{len(prompts)}] {p['text']}")
            try:
                graph, meta = generate_attribution_graph(
                    model, p["text"], p.get("target"), threshold=0.01
                )
                sf = compute_graph_structure_features(graph)
                sf["prompt_text"] = p["text"]
                sf["prompt_type"] = p.get("type", "unknown")
                sf["target"] = p.get("target", "")
                sf["elapsed_seconds"] = meta.get("elapsed_seconds", 0)
                model_results["prompt_results"].append(sf)
            except Exception as e:
                print(f"    ERROR: {e}")
                model_results["prompt_results"].append({
                    "error": str(e), "prompt_text": p["text"]
                })

        scaling_data.append(model_results)

        # 释放 GPU 内存
        del model
        import torch
        torch.cuda.empty_cache()

    # 汇总 scaling trends
    print(f"\n\n{'='*60}")
    print("  Scaling 趋势汇总")
    print(f"{'='*60}")

    for prompt_type in set(p.get("type", "?") for p in prompts):
        print(f"\n  --- {prompt_type} ---")
        print(f"  {'Model':<20} {'Params':>8} {'Nodes':>8} {'Edges':>8} {'Density':>10} {'LayerSpan':>10}")
        for sd in scaling_data:
            type_results = [
                r for r in sd["prompt_results"]
                if r.get("prompt_type") == prompt_type and "error" not in r
            ]
            if type_results:
                avg_nodes = np.mean([r["n_nodes"] for r in type_results])
                avg_edges = np.mean([r["n_edges"] for r in type_results])
                avg_density = np.mean([r["density"] for r in type_results])
                avg_span = np.mean([r["layer_span"] for r in type_results])
                print(f"  {sd['model']:<20} {sd['params_b']:>7.1f}B {avg_nodes:>8.1f} "
                      f"{avg_edges:>8.1f} {avg_density:>10.6f} {avg_span:>10.1f}")

    save_results(scaling_data, output_dir / "scaling_data.json")
    return scaling_data


def run_matched_pair_comparison(pairs, prompts, output_dir, device="cuda:0"):
    """
    对同规模的 Qwen vs Gemma 做 head-to-head 对比。
    """
    output_dir = Path(output_dir)
    pair_results = []

    for qwen_cfg, gemma_cfg in pairs:
        print(f"\n{'='*60}")
        print(f"  对比: {qwen_cfg.name} vs {gemma_cfg.name}")
        print(f"{'='*60}")

        comparison = {
            "qwen_model": qwen_cfg.name,
            "gemma_model": gemma_cfg.name,
            "qwen_params_b": qwen_cfg.params_b,
            "gemma_params_b": gemma_cfg.params_b,
            "prompts": [],
        }

        for model_cfg, model_label in [(qwen_cfg, "qwen"), (gemma_cfg, "gemma")]:
            model_cfg.device = device
            model = load_model_and_transcoders(model_cfg)

            for i, p in enumerate(prompts):
                print(f"\n  [{model_label}] [{i+1}/{len(prompts)}] {p['text']}")
                try:
                    graph, meta = generate_attribution_graph(
                        model, p["text"], p.get("target"), threshold=0.01
                    )
                    sf = compute_graph_structure_features(graph)
                    sf["model"] = model_label
                    sf["prompt_text"] = p["text"]
                    sf["prompt_type"] = p.get("type", "unknown")
                    comparison["prompts"].append(sf)
                except Exception as e:
                    print(f"    ERROR: {e}")

            del model
            import torch
            torch.cuda.empty_cache()

        # 对比统计
        qwen_results = [r for r in comparison["prompts"] if r["model"] == "qwen"]
        gemma_results = [r for r in comparison["prompts"] if r["model"] == "gemma"]

        if qwen_results and gemma_results:
            print(f"\n  对比汇总:")
            print(f"    {'Metric':<25} {'Qwen':>12} {'Gemma':>12}")
            for metric in ["n_nodes", "n_edges", "density", "layer_span", "weight_mean"]:
                qwen_avg = np.mean([r.get(metric, 0) for r in qwen_results])
                gemma_avg = np.mean([r.get(metric, 0) for r in gemma_results])
                print(f"    {metric:<25} {qwen_avg:>12.4f} {gemma_avg:>12.4f}")

        pair_results.append(comparison)

    save_results(pair_results, output_dir / "matched_pair_results.json")
    return pair_results


def _filter_large(configs, skip_large):
    """首轮砍 14B / 27B 节省 GPU 时间"""
    if not skip_large:
        return configs
    return [c for c in configs if c.params_b < 13.0]


def _filter_pairs_large(pairs, skip_large):
    if not skip_large:
        return pairs
    return [(q, g) for q, g in pairs if q.params_b < 13.0 and g.params_b < 13.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["qwen_scaling", "gemma_scaling", "matched_pairs", "full", "dev"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-large", action="store_true",
                        help="跳过 Qwen3-14B 和 Gemma3-27B (首轮节省 GPU)")
    args = parser.parse_args()

    base_dir = RESULTS_DIR / "05_cross_model"

    qwen_set = _filter_large(QWEN3_SCALING, args.skip_large)
    gemma_set = _filter_large(GEMMA3_SCALING, args.skip_large)
    pair_set = _filter_pairs_large(MATCHED_PAIRS, args.skip_large)

    if args.mode == "dev":
        print("DEV MODE: 用最小模型快速验证 pipeline")
        run_scaling_analysis(
            DEV_MODELS, UNIVERSAL_PROMPTS[:2],
            base_dir / "dev", args.device,
        )

    elif args.mode == "qwen_scaling":
        run_scaling_analysis(
            qwen_set, UNIVERSAL_PROMPTS,
            base_dir / "qwen_scaling", args.device,
        )

    elif args.mode == "gemma_scaling":
        run_scaling_analysis(
            gemma_set, UNIVERSAL_PROMPTS,
            base_dir / "gemma_scaling", args.device,
        )

    elif args.mode == "matched_pairs":
        run_matched_pair_comparison(
            pair_set, UNIVERSAL_PROMPTS,
            base_dir / "matched_pairs", args.device,
        )

    elif args.mode == "full":
        print("FULL MODE: 运行所有分析")
        run_scaling_analysis(
            qwen_set, UNIVERSAL_PROMPTS,
            base_dir / "qwen_scaling", args.device,
        )
        run_scaling_analysis(
            gemma_set, UNIVERSAL_PROMPTS,
            base_dir / "gemma_scaling", args.device,
        )
        run_matched_pair_comparison(
            pair_set, UNIVERSAL_PROMPTS,
            base_dir / "matched_pairs", args.device,
        )

    print("\n  所有分析完成!")


if __name__ == "__main__":
    main()
