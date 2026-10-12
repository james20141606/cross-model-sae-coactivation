"""
实验 04: Qwen3 Thinking Mode 电路对比
比较 think / no_think 模式下的归因图差异

用法:
    python run.py --model qwen3-4b --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    QWEN3_4B, QWEN3_8B, QWEN3_14B,
    THINKING_MODE_PROMPTS,
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
    "qwen3-4b": QWEN3_4B,
    "qwen3-8b": QWEN3_8B,
    "qwen3-14b": QWEN3_14B,
}


def format_chat_prompt(tokenizer, user_msg, enable_thinking=True):
    """
    构造 Qwen3 chat 格式的 prompt。

    Args:
        tokenizer: Qwen3 tokenizer
        user_msg: 用户消息
        enable_thinking: 是否启用 thinking mode

    Returns:
        formatted prompt string
    """
    messages = [{"role": "user", "content": user_msg}]

    # Qwen3 thinking mode 通过 enable_thinking 参数控制
    # 或者通过在 system prompt 中添加 /think 或 /no_think
    if enable_thinking:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    return prompt


def compare_thinking_modes(model, user_msg, target_token, output_dir, threshold=0.01):
    """
    对比同一 prompt 在 think/no_think 模式下的归因图。

    Returns:
        dict with comparison results
    """
    tokenizer = model.tokenizer

    results = {}

    for mode_name, enable_thinking in [("think", True), ("no_think", False)]:
        print(f"\n  Mode: {mode_name}")

        # 构造 prompt
        try:
            prompt = format_chat_prompt(tokenizer, user_msg, enable_thinking)
        except TypeError:
            # 如果 tokenizer 不支持 enable_thinking 参数，手动构造
            if enable_thinking:
                prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            else:
                prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

        print(f"    Prompt length: {len(prompt)} chars")

        # 生成归因图
        try:
            graph, meta = generate_attribution_graph(
                model, prompt, target_token, threshold
            )
            sf = compute_graph_structure_features(graph)
            meta.update(sf)
            meta["mode"] = mode_name
            meta["enable_thinking"] = enable_thinking

            results[mode_name] = {
                "graph": graph,
                "metadata": meta,
                "features": extract_feature_ids(graph),
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            results[mode_name] = {"error": str(e)}

    # 对比分析
    if "graph" in results.get("think", {}) and "graph" in results.get("no_think", {}):
        overlap = compute_feature_overlap(
            results["think"]["graph"],
            results["no_think"]["graph"],
        )

        comparison = {
            "user_msg": user_msg,
            "target_token": target_token,
            "think_nodes": results["think"]["metadata"]["n_nodes"],
            "think_edges": results["think"]["metadata"]["n_edges"],
            "no_think_nodes": results["no_think"]["metadata"]["n_nodes"],
            "no_think_edges": results["no_think"]["metadata"]["n_edges"],
            "feature_overlap_jaccard": overlap["jaccard"],
            "shared_features": overlap["overlap_count"],
            "think_only_features": overlap["total_a"] - overlap["overlap_count"],
            "no_think_only_features": overlap["total_b"] - overlap["overlap_count"],
            "think_structure": {
                k: v for k, v in results["think"]["metadata"].items()
                if k.startswith(("n_", "density", "layer_", "weight_", "in_degree", "out_degree"))
            },
            "no_think_structure": {
                k: v for k, v in results["no_think"]["metadata"].items()
                if k.startswith(("n_", "density", "layer_", "weight_", "in_degree", "out_degree"))
            },
        }

        print(f"\n  对比结果:")
        print(f"    Think:    {comparison['think_nodes']} nodes, {comparison['think_edges']} edges")
        print(f"    No-Think: {comparison['no_think_nodes']} nodes, {comparison['no_think_edges']} edges")
        print(f"    Feature Overlap (Jaccard): {comparison['feature_overlap_jaccard']:.4f}")
        print(f"    Shared / Think-only / NoThink-only: "
              f"{comparison['shared_features']} / "
              f"{comparison['think_only_features']} / "
              f"{comparison['no_think_only_features']}")

        return comparison
    else:
        return {"error": "One or both modes failed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    cfg = MODEL_MAP[args.model]
    cfg.device = args.device
    output_dir = RESULTS_DIR / "04_thinking_mode" / cfg.name

    print("=" * 60)
    print(f"  Thinking Mode 电路对比")
    print(f"  模型: {cfg.name}")
    print("=" * 60)

    model = load_model_and_transcoders(cfg)

    # 运行所有 prompt
    all_comparisons = []
    prompts = THINKING_MODE_PROMPTS.prompts

    for i, p in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(prompts)}] {p['text']}")

        comparison = compare_thinking_modes(
            model, p["text"], p.get("target"),
            output_dir / f"prompt_{i:04d}",
            args.threshold,
        )
        comparison["prompt_idx"] = i
        all_comparisons.append(comparison)

    # 汇总
    valid = [c for c in all_comparisons if "error" not in c]
    if valid:
        avg_jaccard = np.mean([c["feature_overlap_jaccard"] for c in valid])
        avg_think_nodes = np.mean([c["think_nodes"] for c in valid])
        avg_nothink_nodes = np.mean([c["no_think_nodes"] for c in valid])

        summary = {
            "model": cfg.name,
            "n_prompts": len(prompts),
            "n_valid": len(valid),
            "avg_feature_overlap_jaccard": float(avg_jaccard),
            "avg_think_nodes": float(avg_think_nodes),
            "avg_no_think_nodes": float(avg_nothink_nodes),
            "comparisons": all_comparisons,
        }

        print(f"\n\n{'='*60}")
        print(f"  汇总 ({cfg.name})")
        print(f"    平均 Feature Overlap (Jaccard): {avg_jaccard:.4f}")
        print(f"    平均 Think 节点数: {avg_think_nodes:.1f}")
        print(f"    平均 No-Think 节点数: {avg_nothink_nodes:.1f}")
    else:
        summary = {"error": "No valid comparisons", "comparisons": all_comparisons}

    save_results(summary, output_dir / "summary.json")
    print(f"\n  结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
