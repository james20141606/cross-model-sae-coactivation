"""
实验 02: 复现 Anthropic 经典 Case Study
在 Qwen3 + Gemma3 上复现：反义词、多步推理、幻觉机制

用法:
    python run.py --model qwen3-4b --experiment antonyms
    python run.py --model gemma3-4b --experiment multistep
    python run.py --model qwen3-4b --experiment hallucination
    python run.py --model qwen3-4b --experiment all
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    QWEN3_0_6B, QWEN3_1_7B, QWEN3_4B, QWEN3_8B, QWEN3_14B,
    GEMMA3_4B, GEMMA3_12B,
    ANTONYM_PROMPTS, MULTISTEP_PROMPTS, HALLUCINATION_PROMPTS,
    RESULTS_DIR,
)
from utils import (
    load_model_and_transcoders,
    batch_generate_graphs,
    compute_graph_structure_features,
    run_perturbation,
    save_results,
)

MODEL_MAP = {
    "qwen3-0.6b": QWEN3_0_6B,
    "qwen3-1.7b": QWEN3_1_7B,
    "qwen3-4b": QWEN3_4B,
    "qwen3-8b": QWEN3_8B,
    "qwen3-14b": QWEN3_14B,
    "gemma3-4b": GEMMA3_4B,
    "gemma3-12b": GEMMA3_12B,
}

EXPERIMENT_MAP = {
    "antonyms": ANTONYM_PROMPTS,
    "multistep": MULTISTEP_PROMPTS,
    "hallucination": HALLUCINATION_PROMPTS,
}


def run_experiment(model, prompt_set, output_dir, threshold=0.01):
    """运行单个实验集"""
    print(f"\n{'='*60}")
    print(f"  实验: {prompt_set.name}")
    print(f"  描述: {prompt_set.description}")
    print(f"  Prompt 数量: {len(prompt_set.prompts)}")
    print(f"  Threshold: {threshold}")
    print(f"{'='*60}")

    output_dir = Path(output_dir)

    # 1. 批量生成归因图
    print("\n[1/3] 生成归因图...")
    results = batch_generate_graphs(
        model,
        prompt_set.prompts,
        output_dir / "graphs",
        threshold=threshold,
    )

    # 2. 提取结构特征
    print("\n[2/3] 提取结构特征...")
    structure_features = []
    for graph, meta in results:
        if graph is not None:
            sf = compute_graph_structure_features(graph)
            sf.update(meta)
            structure_features.append(sf)

    save_results(structure_features, output_dir / "structure_features.json")

    # 3. Perturbation 验证 (对 top-3 nodes)
    print("\n[3/3] Perturbation 验证...")
    perturbation_results = []
    for i, (graph, meta) in enumerate(results):
        if graph is None:
            continue

        # 找到最重要的 feature nodes
        important_nodes = sorted(
            [n for n in graph.nodes if hasattr(n, "feature_id")],
            key=lambda n: abs(getattr(n, "direct_effect", 0)),
            reverse=True,
        )[:3]  # top-3

        for node in important_nodes:
            try:
                # 移除该 feature (scale=0)
                pert = run_perturbation(
                    model,
                    meta["prompt"],
                    node.feature_id,
                    scale=0.0,
                    target_token=meta.get("target_token"),
                )
                pert["prompt_idx"] = i
                pert["feature_id"] = str(node.feature_id)
                pert["feature_label"] = getattr(node, "label", "")
                pert["original_effect"] = getattr(node, "direct_effect", 0)
                perturbation_results.append(pert)
            except Exception as e:
                print(f"    Perturbation error: {e}")

    save_results(perturbation_results, output_dir / "perturbation_results.json")

    print(f"\n  实验 {prompt_set.name} 完成！")
    print(f"  结果保存在: {output_dir}")

    return results, structure_features, perturbation_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--experiment", default="all",
                        choices=list(EXPERIMENT_MAP.keys()) + ["all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="attribution threshold; 0.01 default, 0.001 for harder budget")
    args = parser.parse_args()

    cfg = MODEL_MAP[args.model]
    cfg.device = args.device

    # 加载模型
    model = load_model_and_transcoders(cfg)

    # 运行实验
    experiments = (
        list(EXPERIMENT_MAP.keys()) if args.experiment == "all"
        else [args.experiment]
    )

    for exp_name in experiments:
        prompt_set = EXPERIMENT_MAP[exp_name]
        thr_tag = f"_thr{args.threshold:g}" if args.threshold != 0.01 else ""
        output_dir = RESULTS_DIR / "02_reproduce_anthropic" / cfg.name / f"{exp_name}{thr_tag}"
        run_experiment(model, prompt_set, output_dir, threshold=args.threshold)

    print("\n" + "=" * 60)
    print("  所有实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
