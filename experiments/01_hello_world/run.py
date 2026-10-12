"""
实验 01: Hello World — 验证环境 + 生成第一个归因图
用最小模型 (Qwen3-0.6B) 快速验证整个 pipeline。

用法:
    python run.py                    # 默认 Qwen3-0.6B
    python run.py --model qwen3-4b   # 指定模型
    python run.py --device cuda:1    # 指定 GPU
"""
import argparse
import sys
from pathlib import Path

# 添加 experiments 到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import QWEN3_0_6B, QWEN3_4B, GEMMA3_4B, RESULTS_DIR
from utils import load_model_and_transcoders, generate_attribution_graph, save_results


MODEL_MAP = {
    "qwen3-0.6b": QWEN3_0_6B,
    "qwen3-4b": QWEN3_4B,
    "gemma3-4b": GEMMA3_4B,
}


def main():
    parser = argparse.ArgumentParser(description="Hello World: 生成第一个归因图")
    parser.add_argument("--model", default="qwen3-0.6b", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="The opposite of small is")
    parser.add_argument("--target", default=None, help="Target token (default: model's top-1)")
    args = parser.parse_args()

    # 配置
    cfg = MODEL_MAP[args.model]
    cfg.device = args.device
    output_dir = RESULTS_DIR / "01_hello_world" / cfg.name

    print("=" * 60)
    print(f"  Hello World: {cfg.name}")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Device: {cfg.device}")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/3] 加载模型 + transcoder...")
    model = load_model_and_transcoders(cfg)

    # 2. 验证模型基础推理
    print("\n[2/3] 验证模型推理...")
    tokenizer = model.tokenizer
    output = model.run(args.prompt)
    logits = output.logits[0, -1]
    top5 = logits.topk(5)
    print("  Top-5 predictions:")
    for idx, score in zip(top5.indices, top5.values):
        token = tokenizer.decode([idx.item()])
        print(f"    '{token}': {score.item():.3f}")

    # 3. 生成归因图
    print("\n[3/3] 生成归因图...")
    graph, metadata = generate_attribution_graph(
        model, args.prompt, args.target, threshold=0.01
    )

    # 打印图信息
    print(f"\n  结果:")
    print(f"    节点数: {metadata['num_nodes']}")
    print(f"    边数: {metadata['num_edges']}")
    print(f"    目标 token: '{metadata['target_token']}'")
    print(f"    耗时: {metadata['elapsed_seconds']:.1f}s")

    # 打印重要节点
    print(f"\n  Top-10 nodes (by absolute contribution):")
    nodes_sorted = sorted(
        graph.nodes,
        key=lambda n: abs(getattr(n, "direct_effect", 0)),
        reverse=True,
    )
    for node in nodes_sorted[:10]:
        effect = getattr(node, "direct_effect", 0)
        label = getattr(node, "label", str(node))
        layer = getattr(node, "layer", "?")
        print(f"    Layer {layer:>3} | effect={effect:+.4f} | {label}")

    # 保存
    save_results(metadata, output_dir / "metadata.json")
    print(f"\n  已保存到: {output_dir}")

    print("\n" + "=" * 60)
    print("  Hello World 完成! 环境验证通过。")
    print("=" * 60)


if __name__ == "__main__":
    main()
