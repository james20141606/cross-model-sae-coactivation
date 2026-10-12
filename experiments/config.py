"""
全局配置 — 模型路径、Transcoder 路径、实验参数
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# 路径配置 (根据你的集群修改)
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TRANSCODERS_DIR = PROJECT_ROOT / "transcoders"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# 模型配置
# ============================================================

@dataclass
class ModelConfig:
    """单个模型 + transcoder 的配置"""
    name: str                    # 短名称，用于结果文件命名
    model_name: str              # HuggingFace model ID
    transcoder_name: str         # HuggingFace transcoder ID
    transcoder_type: str         # "plt" or "clt"
    family: str                  # "qwen3" or "gemma3"
    params_b: float              # 参数量 (billion)
    num_layers: int              # transformer 层数
    backend: str = "nnsight"     # "transformerlens" or "nnsight"
    dtype: str = "float16"       # 推理精度
    device: str = "cuda:0"       # 默认设备
    cache_dir: Optional[str] = None


# --- Qwen3 系列 (PLT) ---
QWEN3_0_6B = ModelConfig(
    name="qwen3-0.6b",
    model_name="Qwen/Qwen3-0.6B",
    transcoder_name="mwhanna/qwen3-0.6b-transcoders-lowl0",
    transcoder_type="plt",
    family="qwen3",
    params_b=0.6,
    num_layers=28,
)

QWEN3_1_7B = ModelConfig(
    name="qwen3-1.7b",
    model_name="Qwen/Qwen3-1.7B",
    transcoder_name="mwhanna/qwen3-1.7b-transcoders-lowl0",
    transcoder_type="plt",
    family="qwen3",
    params_b=1.7,
    num_layers=28,
)

QWEN3_4B = ModelConfig(
    name="qwen3-4b",
    model_name="Qwen/Qwen3-4B",
    transcoder_name="mwhanna/qwen3-4b-transcoders",
    transcoder_type="plt",
    family="qwen3",
    params_b=4.0,
    num_layers=36,
)

QWEN3_8B = ModelConfig(
    name="qwen3-8b",
    model_name="Qwen/Qwen3-8B",
    transcoder_name="mwhanna/qwen3-8b-transcoders",
    transcoder_type="plt",
    family="qwen3",
    params_b=8.0,
    num_layers=36,
)

QWEN3_14B = ModelConfig(
    name="qwen3-14b",
    model_name="Qwen/Qwen3-14B",
    transcoder_name="mwhanna/qwen3-14b-transcoders",
    transcoder_type="plt",
    family="qwen3",
    params_b=14.0,
    num_layers=40,
)

# --- Qwen3 CLT (BluelightAI, 仅小模型) ---
QWEN3_0_6B_CLT = ModelConfig(
    name="qwen3-0.6b-clt",
    model_name="Qwen/Qwen3-0.6B-Base",
    transcoder_name="bluelightai/qwen3-0.6b-base-clt",
    transcoder_type="clt",
    family="qwen3",
    params_b=0.6,
    num_layers=28,
)

QWEN3_1_7B_CLT = ModelConfig(
    name="qwen3-1.7b-clt",
    model_name="Qwen/Qwen3-1.7B-Base",
    transcoder_name="bluelightai/qwen3-1.7b-base-clt",
    transcoder_type="clt",
    family="qwen3",
    params_b=1.7,
    num_layers=28,
)

# --- Gemma3 系列 (Gemma Scope 2) ---
GEMMA3_4B = ModelConfig(
    name="gemma3-4b",
    model_name="google/gemma-3-4b-it",
    transcoder_name="google/gemma-scope-2-4b-it",
    transcoder_type="plt",  # gemma scope 2 有 plt + sae + clt
    family="gemma3",
    params_b=4.0,
    num_layers=34,
)

GEMMA3_12B = ModelConfig(
    name="gemma3-12b",
    model_name="google/gemma-3-12b-it",
    transcoder_name="google/gemma-scope-2-12b-it",
    transcoder_type="plt",
    family="gemma3",
    params_b=12.0,
    num_layers=48,
)

GEMMA3_27B = ModelConfig(
    name="gemma3-27b",
    model_name="google/gemma-3-27b-it",
    transcoder_name="google/gemma-scope-2-27b-it",
    transcoder_type="plt",
    family="gemma3",
    params_b=27.0,
    num_layers=62,
)

# ============================================================
# 预定义模型组
# ============================================================

# Qwen3 全系列 scaling (PLT)
QWEN3_SCALING = [QWEN3_0_6B, QWEN3_1_7B, QWEN3_4B, QWEN3_8B, QWEN3_14B]

# Gemma3 scaling
GEMMA3_SCALING = [GEMMA3_4B, GEMMA3_12B, GEMMA3_27B]

# 同规模对比 (Qwen vs Gemma)
MATCHED_PAIRS = [
    (QWEN3_4B, GEMMA3_4B),       # ~4B 对比
    (QWEN3_14B, GEMMA3_12B),     # ~13B 对比 (最接近的尺寸)
]

# 快速开发用 (小模型)
DEV_MODELS = [QWEN3_0_6B, QWEN3_4B]

# ============================================================
# 实验 Prompt 集合
# ============================================================

@dataclass
class PromptSet:
    """一组实验 prompt"""
    name: str
    prompts: list
    description: str = ""


# --- 复现 Anthropic 经典 case ---
ANTONYM_PROMPTS = PromptSet(
    name="antonyms",
    description="反义词电路 (Anthropic Biology §1)",
    prompts=[
        # 英文
        {"text": "The opposite of small is", "target": " large", "lang": "en"},
        {"text": "The opposite of hot is", "target": " cold", "lang": "en"},
        {"text": "The opposite of fast is", "target": " slow", "lang": "en"},
        {"text": "The opposite of happy is", "target": " sad", "lang": "en"},
        {"text": "The opposite of light is", "target": " dark", "lang": "en"},
        {"text": "The opposite of old is", "target": " young", "lang": "en"},
        {"text": "The opposite of tall is", "target": " short", "lang": "en"},
        {"text": "The opposite of rich is", "target": " poor", "lang": "en"},
    ],
)

MULTISTEP_PROMPTS = PromptSet(
    name="multistep_reasoning",
    description="多步推理电路 (Anthropic Biology §2)",
    prompts=[
        {"text": "The capital of the country where the Eiffel Tower is located is", "target": " Paris", "lang": "en"},
        {"text": "The currency used in the country where Tokyo is the capital is the", "target": " yen", "lang": "en"},
        {"text": "The language spoken in the country where the Great Wall is located is", "target": " Chinese", "lang": "en"},
        {"text": "The capital of the country that borders Canada to the south is", "target": " Washington", "lang": "en"},
    ],
)

HALLUCINATION_PROMPTS = PromptSet(
    name="hallucination",
    description="幻觉机制 — known entity vs default refusal (Anthropic Biology §4)",
    prompts=[
        # 已知实体 (应该回答)
        {"text": "The CEO of Apple is", "target": " Tim", "known": True},
        {"text": "Albert Einstein was born in", "target": " Ulm", "known": True},
        # 未知实体 (应该拒绝或犹豫)
        {"text": "The CEO of Zyphrex Corp is", "target": None, "known": False},
        {"text": "Professor Xanthos Meridian discovered", "target": None, "known": False},
    ],
)

# --- Qwen 特有实验: 中英跨语言 ---
CROSS_LINGUAL_PROMPTS = PromptSet(
    name="cross_lingual",
    description="跨语言 feature sharing — Anthropic 只测了欧洲语言，我们测中英日",
    prompts=[
        # 反义词 — 同一概念，不同语言
        {"text": "The opposite of small is", "target": " large", "lang": "en"},
        {"text": "小的反义词是", "target": "大", "lang": "zh"},
        {"text": "小さいの反対は", "target": "大きい", "lang": "ja"},
        {"text": "작다의 반대말은", "target": "크다", "lang": "ko"},

        {"text": "The opposite of hot is", "target": " cold", "lang": "en"},
        {"text": "热的反义词是", "target": "冷", "lang": "zh"},
        {"text": "暑いの反対は", "target": "寒い", "lang": "ja"},

        {"text": "The opposite of fast is", "target": " slow", "lang": "en"},
        {"text": "快的反义词是", "target": "慢", "lang": "zh"},

        # 多步推理 — 同一知识，不同语言
        {"text": "The capital of France is", "target": " Paris", "lang": "en"},
        {"text": "法国的首都是", "target": "巴黎", "lang": "zh"},
        {"text": "フランスの首都は", "target": "パリ", "lang": "ja"},

        {"text": "The capital of China is", "target": " Beijing", "lang": "en"},
        {"text": "中国的首都是", "target": "北京", "lang": "zh"},
    ],
)

# --- Qwen 特有实验: 思考模式 ---
THINKING_MODE_PROMPTS = PromptSet(
    name="thinking_mode",
    description="Qwen3 thinking mode on/off 电路对比",
    prompts=[
        # 数学推理 (thinking 应该有显著差异)
        {"text": "What is 17 * 23?", "target": " 391",
         "thinking_prompt": "<|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n<think>\n"},
        {"text": "What is the square root of 144?", "target": " 12",
         "thinking_prompt": "<|im_start|>user\nWhat is the square root of 144?<|im_end|>\n<|im_start|>assistant\n<think>\n"},

        # 简单事实 (thinking 应该差异较小)
        {"text": "The capital of Japan is", "target": " Tokyo",
         "thinking_prompt": "<|im_start|>user\nWhat is the capital of Japan?<|im_end|>\n<|im_start|>assistant\n<think>\n"},
    ],
)

# --- 中文特色推理 ---
CHINESE_REASONING_PROMPTS = PromptSet(
    name="chinese_reasoning",
    description="中文特色推理 — 成语、歇后语等，测试 Qwen 的中文电路",
    prompts=[
        # 成语补全
        {"text": "画蛇添", "target": "足", "type": "idiom"},
        {"text": "守株待", "target": "兔", "type": "idiom"},
        {"text": "亡羊补", "target": "牢", "type": "idiom"},
        {"text": "刻舟求", "target": "剑", "type": "idiom"},

        # 歇后语
        {"text": "黄鼠狼给鸡拜年——", "target": "没安好心", "type": "xiehouyu"},
        {"text": "竹篮打水——", "target": "一场空", "type": "xiehouyu"},

        # 古诗补全
        {"text": "床前明月光，疑是地上", "target": "霜", "type": "poetry"},
        {"text": "春眠不觉晓，处处闻", "target": "啼鸟", "type": "poetry"},
    ],
)

# ============================================================
# 所有 prompt 集合
# ============================================================
ALL_PROMPT_SETS = {
    "antonyms": ANTONYM_PROMPTS,
    "multistep": MULTISTEP_PROMPTS,
    "hallucination": HALLUCINATION_PROMPTS,
    "cross_lingual": CROSS_LINGUAL_PROMPTS,
    "thinking_mode": THINKING_MODE_PROMPTS,
    "chinese_reasoning": CHINESE_REASONING_PROMPTS,
}
