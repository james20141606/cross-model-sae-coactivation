"""Expanded antonym prompts: n=20 per language (en/zh/ja/ko).

20 antonym pairs aligned across the 4 languages, plus 5 capital-city pairs
in same alignment. Total: 25 prompts × 4 langs = 100 prompts; 6 pairs ⇒
6 × 25 paired correlations per layer.
"""
import json

PAIRS = [
    # antonyms (20)
    {"en": ("The opposite of hot is", "cold"), "zh": ("热的反义词是", "冷"),
     "ja": ("熱いの反対は", "冷たい"), "ko": ("뜨거운의 반대는", "차가운")},
    {"en": ("The opposite of big is", "small"), "zh": ("大的反义词是", "小"),
     "ja": ("大きいの反対は", "小さい"), "ko": ("큰의 반대는", "작은")},
    {"en": ("The opposite of fast is", "slow"), "zh": ("快的反义词是", "慢"),
     "ja": ("速いの反対は", "遅い"), "ko": ("빠른의 반대는", "느린")},
    {"en": ("The opposite of dark is", "light"), "zh": ("黑暗的反义词是", "光明"),
     "ja": ("暗いの反対は", "明るい"), "ko": ("어두운의 반대는", "밝은")},
    {"en": ("The opposite of high is", "low"), "zh": ("高的反义词是", "低"),
     "ja": ("高いの反対は", "低い"), "ko": ("높은의 반대는", "낮은")},
    {"en": ("The opposite of wet is", "dry"), "zh": ("湿的反义词是", "干"),
     "ja": ("濡れたの反対は", "乾いた"), "ko": ("젖은의 반대는", "마른")},
    {"en": ("The opposite of long is", "short"), "zh": ("长的反义词是", "短"),
     "ja": ("長いの反対は", "短い"), "ko": ("긴의 반대는", "짧은")},
    {"en": ("The opposite of strong is", "weak"), "zh": ("强的反义词是", "弱"),
     "ja": ("強いの反対は", "弱い"), "ko": ("강한의 반대는", "약한")},
    {"en": ("The opposite of happy is", "sad"), "zh": ("快乐的反义词是", "悲伤"),
     "ja": ("嬉しいの反対は", "悲しい"), "ko": ("기쁜의 반대는", "슬픈")},
    {"en": ("The opposite of full is", "empty"), "zh": ("满的反义词是", "空"),
     "ja": ("満たされたの反対は", "空"), "ko": ("가득한의 반대는", "빈")},
    {"en": ("The opposite of heavy is", "light"), "zh": ("重的反义词是", "轻"),
     "ja": ("重いの反対は", "軽い"), "ko": ("무거운의 반대는", "가벼운")},
    {"en": ("The opposite of clean is", "dirty"), "zh": ("干净的反义词是", "脏"),
     "ja": ("綺麗の反対は", "汚い"), "ko": ("깨끗한의 반대는", "더러운")},
    {"en": ("The opposite of rich is", "poor"), "zh": ("富有的反义词是", "贫穷"),
     "ja": ("豊かの反対は", "貧しい"), "ko": ("부유한의 반대는", "가난한")},
    {"en": ("The opposite of young is", "old"), "zh": ("年轻的反义词是", "年老"),
     "ja": ("若いの反対は", "年老いた"), "ko": ("젊은의 반대는", "늙은")},
    {"en": ("The opposite of warm is", "cool"), "zh": ("温暖的反义词是", "凉爽"),
     "ja": ("暖かいの反対は", "涼しい"), "ko": ("따뜻한의 반대는", "시원한")},
    {"en": ("The opposite of hard is", "soft"), "zh": ("硬的反义词是", "软"),
     "ja": ("硬いの反対は", "柔らかい"), "ko": ("딱딱한의 반대는", "부드러운")},
    {"en": ("The opposite of new is", "old"), "zh": ("新的反义词是", "旧"),
     "ja": ("新しいの反対は", "古い"), "ko": ("새로운의 반대는", "오래된")},
    {"en": ("The opposite of open is", "closed"), "zh": ("打开的反义词是", "关闭"),
     "ja": ("開いたの反対は", "閉じた"), "ko": ("열린의 반대는", "닫힌")},
    {"en": ("The opposite of true is", "false"), "zh": ("真的反义词是", "假"),
     "ja": ("本当の反対は", "嘘"), "ko": ("진실의 반대는", "거짓")},
    {"en": ("The opposite of love is", "hate"), "zh": ("爱的反义词是", "恨"),
     "ja": ("愛の反対は", "憎しみ"), "ko": ("사랑의 반대는", "미움")},
]

OUT_PATH = "<REPO_ROOT>/data/prompts_expanded_v2.jsonl"
import os
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for i, pair in enumerate(PAIRS):
        for lang in ["en", "zh", "ja", "ko"]:
            prompt, target = pair[lang]
            f.write(json.dumps({"pair_id": i, "lang": lang, "prompt": prompt, "target": target}, ensure_ascii=False) + "\n")
print(f"wrote {len(PAIRS) * 4} prompts to {OUT_PATH}")
