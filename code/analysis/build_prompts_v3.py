#!/usr/bin/env python3
"""Build the n>=200 multilingual concept-pair prompt set for the construct-validity scale-up.

Extends the original 20 antonym pairs (data/prompts_expanded_v2.jsonl) to >=200 concept
pairs across FIVE concept families, each parallel in en/zh/ja/ko:

  1. antonyms        (lexical-semantic)   - the original 20, reused verbatim.
  2. capitals        (factual recall)     - country -> capital, 38 curated countries.
  3. arithmetic      (numeric operation)  - addition + subtraction over number words.
  4. number-sequence (numeric ordering)   - successor / predecessor / doubling.
  5. day/month       (temporal sequence)  - weekday-after, month-after.

Targets are taken to be the natural single answer; the logit-lens reads the FIRST target
token (matching the original generator), so multi-token targets are fine and treated
identically under the real vs mismatched-pair controls.

Translation policy / honesty note: only families with UNAMBIGUOUS, AUDITABLE 4-language
targets are included (number words, capital transliterations, weekday/month names,
the native-authored antonyms). Any residual grammatical imperfection in zh/ja/ko adds
NOISE symmetrically to the real and mismatched-pair conditions, so it can only REDUCE
the measured semantic residual -- it cannot manufacture a positive separation. The test
is therefore conservative.

Output: data/prompts_v3_multi.jsonl  (one row per (pair_id, lang); 4 rows per pair).
"""
import json
from pathlib import Path

LANGS = ["en", "zh", "ja", "ko"]


def ko_topic(word):
    """Append the Korean topic particle 은/는 with correct 받침 agreement."""
    last = word[-1]
    if "가" <= last <= "힣":          # Hangul syllable
        has_final = (ord(last) - 0xAC00) % 28 != 0
        return word + ("은" if has_final else "는")
    return word + "는"

# ---- number words 0..20 -------------------------------------------------------
NUM = {
    "en": ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
           "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
           "eighteen","nineteen","twenty"],
    "zh": ["零","一","二","三","四","五","六","七","八","九","十",
           "十一","十二","十三","十四","十五","十六","十七","十八","十九","二十"],
    "ja": ["ゼロ","一","二","三","四","五","六","七","八","九","十",
           "十一","十二","十三","十四","十五","十六","十七","十八","十九","二十"],
    "ko": ["영","일","이","삼","사","오","육","칠","팔","구","십",
           "십일","십이","십삼","십사","십오","십육","십칠","십팔","십구","이십"],
}

# ---- capitals: country (4 langs) -> capital (4 langs) -------------------------
CAPITALS = [
    # (en_country, zh, ja, ko), (en_cap, zh, ja, ko)
    (("France","法国","フランス","프랑스"),       ("Paris","巴黎","パリ","파리")),
    (("Japan","日本","日本","일본"),              ("Tokyo","东京","東京","도쿄")),
    (("China","中国","中国","중국"),              ("Beijing","北京","北京","베이징")),
    (("Korea","韩国","韓国","한국"),              ("Seoul","首尔","ソウル","서울")),
    (("Germany","德国","ドイツ","독일"),          ("Berlin","柏林","ベルリン","베를린")),
    (("Italy","意大利","イタリア","이탈리아"),     ("Rome","罗马","ローマ","로마")),
    (("Spain","西班牙","スペイン","스페인"),       ("Madrid","马德里","マドリード","마드리드")),
    (("Russia","俄罗斯","ロシア","러시아"),        ("Moscow","莫斯科","モスクワ","모스크바")),
    (("England","英国","イギリス","영국"),         ("London","伦敦","ロンドン","런던")),
    (("America","美国","アメリカ","미국"),         ("Washington","华盛顿","ワシントン","워싱턴")),
    (("Canada","加拿大","カナダ","캐나다"),        ("Ottawa","渥太华","オタワ","오타와")),
    (("Egypt","埃及","エジプト","이집트"),         ("Cairo","开罗","カイロ","카이로")),
    (("India","印度","インド","인도"),            ("Delhi","新德里","ニューデリー","뉴델리")),
    (("Brazil","巴西","ブラジル","브라질"),        ("Brasilia","巴西利亚","ブラジリア","브라질리아")),
    (("Greece","希腊","ギリシャ","그리스"),        ("Athens","雅典","アテネ","아테네")),
    (("Thailand","泰国","タイ","태국"),           ("Bangkok","曼谷","バンコク","방콕")),
    (("Turkey","土耳其","トルコ","터키"),          ("Ankara","安卡拉","アンカラ","앙카라")),
    (("Australia","澳大利亚","オーストラリア","호주"),("Canberra","堪培拉","キャンベラ","캔버라")),
    (("Mexico","墨西哥","メキシコ","멕시코"),      ("Mexico","墨西哥城","メキシコシティ","멕시코시티")),
    (("Netherlands","荷兰","オランダ","네덜란드"), ("Amsterdam","阿姆斯特丹","アムステルダム","암스테르담")),
    (("Portugal","葡萄牙","ポルトガル","포르투갈"),("Lisbon","里斯本","リスボン","리스본")),
    (("Sweden","瑞典","スウェーデン","스웨덴"),    ("Stockholm","斯德哥尔摩","ストックホルム","스톡홀름")),
    (("Norway","挪威","ノルウェー","노르웨이"),    ("Oslo","奥斯陆","オスロ","오슬로")),
    (("Finland","芬兰","フィンランド","핀란드"),   ("Helsinki","赫尔辛基","ヘルシンキ","헬싱키")),
    (("Austria","奥地利","オーストリア","오스트리아"),("Vienna","维也纳","ウィーン","빈")),
    (("Switzerland","瑞士","スイス","스위스"),     ("Bern","伯尔尼","ベルン","베른")),
    (("Poland","波兰","ポーランド","폴란드"),      ("Warsaw","华沙","ワルシャワ","바르샤바")),
    (("Argentina","阿根廷","アルゼンチン","아르헨티나"),("Buenos","布宜诺斯艾利斯","ブエノスアイレス","부에노스아이레스")),
    (("Cuba","古巴","キューバ","쿠바"),           ("Havana","哈瓦那","ハバナ","아바나")),
    (("Vietnam","越南","ベトナム","베트남"),       ("Hanoi","河内","ハノイ","하노이")),
    (("Indonesia","印度尼西亚","インドネシア","인도네시아"),("Jakarta","雅加达","ジャカルタ","자카르타")),
    (("Iran","伊朗","イラン","이란"),             ("Tehran","德黑兰","テヘラン","테헤란")),
    (("Iraq","伊拉克","イラク","이라크"),          ("Baghdad","巴格达","バグダッド","바그다드")),
    (("Cambodia","柬埔寨","カンボジア","캄보디아"),("Phnom","金边","プノンペン","프놈펜")),
    (("Philippines","菲律宾","フィリピン","필리핀"),("Manila","马尼拉","マニラ","마닐라")),
    (("Chile","智利","チリ","칠레"),              ("Santiago","圣地亚哥","サンティアゴ","산티아고")),
    (("Peru","秘鲁","ペルー","페루"),             ("Lima","利马","リマ","리마")),
    (("Kenya","肯尼亚","ケニア","케냐"),           ("Nairobi","内罗毕","ナイロビ","나이로비")),
]

# ---- weekdays Mon..Sun -------------------------------------------------------
DAYS = {
    "en": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
    "zh": ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"],
    "ja": ["月曜日","火曜日","水曜日","木曜日","金曜日","土曜日","日曜日"],
    "ko": ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"],
}

# ---- months Jan..Dec (note Korean 6=유월, 10=시월 irregular) -----------------
MONTHS = {
    "en": ["January","February","March","April","May","June","July","August",
           "September","October","November","December"],
    "zh": ["一月","二月","三月","四月","五月","六月","七月","八月",
           "九月","十月","十一月","十二月"],
    "ja": ["一月","二月","三月","四月","五月","六月","七月","八月",
           "九月","十月","十一月","十二月"],
    "ko": ["일월","이월","삼월","사월","오월","유월","칠월","팔월",
           "구월","시월","십일월","십이월"],
}


def emit(rows, pid, family, by_lang_prompt, by_lang_target):
    for lg in LANGS:
        rows.append({"pair_id": pid, "lang": lg, "family": family,
                     "prompt": by_lang_prompt[lg], "target": by_lang_target[lg]})


def main():
    here = Path(__file__).resolve().parents[2]
    rows = []
    pid = 0

    # 1) antonyms: reuse the original 20 verbatim ------------------------------
    orig = here / "data" / "prompts_expanded_v2.jsonl"
    seen = set()
    with open(orig) as f:
        for line in f:
            r = json.loads(line)
            r["family"] = "antonym"
            rows.append(r)
            seen.add(r["pair_id"])
    pid = max(seen) + 1  # = 20

    # 2) capitals -------------------------------------------------------------
    for (c, cap) in CAPITALS:
        cd = dict(zip(LANGS, c)); kd = dict(zip(LANGS, cap))
        prompt = {
            "en": f"The capital of {cd['en']} is",
            "zh": f"{cd['zh']}的首都是",
            "ja": f"{cd['ja']}の首都は",
            "ko": f"{cd['ko']}의 수도는",
        }
        target = {lg: kd[lg] for lg in LANGS}
        emit(rows, pid, "capital", prompt, target); pid += 1

    # 3a) arithmetic addition: all 1<=a<=b<=9 ---------------------------------
    for a in range(1, 10):
        for b in range(a, 10):
            c = a + b
            prompt = {
                "en": f"{NUM['en'][a]} plus {NUM['en'][b]} equals",
                "zh": f"{NUM['zh'][a]}加{NUM['zh'][b]}等于",
                "ja": f"{NUM['ja'][a]}足す{NUM['ja'][b]}は",
                "ko": f"{NUM['ko'][a]} 더하기 {ko_topic(NUM['ko'][b])}",
            }
            target = {lg: NUM[lg][c] for lg in LANGS}
            emit(rows, pid, "arith_add", prompt, target); pid += 1

    # 3b) arithmetic subtraction: all 1<=b<a<=9 -------------------------------
    for a in range(2, 10):
        for b in range(1, a):
            c = a - b
            prompt = {
                "en": f"{NUM['en'][a]} minus {NUM['en'][b]} equals",
                "zh": f"{NUM['zh'][a]}减{NUM['zh'][b]}等于",
                "ja": f"{NUM['ja'][a]}引く{NUM['ja'][b]}は",
                "ko": f"{NUM['ko'][a]} 빼기 {ko_topic(NUM['ko'][b])}",
            }
            target = {lg: NUM[lg][c] for lg in LANGS}
            emit(rows, pid, "arith_sub", prompt, target); pid += 1

    # 4a) successor n -> n+1  (n=1..19) ---------------------------------------
    for n in range(1, 20):
        prompt = {
            "en": f"The number after {NUM['en'][n]} is",
            "zh": f"{NUM['zh'][n]}的下一个数是",
            "ja": f"{NUM['ja'][n]}の次の数は",
            "ko": f"{NUM['ko'][n]} 다음 수는",
        }
        target = {lg: NUM[lg][n + 1] for lg in LANGS}
        emit(rows, pid, "successor", prompt, target); pid += 1

    # 4b) predecessor n -> n-1 (n=2..20) -------------------------------------
    for n in range(2, 21):
        prompt = {
            "en": f"The number before {NUM['en'][n]} is",
            "zh": f"{NUM['zh'][n]}的上一个数是",
            "ja": f"{NUM['ja'][n]}の前の数は",
            "ko": f"{NUM['ko'][n]} 이전 수는",
        }
        target = {lg: NUM[lg][n - 1] for lg in LANGS}
        emit(rows, pid, "predecessor", prompt, target); pid += 1

    # 4c) doubling n -> 2n (n=1..10) -----------------------------------------
    for n in range(1, 11):
        prompt = {
            "en": f"{NUM['en'][n]} times two equals",
            "zh": f"{NUM['zh'][n]}乘二等于",
            "ja": f"{NUM['ja'][n]}かける二は",
            "ko": f"{NUM['ko'][n]} 곱하기 이는",
        }
        target = {lg: NUM[lg][2 * n] for lg in LANGS}
        emit(rows, pid, "doubling", prompt, target); pid += 1

    # 5a) weekday-after (cyclic) ---------------------------------------------
    for i in range(7):
        j = (i + 1) % 7
        prompt = {
            "en": f"The day after {DAYS['en'][i]} is",
            "zh": f"{DAYS['zh'][i]}的下一天是",
            "ja": f"{DAYS['ja'][i]}の次の日は",
            "ko": f"{DAYS['ko'][i]} 다음 날은",
        }
        target = {lg: DAYS[lg][j] for lg in LANGS}
        emit(rows, pid, "weekday", prompt, target); pid += 1

    # 5b) month-after (cyclic) ------------------------------------------------
    for i in range(12):
        j = (i + 1) % 12
        prompt = {
            "en": f"The month after {MONTHS['en'][i]} is",
            "zh": f"{MONTHS['zh'][i]}的下一个月是",
            "ja": f"{MONTHS['ja'][i]}の次の月は",
            "ko": f"{MONTHS['ko'][i]} 다음 달은",
        }
        target = {lg: MONTHS[lg][j] for lg in LANGS}
        emit(rows, pid, "month", prompt, target); pid += 1

    out = here / "data" / "prompts_v3_multi.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pairs = pid
    from collections import Counter
    fam = Counter(r["family"] for r in rows if r["lang"] == "en")
    print(f"wrote {out}")
    print(f"  total concept pairs = {n_pairs}  ({len(rows)} rows = {n_pairs}x{len(LANGS)})")
    print("  family breakdown (pairs):")
    for k, v in fam.items():
        print(f"    {k:14s} {v}")


if __name__ == "__main__":
    main()
