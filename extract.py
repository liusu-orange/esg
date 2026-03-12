import json
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

# =========================
# 1. 配置
# =========================

INPUT_JSON = r"/home/featurize/work/esg/test.json"
OUTPUT_CSV = r"/home/featurize/work/esg/esg_sections.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36; [Zhongyi/] [zhongyi@bupt.edu.cn]"
}

# ✅【修改点 1】明确“起始 Item + 终止 Item”
# ITEM_CONFIG = {
#     "item_1a_risk_factors": {
#         "start": r"item\s+1a[\.\:\-]?\s?risk\s+factors",
#         "end": r"item\s+1b[\.\:\-]?"
#     },
#     "item_7_mdna": {
#         "start": r"item\s+7[\.\:\-]?\s+management[’']?s\s+discussion",
#         "end": r"item\s+7a[\.\:\-]?"
#     }
# }
ITEM_CONFIG = {
    "item_1a_risk_factors": {
        "start": r"item\s*1a[\.\:\-]?\s*risk\s*factors",
        "end": r"item\s+1b[\.\:\-]?"
    },
    "item_7_mdna": {
        "start": r"item\s*7[\.\:\-]?\s*management[’']?s\s*discussion",
        "end": r"item\s+7a[\.\:\-]?"
    },
    "item_7_A": {
        "start": r"item\s*7a[\.\:\-]?\s*quantitative\s*and\s*qualitative",
        "end": r"item\s+8[\.\:\-]?"
    },
    "item_8": {
        "start": r"item\s*8[\.\:\-]?\s*financial\s*statements\s*and\s*supplementary",
        "end": r"item\s+9[\.\:\-]?"
        }
    
}

# =========================
# 2. HTML → 清洗文本
# =========================

# 2. 核心函数：从XBRL的.htm文件提取可读文本
def load_10k_text(url: str) -> str:
    """
    从SEC的10-K文件提取可读文本（自适应2020年前/后格式）
    :param url: SEC的10-K文件URL
    :return: 清洗后的文本（非空）
    """
    
    
    # --------------------------
    # 步骤1：发送请求（带防反爬）
    # --------------------------
    try:
       
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        print(f"✅ 请求成功 - 状态码: {resp.status_code}")
    except Exception as e:
        print(f"❌ 下载失败：{type(e).__name__}: {e}")
        return ""

    # --------------------------
    # 步骤2：自适应解析（先HTML，后XML）
    # --------------------------
    # 初始化文本容器
    text_parts = []
    
    # 模式1：HTML解析（适配2020年后）
    soup_html = BeautifulSoup(resp.text, "html.parser")
    # 提取HTML核心标签（div/p/span/td）
    html_tags = soup_html.find_all(["div", "p"])
    for tag in html_tags:
        tag_text = tag.get_text(strip=True)
        if tag_text and len(tag_text) > 5:
            text_parts.append(tag_text)
    
    
    # 合并并清洗
    full_text = "\n".join(text_parts)
    # 替换特殊字符
    full_text = re.sub(r"\xa0|&#160;|&nbsp;", " ", full_text)
    # 标准化换行和空格
    full_text = re.sub(r"\n+", "\n", full_text)
    full_text = re.sub(r"\s{2,}", " ", full_text)
    # 转小写（保持原逻辑）
    full_text = full_text.lower()
    
    return full_text




# =========================
# 3. 去掉 Table of Contents
# =========================

def strip_table_of_contents(text: str) -> str:
    """
    去掉 Table of Contents，只保留正文部分。
    
    核心思想：
    - 目录和正文都会出现 Item 1 / Item 1A
    - 目录一定在前，正文一定在后
    - 从“第二次出现 Item 1.” 或 “Item 1A.” 开始，认为是正文
    """
    # patterns =r"\nitem\s+1[\.\:\-]?\s+business"  
    patterns = r"i?item\s*1[\.]+\s*business"
    matches = list(re.finditer(patterns, text))
    if(len(matches)>=2):
        return text[matches[-1].start():]
    return text

    


def extract_item(text: str, start_pat: str, end_pat: str) -> str | None:
    starts = list(re.finditer(start_pat, text))
    if not starts:
        return None

    # ✅【修改点 3】使用正文中的第一个 start（目录已被 strip 掉）
    start = starts[0].start()
    rest = text[start + 1:]

    end_match = re.search(end_pat, rest)
    if end_match:
        end = start + 1 + end_match.start()
    else:
        end = len(text)

    section = text[start:end].strip()

    if len(section.split()) < 30:
        print("none")
        return None

    return section


# =========================
# 5. 主抽取逻辑
# =========================

def extract_esg_sections(url: str) -> dict:
    text = load_10k_text(url)
    text = strip_table_of_contents(text)
    sections = {}
    for name, cfg in ITEM_CONFIG.items():
        sec = extract_item(text, cfg["start"], cfg["end"])
        if sec:
            sections[name] = sec

    return sections




# =========================
# 5. 年份分配逻辑（按列表位置）
# =========================

def infer_year(url: str, position: int, start_year: int = 2017):
    """
    每个公司 urls 列表中：
    position=0 -> 2017
    position=1 -> 2018
    ...
    position=7 -> 2024
    """
    _ = url
    return start_year + position


# =========================
# 6. 主流程
# =========================

def main():
    with open(INPUT_JSON, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    rows = []

    for firm, urls in tqdm(data.items(), desc="Processing firms"):
        for idx, url in enumerate(urls):
            year = infer_year(url, idx)
            sections = extract_esg_sections(url)
            for sec_name, text in sections.items():
                rows.append({
                    "firm": firm,
                    "year": year,
                    "section": sec_name,
                    "url": url,
                    "text": text
                })
            

           

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["firm", "year"])
    df.to_csv(OUTPUT_CSV, index=False)
    pd.set_option("display.max_rows", None)
    print(f"[DONE] Saved to {OUTPUT_CSV}")
    print(df.head(500))


if __name__ == "__main__":
    main()
