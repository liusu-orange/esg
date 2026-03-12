import json
import requests
import pandas as pd
import time
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# 1. 核心配置项
# =========================
INPUT_JSON = r"/home/featurize/work/esg/test1.json"
EXISTING_CSV = r"/home/featurize/work/esg/feature.csv"
UPDATED_CSV = r"/home/featurize/work/esg/feature_target1.csv"

# SEC 强制要求 Header 中包含你的邮箱
HEADERS = {
    "User-Agent": "Zhongyi / zhongyi@bupt.edu.cn"
}

# 手动 CIK 映射表（补充 SEC 自动匹配不到的异名公司，CIK去前导零）
MANUAL_CIK_MAP = {
    "ALPHABET INC.": "1652044",
    "AMAZON COM INC": "1018724",
    "APPLE INC.": "320193",
    "MICROSOFT CORP": "789019",
    "MICROSTRATEGY INC": "1050446",
    "SIMON PROPERTY GROUP INC /DE/": "1063761"
}

# 财务四大维度及对应的 SEC XBRL 备选标签（按优先级排序）
METRICS_DICT = {
    "Revenue": [
        "Revenues", 
        "SalesRevenueNet", 
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    ],
    "Operating_Income": [
        "OperatingIncomeLoss"
    ],
    "Cash_Flow": [
        "NetCashProvidedByUsedInOperatingActivities"
    ],
    "Net_Income": [
        "NetIncomeLoss", 
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss"
    ]
}

# 四大维度的权重配置（防操纵、重现金、重主业）
METRICS_WEIGHTS = {
    "Cash_Flow": 0.40,        # 经营现金流 (含金量最高)
    "Operating_Income": 0.30, # 营业利润 (主业赚钱能力)
    "Revenue": 0.20,          # 营业收入 (市场扩张能力)
    "Net_Income": 0.10        # 净利润 (易受非经营性损益干扰)
}

# =========================
# 2. 网络请求与重试机制初始化
# =========================
session = requests.Session()
retry_strategy = Retry(
    total=5,  # 最大重试5次
    backoff_factor=0.5,  # 重试间隔逐渐变长
    status_forcelist=[429, 500, 502, 503, 504], # 遇到这些服务器错误自动重试
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# =========================
# 3. 核心功能函数
# =========================
def get_sec_cik_mapping() -> dict:
    """获取全美所有上市公司的 名称 -> CIK 映射表"""
    print("🌐 正在从 SEC 下载最新的公司 CIK 映射表...")
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = session.get(url, headers=HEADERS, timeout=10)
        data = resp.json()
        mapping = {}
        for _, v in data.items():
            clean_name = v["title"].upper().replace(",", "").replace(".", "").replace(" INC", "").replace(" CORP", "").strip()
            mapping[clean_name] = str(v["cik_str"])
        return mapping
    except Exception as e:
        print(f"❌ 下载 CIK 映射表失败: {e}")
        return {}

def clean_firm_name(firm: str) -> str:
    """清洗本地 JSON 里的公司名，对齐 SEC 格式"""
    return firm.upper().replace(",", "").replace(".", "").replace(" INC", "").replace(" CORP", "").strip()

def fetch_comprehensive_metrics(cik: str) -> dict:
    """获取公司四大维度的历史数据"""
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    
    result_data = {key: {} for key in METRICS_DICT.keys()}
    
    try:
        resp = session.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return result_data
            
        data = resp.json()
        gaap_data = data.get("facts", {}).get("us-gaap", {})
        
        for metric_name, possible_tags in METRICS_DICT.items():
            for tag in possible_tags:
                if tag not in gaap_data:
                    continue
                    
                units_usd = gaap_data[tag].get("units", {}).get("USD", [])
                for item in units_usd:
                    form_type = item.get("form", "")
                    fy = item.get("fy")
                    
                    if fy and ("10-K" in form_type): # 兼容 10-K 和 10-K/A
                        if fy not in result_data[metric_name]:
                            result_data[metric_name][fy] = item
                        # 同一年如果有多次修正，保留 end 最大的最新数据
                        elif item.get("end", "") > result_data[metric_name][fy].get("end", ""):
                            result_data[metric_name][fy] = item
                
                # 如果当前标签找到了足够的数据（>=4年），就不用找备选标签了
                if len(result_data[metric_name]) >= 4:
                    break
                    
        # 将结构简化为 { "Revenue": {2016: 100, 2017: 120...}, ... }
        for metric_name in result_data:
            result_data[metric_name] = {year: info["val"] for year, info in result_data[metric_name].items()}
            
        return result_data
        
    except Exception as e:
        print(f"\n❌ 获取 CIK {cik} 数据时出错: {type(e).__name__}")
        return result_data

def evaluate_performance_weighted(metrics_history: dict, target_year: int,firm) -> int | None:
    """根据加权算法计算综合表现 Target (1或0)"""
    valid_score = 0.0          
    total_valid_weight = 0.0   
    
    for metric_name, history in metrics_history.items():
        val_curr = history.get(target_year)
        val_prev = history.get(target_year - 1)
        weight = METRICS_WEIGHTS.get(metric_name, 0)
        
        if val_curr is not None and val_prev is not None:
            total_valid_weight += weight
            if val_curr > val_prev:
                valid_score += weight
                
    # 至少要有占总权重 50% 以上的数据有效，才进行评判
    if total_valid_weight >= 0.50:
        normalized_score = valid_score / total_valid_weight
        print(f"score:{firm}:{normalized_score}")
        return 1 if normalized_score >= 0.5 else 0
    else:
        print(f"{firm}:{total_valid_weight}weight不够返回0")
        return 0

# =========================
# 4. Target DataFrame 构建
# =========================
def generate_target_df() -> pd.DataFrame:
    sec_mapping = get_sec_cik_mapping()
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        firm_urls_map = json.load(f)
        
    firm_list = []
    url_list = []
    target_list = []
    
    for firm, urls in tqdm(firm_urls_map.items(), desc="处理公司进度"):
        clean_name = clean_firm_name(firm)
        cik = MANUAL_CIK_MAP.get(firm.upper()) or sec_mapping.get(clean_name)
        
        if not cik:
            print(f"\n⚠️ 找不到公司 '{firm}' 的 CIK，请在 MANUAL_CIK_MAP 中手动添加。")
            labels = [None] * 8
        else:
            time.sleep(0.15) # 友好的请求间隔，防封禁
            metrics_history = fetch_comprehensive_metrics(cik)
            
            labels = []
            # 遍历 2017(对比2016) 到 2024(对比2023)，共8个Label
            for target_year in range(2017, 2025):
                target_val = evaluate_performance_weighted(metrics_history, target_year,firm)
                labels.append(target_val)
        
        # 将生成的 8 个 label 匹配到对应的 URL 上
        for idx, url in enumerate(urls):
            t_val = labels[idx] if idx < len(labels) else None
            firm_list.append(firm)
            url_list.append(url)
            target_list.append(t_val)

    return pd.DataFrame({
        "firm": firm_list,
        "url": url_list,
        "target": target_list
    })

# =========================
# 5. 主流程
# =========================
def main():
    print("\n📖 读取原有CSV文件...")
    try:
        df_original = pd.read_csv(EXISTING_CSV, encoding="utf-8")
    except Exception as e:
        print(f"❌ 读取CSV失败：{e}")
        return
        
    print("\n🔧 正在通过 SEC API 提取四大财务维度并生成加权 Target...")
    df_target = generate_target_df()
    
    if len(df_original) != len(df_target):
        print(f"\n⚠️ 行数不匹配！原有CSV行数：{len(df_original)}，target行数：{len(df_target)}")
        min_rows = min(len(df_original), len(df_target))
        df_original = df_original.head(min_rows).reset_index(drop=True)
        df_target = df_target.head(min_rows).reset_index(drop=True)
        print(f"✅ 已自动截断至相同行数：{min_rows}")
        
    print("\n🔗 正在拼接 target 列到原有 CSV...")
    df_target_only = df_target[["target"]].reset_index(drop=True)
    df_original_reset = df_original.reset_index(drop=True)
    df_updated = pd.concat([df_original_reset, df_target_only], axis=1)
    
    try:
        df_updated.to_csv(UPDATED_CSV, index=False, encoding="utf-8")
        print(f"\n✅ 大功告成！API 提取结果保存至：{UPDATED_CSV}")
        print(f"📊 统计：")
        print(f"   - 最终CSV行数：{len(df_updated)}")
        print(f"   - target 缺失值：{df_updated['target'].isna().sum()}")
        print(f"   - target 涨跌分布：\n{df_updated['target'].value_counts(dropna=True)}")
    except Exception as e:
        print(f"❌ 保存CSV失败：{e}")

if __name__ == "__main__":
    main()