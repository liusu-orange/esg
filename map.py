import json
import requests
import pandas as pd
import time
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# 1. 配置项
# =========================
INPUT_JSON = r"/home/featurize/work/esg/test1.json"
EXISTING_CSV = r"/home/featurize/work/esg/feature.csv"
UPDATED_CSV = r"/home/featurize/work/esg/feature_target.csv"

HEADERS = {
    # SEC 强制要求 Header 中必须包含你的邮箱，否则会拒绝访问
    "User-Agent": "Zhongyi / zhongyi@bupt.edu.cn"
}

# 若某些公司名在 SEC 自动匹配中失败，可以在这里手动指定其 CIK（去除前导零）
# 比如苹果的 CIK 是 320193
MANUAL_CIK_MAP = {
    "ALPHABET INC.": "1652044",
    "AMAZON COM INC": "1018724",
    "APPLE INC.": "320193",
    "MICROSOFT CORP": "789019",
    "MICROSTRATEGY INC": "1050446",
    "SIMON PROPERTY GROUP INC /DE/": "1063761"
}

# =========================
# 2. 核心数据获取模块
# =========================
def get_sec_cik_mapping() -> dict:
    """获取全美所有上市公司的 名称 -> CIK 映射表"""
    print("🌐 正在从 SEC 下载最新的公司 CIK 映射表...")
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS)
    data = resp.json()
    
    mapping = {}
    for _, v in data.items():
        # 清洗名称：大写，去掉常见标点和后缀，方便模糊匹配
        clean_name = v["title"].upper().replace(",", "").replace(".", "").replace(" INC", "").replace(" CORP", "").strip()
        mapping[clean_name] = str(v["cik_str"])
    return mapping

def clean_firm_name(firm: str) -> str:
    """清洗我们自己 JSON 里的公司名，对齐 SEC 的格式"""
    return firm.upper().replace(",", "").replace(".", "").replace(" INC", "").replace(" CORP", "").strip()

# 初始化一个带有自动重试机制的全局 requests Session
session = requests.Session()
# 设置重试策略：遇到特定错误时最多重试 5 次，每次间隔时间逐渐变长 (0.5s, 1s, 2s...)
retry_strategy = Retry(
    total=5,  
    backoff_factor=0.5,  
    status_forcelist=[429, 500, 502, 503, 504], # 遇到这些状态码自动重试
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

def fetch_net_income_from_sec(cik: str) -> dict:
    """
    通过 SEC API 获取历史 Net Income，自带备选标签和修正案兼容机制
    """
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    
    try:
        resp = session.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return {}
            
        data = resp.json()
        gaap_data = data.get("facts", {}).get("us-gaap", {})
        
        # 定义净利润的备选标签列表（按优先级从高到低）
        possible_tags = [
            "NetIncomeLoss", 
            "NetIncomeLossAvailableToCommonStockholdersBasic",
            "ProfitLoss",
            "ComprehensiveIncomeNetOfTax",
            "IncomeLossFromContinuingOperations"
        ]
        
        yearly_income = {}
        
        # 遍历所有备选标签，寻找年度数据
        for tag in possible_tags:
            if tag not in gaap_data:
                continue
                
            units_usd = gaap_data[tag].get("units", {}).get("USD", [])
            
            for item in units_usd:
                form_type = item.get("form", "")
                fy = item.get("fy") # 财年
                
                # 放宽条件：只要是年报 (10-K) 或年报修正案 (10-K/A) 都接收
                if fy and ("10-K" in form_type):
                    # 如果这个财年还没有数据，直接写入
                    if fy not in yearly_income:
                        yearly_income[fy] = item
                    # 如果已经有数据，取最新发布的报告（end 日期更大）
                    elif item.get("end", "") > yearly_income[fy].get("end", ""):
                        yearly_income[fy] = item
            
            # 如果当前优先级最高的标签已经成功提取到了足够多年的数据，就可以提前结束寻找
            if len(yearly_income) >= 5: 
                break
                
        # 简化为 {year: value} 字典
        return {year: info["val"] for year, info in yearly_income.items()}
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 获取 CIK {cik} 数据时网络超时，已跳过")
        return {}
    except ValueError:
        print(f"\n❌ 获取 CIK {cik} 数据时 JSON 解析失败")
        return {}

# =========================
# 3. 标签生成与 DataFrame 构建
# =========================
def generate_target_df() -> pd.DataFrame:
    # 1. 准备 CIK 映射表
    sec_mapping = get_sec_cik_mapping()
    
    # 2. 读取 JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        firm_urls_map = json.load(f)
        
    firm_list = []
    url_list = []
    target_list = []
    
    for firm, urls in tqdm(firm_urls_map.items(), desc="处理公司进度"):
        # --- 步骤 A：匹配公司 CIK ---
        clean_name = clean_firm_name(firm)
        cik = MANUAL_CIK_MAP.get(firm.upper()) or sec_mapping.get(clean_name)
        
        if not cik:
            print(f"\n⚠️ 找不到公司 '{firm}' 的 CIK，请在 MANUAL_CIK_MAP 中手动添加。")
            labels = [None] * 8
        else:
            # --- 步骤 B：获取历史财务数据并计算 8 个 Label ---
            time.sleep(0.1) # SEC 限制每秒最多 10 个请求，稍微休眠防封
            income_history = fetch_net_income_from_sec(cik)
            
            labels = []
            # 遍历要求的 8 个年份对比：2017vs16, 2018vs17 ... 2024vs23
            for target_year in range(2017, 2025):
                val_curr = income_history.get(target_year)
                val_prev = income_history.get(target_year - 1)
                
                if val_curr is not None and val_prev is not None:
                    labels.append(1 if val_curr > val_prev else 0)
                    print(f"{firm}:prev:{val_prev}:curr:{val_curr}")
                else:
                    print(f"{firm}:返回了None")
                    labels.append(0) # 如果某一年数据缺失，记为 None
        
        # --- 步骤 C：将这 8 个 Label 分配给该公司的 URLs ---
        # 假设 JSON 里的 urls 列表是按年份升序排列的（2017 对应第一个，2024 对应第八个）
        for idx, url in enumerate(urls):
            target_val = labels[idx] if idx < len(labels) else None
            
            firm_list.append(firm)
            url_list.append(url)
            target_list.append(target_val)

    # 生成最终 DataFrame
    return pd.DataFrame({
        "firm": firm_list,
        "url": url_list,
        "target": target_list
    })

# =========================
# 4. 主流程拼接
# =========================
def main():
    print("\n📖 读取原有CSV文件...")
    try:
        df_original = pd.read_csv(EXISTING_CSV, encoding="utf-8")
    except Exception as e:
        print(f"❌ 读取CSV失败：{e}")
        return
        
    print("\n🔧 正在通过 SEC API 获取确切财务数据并生成 target...")
    df_target = generate_target_df()
    
    if len(df_original) != len(df_target):
        print(f"⚠️ 行数不匹配！原有CSV行数：{len(df_original)}，target行数：{len(df_target)}")
        min_rows = min(len(df_original), len(df_target))
        df_original = df_original.head(min_rows).reset_index(drop=True)
        df_target = df_target.head(min_rows).reset_index(drop=True)
        print(f"✅ 已自动截断至相同行数：{min_rows}")
        
    print("\n🔗 直接拼接 target 列到原有 CSV...")
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