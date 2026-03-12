import pandas as pd

INPUT_CSV = r"/home/featurize/work/esg/preprocessed_esg_data.csv"
OUTPUT_CSV = r"/home/featurize/work/esg/preprocessed_esg_data1.csv"
MIN_YEARS = 8


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    required_columns = {"firm", "year"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV 缺少必要字段: {missing}")

    df = df.dropna(subset=["firm", "year"]).copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    firm_year_counts = df.groupby("firm")["year"].nunique()
    valid_firms = firm_year_counts[firm_year_counts >= MIN_YEARS].index

    filtered_df = df[df["firm"].isin(valid_firms)].copy()
    filtered_df = filtered_df.sort_values(by=["firm", "year"]).reset_index(drop=True)

    filtered_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    removed_firms = firm_year_counts[firm_year_counts < MIN_YEARS]
    print(f"原始行数: {len(df)}")
    print(f"过滤后行数: {len(filtered_df)}")
    print(f"保留公司数: {len(valid_firms)}")
    print(f"移除公司数: {len(removed_firms)}")
    if not removed_firms.empty:
        print("\n被移除的公司(唯一年份数 < 8):")
        print(removed_firms.sort_values())


if __name__ == "__main__":
    main()
