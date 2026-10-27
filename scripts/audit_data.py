"""Per-site audit: row count, missing values per feature, label balance."""
from fl_heart.data.loader import load_all_sites, TARGET

for site, df in load_all_sites().items():
    print(f"\n=== {site} ({len(df)} rows) ===")
    print("missing values:")
    print(df.isna().sum()[lambda s: s > 0].to_string() or "  none")
    print("label balance (num>0):")
    print((df[TARGET] > 0).value_counts(normalize=True).round(3).to_string())
