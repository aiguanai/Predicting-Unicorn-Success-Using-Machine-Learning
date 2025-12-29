import pandas as pd
import numpy as np

# Load augmented data
df = pd.read_excel('unicorn_data_augmented.xlsx')

print("="*60)
print("DATA VALIDATION REPORT")
print("="*60)

# 1. Coverage statistics
total = len(df)
found = df['Year_Founded'].notna().sum()
missing = df['Year_Founded'].isna().sum()
coverage = found / total * 100

print(f"\nData Coverage:")
print(f"  Total companies: {total}")
print(f"  Year Founded found: {found} ({coverage:.1f}%)")
print(f"  Missing: {missing} ({100-coverage:.1f}%)")

# 2. Source breakdown
print(f"\nData Sources:")
sources = df['Data_Source'].value_counts()
for source, count in sources.items():
    pct = count / total * 100
    print(f"  {source}: {count} ({pct:.1f}%)")

# 3. Data quality checks
print(f"\nData Quality:")

# Check 1: Invalid dates (founded after unicorn)
invalid_dates = df[df['Valid'] == False]
print(f"  Invalid dates: {len(invalid_dates)}")
if len(invalid_dates) > 0:
    print("  ⚠️  These need manual review:")
    for _, row in invalid_dates.head(10).iterrows():
        print(f"    - {row['Company']}: Founded {row['Year_Founded']}, Unicorn {row['Date_Joined_Year']}")

# Check 2: Unreasonable founding years
too_old = df[df['Year_Founded'] < 1950]
print(f"  Founded before 1950: {len(too_old)}")

# Check 3: Very recent founding years
too_new = df[df['Year_Founded'] > 2023]
print(f"  Founded after 2023: {len(too_new)}")

# 4. Years to Unicorn statistics
valid_growth = df[df['Years_to_Unicorn'].notna() & (df['Years_to_Unicorn'] > 0)]
print(f"\nGrowth Speed Statistics (Years to Unicorn):")
print(f"  Count: {len(valid_growth)}")
print(f"  Mean: {valid_growth['Years_to_Unicorn'].mean():.1f} years")
print(f"  Median: {valid_growth['Years_to_Unicorn'].median():.1f} years")
print(f"  Min: {valid_growth['Years_to_Unicorn'].min():.0f} years")
print(f"  Max: {valid_growth['Years_to_Unicorn'].max():.0f} years")

# 5. Fast growers (< 5 years)
fast = df[df['Years_to_Unicorn'] < 5]
print(f"\nFast Growers (<5 years): {len(fast)}")
if len(fast) > 0:
    print("  Examples:")
    for _, row in fast.head(5).iterrows():
        print(f"    - {row['Company']}: {row['Years_to_Unicorn']:.0f} years")

# 6. Slow growers (> 15 years)
slow = df[df['Years_to_Unicorn'] > 15]
print(f"\nPatient Growers (>15 years): {len(slow)}")
if len(slow) > 0:
    print("  Examples:")
    for _, row in slow.head(5).iterrows():
        print(f"    - {row['Company']}: {row['Years_to_Unicorn']:.0f} years")

# 7. Industry breakdown (top 10)
print(f"\nTop 10 Industries:")
top_industries = df['Industry'].value_counts().head(10)
for industry, count in top_industries.items():
    pct = count / total * 100
    print(f"  {industry}: {count} ({pct:.1f}%)")

# 8. Geographic breakdown (top 10)
print(f"\nTop 10 Countries:")
top_countries = df['Country'].value_counts().head(10)
for country, count in top_countries.items():
    pct = count / total * 100
    print(f"  {country}: {count} ({pct:.1f}%)")

# Save clean dataset (only valid entries)
clean_df = df[df['Valid'] != False].copy()
clean_df.to_excel('unicorn_data_clean.xlsx', index=False)
print(f"\n✓ Clean dataset saved: unicorn_data_clean.xlsx ({len(clean_df)} companies)")

# Save failed companies for manual lookup
if missing > 0:
    failed = df[df['Year_Founded'].isna()][['Company', 'Valuation ($B)', 'Industry', 'Country']]
    failed = failed.sort_values('Valuation ($B)', ascending=False)
    failed.to_csv('failed_companies_manual_lookup.csv', index=False)
    print(f"✓ Failed companies list saved: failed_companies_manual_lookup.csv")

print("\n" + "="*60)
print("VALIDATION COMPLETE")
print("="*60)