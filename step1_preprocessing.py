"""
Step 1: Data Preprocessing & Feature Engineering
Run this FIRST before ML models
"""

import pandas as pd
import numpy as np
import pickle

print("="*80)
print("STEP 1: DATA PREPROCESSING & FEATURE ENGINEERING")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n1. Loading augmented data...")
df = pd.read_excel('unicorn_data_augmented.xlsx')

print(f"   Total companies: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# ============================================================================
# DATA CLEANING
# ============================================================================

print("\n2. Data cleaning...")

# Remove rows without Years_to_Unicorn (our target variable)
initial_count = len(df)
df_clean = df[df['Years_to_Unicorn'].notna()].copy()
removed = initial_count - len(df_clean)
print(f"   Removed {removed} rows with missing Years_to_Unicorn")

# Remove unrealistic values (negative or >50 years)
df_clean = df_clean[df_clean['Years_to_Unicorn'] > 0].copy()
df_clean = df_clean[df_clean['Years_to_Unicorn'] <= 50].copy()
print(f"   After removing outliers: {len(df_clean)} companies")

# Remove rows with missing critical features
df_clean = df_clean[df_clean['Industry'].notna()].copy()
df_clean = df_clean[df_clean['Country'].notna()].copy()
print(f"   After removing missing features: {len(df_clean)} companies")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n3. Feature engineering...")

# === TEMPORAL FEATURES ===
df_clean['Company_Age_2025'] = 2025 - df_clean['Year_Founded']

# Founding Era
def categorize_era(year):
    if pd.isna(year):
        return 'Unknown'
    if year < 2000:
        return 'Pre-2000'
    elif year < 2010:
        return '2000-2009'
    elif year < 2015:
        return '2010-2014'
    elif year < 2020:
        return '2015-2019'
    else:
        return '2020+'

df_clean['Founding_Era'] = df_clean['Year_Founded'].apply(categorize_era)

# === VALUATION FEATURES ===
df_clean['Log_Valuation'] = np.log(df_clean['Valuation ($B)'])

def categorize_valuation(val):
    if val < 5:
        return 'Small'
    elif val < 10:
        return 'Medium'
    elif val < 50:
        return 'Large'
    else:
        return 'Mega'

df_clean['Valuation_Category'] = df_clean['Valuation ($B)'].apply(categorize_valuation)

# === GEOGRAPHIC FEATURES ===
tech_hubs = ['San Francisco', 'Palo Alto', 'Mountain View', 'Menlo Park', 'San Jose',
             'New York', 'Beijing', 'Shenzhen', 'Bangalore', 'London', 'Tel Aviv', 
             'Boston', 'Seattle', 'Los Angeles']

df_clean['Is_Tech_Hub'] = df_clean['City'].isin(tech_hubs).astype(int)

# Silicon Valley specifically
sv_cities = ['San Francisco', 'Palo Alto', 'Mountain View', 'Menlo Park', 'San Jose']
df_clean['Is_Silicon_Valley'] = df_clean['City'].isin(sv_cities).astype(int)

# Country tier
tier1 = ['United States', 'China']
tier2 = ['India', 'United Kingdom', 'Germany', 'Israel', 'Singapore', 
         'South Korea', 'Japan', 'Canada', 'France', 'Sweden']

def categorize_country(country):
    if country in tier1:
        return 1
    elif country in tier2:
        return 2
    else:
        return 3

df_clean['Country_Tier'] = df_clean['Country'].apply(categorize_country)

# === INVESTOR FEATURES ===
# Extract investor count from Select Investors
df_clean['Investor_Count'] = df_clean['Select Investors'].fillna('').str.count(',') + 1
df_clean.loc[df_clean['Select Investors'].isna(), 'Investor_Count'] = 0

# Has top-tier VC
top_vcs = ['sequoia', 'andreessen horowitz', 'a16z', 'tiger global', 'softbank',
           'accel', 'benchmark', 'insight partners', 'general catalyst', 
           'lightspeed', 'greylock', 'kleiner perkins', 'khosla']

def has_top_vc(investors_str):
    if pd.isna(investors_str):
        return 0
    investors_lower = investors_str.lower()
    for vc in top_vcs:
        if vc in investors_lower:
            return 1
    return 0

df_clean['Has_Top_VC'] = df_clean['Select Investors'].apply(has_top_vc)

# === INDUSTRY FEATURES ===
# Simplify industry categories
def group_industry(industry):
    if pd.isna(industry):
        return 'Other'
    ind_lower = str(industry).lower()
    
    if any(word in ind_lower for word in ['fintech', 'financial', 'payment', 'banking', 'insurance']):
        return 'Fintech'
    elif any(word in ind_lower for word in ['software', 'saas', 'enterprise', 'tech', 'data', 'cloud', 'cybersecurity']):
        return 'Enterprise_Tech'
    elif any(word in ind_lower for word in ['ai', 'artificial intelligence', 'machine learning']):
        return 'AI_ML'
    elif any(word in ind_lower for word in ['health', 'bio', 'medical', 'pharma']):
        return 'Healthcare'
    elif any(word in ind_lower for word in ['commerce', 'retail', 'marketplace', 'consumer']):
        return 'E-commerce'
    elif any(word in ind_lower for word in ['media', 'entertainment', 'gaming', 'content']):
        return 'Media'
    elif any(word in ind_lower for word in ['transport', 'mobility', 'logistics', 'delivery', 'auto']):
        return 'Mobility'
    else:
        return 'Other'

df_clean['Industry_Group'] = df_clean['Industry'].apply(group_industry)

# Tech intensity (1 = high tech, 0 = low tech)
high_tech_industries = ['Enterprise_Tech', 'AI_ML', 'Fintech']
df_clean['Is_Tech_Intensive'] = df_clean['Industry_Group'].isin(high_tech_industries).astype(int)

print(f"   Created {df_clean.shape[1] - df.shape[1]} new features")

# ============================================================================
# ENCODING CATEGORICAL VARIABLES
# ============================================================================

print("\n4. Encoding categorical variables...")

# One-hot encoding for nominal variables
df_encoded = pd.get_dummies(df_clean, 
                             columns=['Industry_Group', 'Founding_Era', 'Valuation_Category'],
                             prefix=['Ind', 'Era', 'ValCat'],
                             drop_first=True)

print(f"   Total columns after encoding: {df_encoded.shape[1]}")

# ============================================================================
# FEATURE SELECTION FOR MODELING
# ============================================================================

print("\n5. Selecting features for modeling...")

# Target variable
target_col = 'Years_to_Unicorn'

# Feature columns
numeric_features = [
    'Valuation ($B)',
    'Log_Valuation',
    'Company_Age_2025',
    'Date_Joined_Year',
    'Year_Founded',
    'Investor_Count',
    'Is_Tech_Hub',
    'Is_Silicon_Valley',
    'Country_Tier',
    'Has_Top_VC',
    'Is_Tech_Intensive'
]

# Add one-hot encoded columns
encoded_features = [col for col in df_encoded.columns if col.startswith(('Ind_', 'Era_', 'ValCat_'))]

feature_cols = numeric_features + encoded_features

# Verify all features exist
feature_cols = [col for col in feature_cols if col in df_encoded.columns]

print(f"   Total features for modeling: {len(feature_cols)}")
print(f"   Numeric features: {len(numeric_features)}")
print(f"   Encoded features: {len(encoded_features)}")

# Create feature matrix and target
X = df_encoded[feature_cols].copy()
y = df_encoded[target_col].copy()

print(f"\n   Feature matrix shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================

print("\n6. Creating train-test split...")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# ============================================================================
# FEATURE SCALING
# ============================================================================

print("\n7. Scaling features...")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"   Features scaled using StandardScaler")

# ============================================================================
# SAVE PREPROCESSED DATA
# ============================================================================

print("\n8. Saving preprocessed data...")

data_package = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'X_train_scaled': X_train_scaled,
    'X_test_scaled': X_test_scaled,
    'feature_names': list(X.columns),
    'df_full': df_encoded,
    'scaler': scaler
}

with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(data_package, f)

print(f"   ✓ Saved: preprocessed_data.pkl")

# Also save as CSV for inspection
df_encoded.to_csv('data_with_features.csv', index=False)
print(f"   ✓ Saved: data_with_features.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("PREPROCESSING SUMMARY")
print("="*80)

print(f"\nDataset Size:")
print(f"  Original: {initial_count} companies")
print(f"  After cleaning: {len(df_clean)} companies")
print(f"  Training set: {len(X_train)} companies")
print(f"  Test set: {len(X_test)} companies")

print(f"\nTarget Variable (Years_to_Unicorn):")
print(f"  Mean: {y.mean():.2f} years")
print(f"  Median: {y.median():.2f} years")
print(f"  Std Dev: {y.std():.2f} years")
print(f"  Min: {y.min():.0f} years")
print(f"  Max: {y.max():.0f} years")

print(f"\nFeatures:")
print(f"  Total features: {len(feature_cols)}")
print(f"  Numeric features: {len(numeric_features)}")
print(f"  One-hot encoded: {len(encoded_features)}")

print(f"\nTop 10 Industries:")
top_industries = df_clean['Industry_Group'].value_counts().head(10)
for ind, count in top_industries.items():
    pct = count / len(df_clean) * 100
    print(f"  {ind}: {count} ({pct:.1f}%)")

print(f"\nTop 5 Countries:")
top_countries = df_clean['Country'].value_counts().head(5)
for country, count in top_countries.items():
    pct = count / len(df_clean) * 100
    print(f"  {country}: {count} ({pct:.1f}%)")

print(f"\nInvestor Statistics:")
print(f"  Companies with Top VC: {df_clean['Has_Top_VC'].sum()} ({df_clean['Has_Top_VC'].mean()*100:.1f}%)")
print(f"  Avg investors per company: {df_clean['Investor_Count'].mean():.1f}")

print(f"\nGeographic Distribution:")
print(f"  Tech Hub companies: {df_clean['Is_Tech_Hub'].sum()} ({df_clean['Is_Tech_Hub'].mean()*100:.1f}%)")
print(f"  Silicon Valley: {df_clean['Is_Silicon_Valley'].sum()} ({df_clean['Is_Silicon_Valley'].mean()*100:.1f}%)")

print("\n" + "="*80)
print("✅ PREPROCESSING COMPLETE")
print("="*80)
print("\nNext steps:")
print("  1. Run 'step2_ml_models.py' for regression analysis (Objective 2)")
print("  2. Feature importance will be calculated automatically (Objective 3)")
print("  3. Then run 'step3_porters_analysis.py' (Objective 4)")
print("\nReady for modeling!")