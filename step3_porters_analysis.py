"""
Step 3: Enhanced Strategic Framework Analysis
OBJECTIVE 4: Theoretical Validation with Advanced Analytics
Porter's Five Forces + Resource-Based View + Network Effects
WITH: Effect Sizes, ML Integration, Enhanced Visualizations, Strategic Recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway
import pickle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

print("="*80)
print("ENHANCED STRATEGIC FRAMEWORK ANALYSIS")
print("Porter's Five Forces | Resource-Based View | Network Effects")
print("WITH: Effect Sizes, ML Integration, Enhanced Visualizations")
print("="*80)

# ============================================================================
# HELPER FUNCTIONS FOR EFFECT SIZES
# ============================================================================

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def eta_squared_anova(groups):
    """Calculate eta-squared effect size for ANOVA"""
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    
    if ss_total == 0:
        return 0
    return ss_between / ss_total

def interpret_eta_squared(eta2):
    """Interpret eta-squared effect size"""
    if eta2 < 0.01:
        return "negligible"
    elif eta2 < 0.06:
        return "small"
    elif eta2 < 0.14:
        return "medium"
    else:
        return "large"

def confidence_interval_mean(data, confidence=0.95):
    """Calculate confidence interval for mean"""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
import os
preprocessed_path = 'output/models/preprocessed_data.pkl'
if not os.path.exists(preprocessed_path):
    preprocessed_path = 'preprocessed_data.pkl'

with open(preprocessed_path, 'rb') as f:
    data = pickle.load(f)

df = data['df_full']

# Load feature importance from final model
feature_importance_paths = [
    'output/data/final_feature_importance.csv',
    'output/data/investor_temporal_feature_importance.csv',
    'output/data/improved_feature_importance.csv',
    'output/data/advanced_feature_importance.csv',
    'feature_importance_rankings.csv'
]

feature_importance = None
for path in feature_importance_paths:
    if os.path.exists(path):
        feature_importance = pd.read_csv(path)
        print(f"[OK] Loaded feature importance from: {path}")
        break

if feature_importance is None:
    print("[WARNING] Feature importance file not found, continuing without it...")
else:
    # Create mapping for ML integration
    feature_importance_dict = dict(zip(
        feature_importance['Feature'], 
        feature_importance['Importance']
    ))

print(f"[OK] Loaded {len(df)} companies")

# Reconstruct Industry_Group if needed
if 'Industry_Group' not in df.columns and 'Industry' in df.columns:
    print("Reconstructing Industry_Group...")
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
    df['Industry_Group'] = df['Industry'].apply(group_industry)

# Recreate Is_Tech_Intensive
if 'Is_Tech_Intensive' not in df.columns and 'Industry_Group' in df.columns:
    high_tech = ['Enterprise_Tech', 'AI_ML', 'Fintech']
    df['Is_Tech_Intensive'] = df['Industry_Group'].isin(high_tech).astype(int)

# ============================================================================
# ENHANCED PORTER'S FIVE FORCES ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PART 1: ENHANCED PORTER'S FIVE FORCES ANALYSIS")
print("="*80)

# Store all results for comprehensive analysis
framework_results = {}

# ============================================================================
# FORCE 1: COMPETITIVE RIVALRY
# ============================================================================

print("\n" + "-"*80)
print("FORCE 1: COMPETITIVE RIVALRY")
print("-"*80)

industry_counts = df.groupby('Industry_Group').size().sort_values(ascending=False)
print("\nIndustry Distribution:")
for ind, count in industry_counts.items():
    print(f"  {ind:20s} {count:4d} companies")

def classify_rivalry(industry_group):
    count = industry_counts.get(industry_group, 0)
    if count >= 150:
        return 'High'
    elif count >= 50:
        return 'Medium'
    else:
        return 'Low'

df['Competitive_Rivalry'] = df['Industry_Group'].apply(classify_rivalry)

print("\nCompetitive Rivalry Impact on Growth Speed:")
rivalry_stats = df.groupby('Competitive_Rivalry')['Years_to_Unicorn'].agg([
    ('Count', 'count'),
    ('Mean_Years', 'mean'),
    ('Median_Years', 'median'),
    ('Std_Years', 'std')
]).round(2)
print(rivalry_stats)

# Enhanced ANOVA with effect size
groups = [df[df['Competitive_Rivalry'] == level]['Years_to_Unicorn'].dropna() 
          for level in ['Low', 'Medium', 'High'] if level in df['Competitive_Rivalry'].unique()]

if len(groups) >= 2:
    f_stat1, p_value1 = f_oneway(*groups)
    eta2_1 = eta_squared_anova(groups)
    
    print(f"\nEnhanced Statistical Analysis:")
    print(f"  F-statistic: {f_stat1:.4f}")
    print(f"  P-value: {p_value1:.4f}")
    print(f"  Effect Size (Eta-squared): {eta2_1:.4f} ({interpret_eta_squared(eta2_1)})")
    
    # Confidence intervals
    print(f"\n  Confidence Intervals (95%):")
    for level in ['Low', 'Medium', 'High']:
        if level in df['Competitive_Rivalry'].unique():
            data = df[df['Competitive_Rivalry'] == level]['Years_to_Unicorn'].dropna()
            ci = confidence_interval_mean(data)
            print(f"    {level}: [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    force1_validated = p_value1 < 0.05
    framework_results['Force1'] = {
        'validated': force1_validated,
        'p_value': p_value1,
        'effect_size': eta2_1,
        'effect_interpretation': interpret_eta_squared(eta2_1),
        'ml_feature': 'Industry features' if feature_importance is not None else None,
        'ml_importance': None
    }
else:
    force1_validated = False
    p_value1 = 1.0
    eta2_1 = 0.0
    framework_results['Force1'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}

# ============================================================================
# FORCE 2: THREAT OF NEW ENTRANTS (Barriers to Entry)
# ============================================================================

print("\n" + "-"*80)
print("FORCE 2: THREAT OF NEW ENTRANTS (BARRIERS TO ENTRY)")
print("-"*80)

def classify_barriers(investor_count):
    if investor_count >= 8:
        return 'High'
    elif investor_count >= 4:
        return 'Medium'
    else:
        return 'Low'

df['Entry_Barriers'] = df['Investor_Count'].apply(classify_barriers)

print("\nEntry Barriers Distribution:")
print(df['Entry_Barriers'].value_counts())

print("\nEntry Barriers Impact on Valuation:")
barriers_stats = df.groupby('Entry_Barriers')['Valuation ($B)'].agg([
    ('Count', 'count'),
    ('Mean_Val', 'mean'),
    ('Median_Val', 'median'),
    ('Std_Val', 'std')
]).round(2)
print(barriers_stats)

# Enhanced ANOVA with effect size
groups_val = [df[df['Entry_Barriers'] == level]['Valuation ($B)'].dropna() 
              for level in ['Low', 'Medium', 'High'] if level in df['Entry_Barriers'].unique()]

if len(groups_val) >= 2:
    f_stat2, p_value2 = f_oneway(*groups_val)
    eta2_2 = eta_squared_anova(groups_val)
    
    print(f"\nEnhanced Statistical Analysis:")
    print(f"  F-statistic: {f_stat2:.4f}")
    print(f"  P-value: {p_value2:.4f}")
    print(f"  Effect Size (Eta-squared): {eta2_2:.4f} ({interpret_eta_squared(eta2_2)})")
    
    # Confidence intervals
    print(f"\n  Confidence Intervals (95%):")
    for level in ['Low', 'Medium', 'High']:
        if level in df['Entry_Barriers'].unique():
            data = df[df['Entry_Barriers'] == level]['Valuation ($B)'].dropna()
            ci = confidence_interval_mean(data)
            print(f"    {level}: [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    # ML Integration
    ml_importance = None
    if feature_importance is not None:
        investor_features = ['Investor_Count', 'Val_per_Investor', 'Investor_Efficiency']
        for feat in investor_features:
            if feat in feature_importance_dict:
                ml_importance = feature_importance_dict[feat]
                print(f"\n  ML Integration: '{feat}' importance = {ml_importance:.4f}")
                break
    
    force2_validated = p_value2 < 0.05
    framework_results['Force2'] = {
        'validated': force2_validated,
        'p_value': p_value2,
        'effect_size': eta2_2,
        'effect_interpretation': interpret_eta_squared(eta2_2),
        'ml_feature': 'Investor_Count' if feature_importance is not None else None,
        'ml_importance': ml_importance
    }
else:
    force2_validated = False
    p_value2 = 1.0
    eta2_2 = 0.0
    framework_results['Force2'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}

# ============================================================================
# FORCE 3: BARGAINING POWER OF SUPPLIERS
# ============================================================================

print("\n" + "-"*80)
print("FORCE 3: BARGAINING POWER OF SUPPLIERS (VCs)")
print("-"*80)

supplier_stats = df.groupby('Has_Top_VC').agg({
    'Valuation ($B)': ['count', 'mean', 'median', 'std'],
    'Years_to_Unicorn': ['mean', 'median']
}).round(2)
supplier_stats.index = ['No Top VC (Low Power)', 'Top VC (High Power)']
print(supplier_stats)

# Enhanced T-test with effect size
if df['Has_Top_VC'].sum() > 0:
    top_vc_vals = df[df['Has_Top_VC'] == 1]['Valuation ($B)'].dropna()
    no_vc_vals = df[df['Has_Top_VC'] == 0]['Valuation ($B)'].dropna()
    
    if len(top_vc_vals) > 0 and len(no_vc_vals) > 0:
        t_stat3, p_value3 = ttest_ind(top_vc_vals, no_vc_vals)
        cohens_d3 = cohens_d(top_vc_vals, no_vc_vals)
        
        print(f"\nEnhanced Statistical Analysis:")
        print(f"  T-statistic: {t_stat3:.4f}")
        print(f"  P-value: {p_value3:.4f}")
        print(f"  Effect Size (Cohen's d): {cohens_d3:.4f} ({interpret_cohens_d(cohens_d3)})")
        
        # Confidence intervals
        ci_top = confidence_interval_mean(top_vc_vals)
        ci_no = confidence_interval_mean(no_vc_vals)
        print(f"\n  Confidence Intervals (95%):")
        print(f"    Top VC: [{ci_top[0]:.2f}, {ci_top[1]:.2f}]")
        print(f"    No Top VC: [{ci_no[0]:.2f}, {ci_no[1]:.2f}]")
        
        # ML Integration
        ml_importance = None
        if feature_importance is not None:
            vc_features = ['Has_Top_VC', 'VC_Quality_Score', 'Hub_x_TopVC']
            for feat in vc_features:
                if feat in feature_importance_dict:
                    ml_importance = feature_importance_dict[feat]
                    print(f"\n  ML Integration: '{feat}' importance = {ml_importance:.4f}")
                    break
        
        force3_validated = p_value3 < 0.05
        framework_results['Force3'] = {
            'validated': force3_validated,
            'p_value': p_value3,
            'effect_size': abs(cohens_d3),
            'effect_interpretation': interpret_cohens_d(cohens_d3),
            'ml_feature': 'Has_Top_VC' if feature_importance is not None else None,
            'ml_importance': ml_importance
        }
    else:
        force3_validated = False
        p_value3 = 1.0
        cohens_d3 = 0.0
        framework_results['Force3'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}
else:
    force3_validated = False
    p_value3 = 1.0
    cohens_d3 = 0.0
    framework_results['Force3'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}

# ============================================================================
# FORCE 4: BARGAINING POWER OF BUYERS
# ============================================================================

print("\n" + "-"*80)
print("FORCE 4: BARGAINING POWER OF BUYERS (CUSTOMERS)")
print("-"*80)

b2b_industries = ['Enterprise_Tech', 'AI_ML']
df['Is_B2B'] = df['Industry_Group'].isin(b2b_industries).astype(int)

buyer_stats = df.groupby('Is_B2B').agg({
    'Valuation ($B)': ['count', 'mean', 'median', 'std'],
    'Years_to_Unicorn': ['mean', 'median']
}).round(2)
buyer_stats.index = ['B2C (Low Buyer Power)', 'B2B (High Buyer Power)']
print(buyer_stats)

# Enhanced T-test
if df['Is_B2B'].sum() > 0 and df['Is_B2B'].sum() < len(df):
    b2b_vals = df[df['Is_B2B'] == 1]['Valuation ($B)'].dropna()
    b2c_vals = df[df['Is_B2B'] == 0]['Valuation ($B)'].dropna()
    
    if len(b2b_vals) > 0 and len(b2c_vals) > 0:
        t_stat4, p_value4 = ttest_ind(b2b_vals, b2c_vals)
        cohens_d4 = cohens_d(b2b_vals, b2c_vals)
        
        print(f"\nEnhanced Statistical Analysis:")
        print(f"  T-statistic: {t_stat4:.4f}")
        print(f"  P-value: {p_value4:.4f}")
        print(f"  Effect Size (Cohen's d): {cohens_d4:.4f} ({interpret_cohens_d(cohens_d4)})")
        
        force4_validated = p_value4 < 0.05
        framework_results['Force4'] = {
            'validated': force4_validated,
            'p_value': p_value4,
            'effect_size': abs(cohens_d4),
            'effect_interpretation': interpret_cohens_d(cohens_d4),
            'ml_feature': 'Industry features' if feature_importance is not None else None,
            'ml_importance': None
        }
    else:
        force4_validated = False
        p_value4 = 1.0
        cohens_d4 = 0.0
        framework_results['Force4'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}
else:
    force4_validated = False
    p_value4 = 1.0
    cohens_d4 = 0.0
    framework_results['Force4'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}

# ============================================================================
# FORCE 5: THREAT OF SUBSTITUTES
# ============================================================================

print("\n" + "-"*80)
print("FORCE 5: THREAT OF SUBSTITUTES")
print("-"*80)

substitute_stats = df.groupby('Is_Tech_Intensive').agg({
    'Years_to_Unicorn': ['count', 'mean', 'median', 'std'],
    'Valuation ($B)': ['mean', 'median']
}).round(2)
substitute_stats.index = ['High Substitute Threat', 'Low Substitute Threat']
print(substitute_stats)

# Enhanced T-test
if df['Is_Tech_Intensive'].sum() > 0 and df['Is_Tech_Intensive'].sum() < len(df):
    low_sub_vals = df[df['Is_Tech_Intensive'] == 1]['Years_to_Unicorn'].dropna()
    high_sub_vals = df[df['Is_Tech_Intensive'] == 0]['Years_to_Unicorn'].dropna()
    
    if len(low_sub_vals) > 0 and len(high_sub_vals) > 0:
        t_stat5, p_value5 = ttest_ind(low_sub_vals, high_sub_vals)
        cohens_d5 = cohens_d(low_sub_vals, high_sub_vals)
        
        print(f"\nEnhanced Statistical Analysis:")
        print(f"  T-statistic: {t_stat5:.4f}")
        print(f"  P-value: {p_value5:.4f}")
        print(f"  Effect Size (Cohen's d): {cohens_d5:.4f} ({interpret_cohens_d(cohens_d5)})")
        
        force5_validated = p_value5 < 0.05
        framework_results['Force5'] = {
            'validated': force5_validated,
            'p_value': p_value5,
            'effect_size': abs(cohens_d5),
            'effect_interpretation': interpret_cohens_d(cohens_d5),
            'ml_feature': 'Is_Tech_Intensive' if feature_importance is not None else None,
            'ml_importance': feature_importance_dict.get('Is_Tech_Intensive', None) if feature_importance is not None else None
        }
    else:
        force5_validated = False
        p_value5 = 1.0
        cohens_d5 = 0.0
        framework_results['Force5'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}
else:
    force5_validated = False
    p_value5 = 1.0
    cohens_d5 = 0.0
    framework_results['Force5'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}

# ============================================================================
# ENHANCED RBV ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PART 2: ENHANCED RESOURCE-BASED VIEW (RBV) ANALYSIS")
print("="*80)

# RARE RESOURCE: Geographic Advantage
print("\nRARE RESOURCE: Geographic Location (Tech Hub)")
geo_analysis = df.groupby('Is_Tech_Hub').agg({
    'Valuation ($B)': ['count', 'mean', 'median', 'std'],
    'Years_to_Unicorn': ['mean', 'median']
}).round(2)
geo_analysis.index = ['Non-Tech Hub', 'Tech Hub']
print(geo_analysis)

tech_hub_vals = df[df['Is_Tech_Hub'] == 1]['Valuation ($B)'].dropna()
non_tech_vals = df[df['Is_Tech_Hub'] == 0]['Valuation ($B)'].dropna()

if len(tech_hub_vals) > 0 and len(non_tech_vals) > 0:
    t_stat_rbv1, p_value_rbv1 = ttest_ind(tech_hub_vals, non_tech_vals)
    cohens_d_rbv1 = cohens_d(tech_hub_vals, non_tech_vals)
    
    print(f"\nEnhanced Statistical Analysis:")
    print(f"  T-statistic: {t_stat_rbv1:.4f}")
    print(f"  P-value: {p_value_rbv1:.4f}")
    print(f"  Effect Size (Cohen's d): {cohens_d_rbv1:.4f} ({interpret_cohens_d(cohens_d_rbv1)})")
    
    # ML Integration
    ml_importance = None
    if feature_importance is not None:
        geo_features = ['Is_Tech_Hub', 'Is_Silicon_Valley', 'Geo_Advantage']
        for feat in geo_features:
            if feat in feature_importance_dict:
                ml_importance = feature_importance_dict[feat]
                print(f"\n  ML Integration: '{feat}' importance = {ml_importance:.4f}")
                break
    
    rbv_geo_validated = p_value_rbv1 < 0.05
    framework_results['RBV_Geo'] = {
        'validated': rbv_geo_validated,
        'p_value': p_value_rbv1,
        'effect_size': abs(cohens_d_rbv1),
        'effect_interpretation': interpret_cohens_d(cohens_d_rbv1),
        'ml_feature': 'Is_Tech_Hub' if feature_importance is not None else None,
        'ml_importance': ml_importance
    }
else:
    rbv_geo_validated = False
    p_value_rbv1 = 1.0
    cohens_d_rbv1 = 0.0
    framework_results['RBV_Geo'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}

# VALUABLE RESOURCE: VC Network
print("\nVALUABLE RESOURCE: Top-Tier VC Network")
vc_analysis = df.groupby('Has_Top_VC').agg({
    'Valuation ($B)': ['count', 'mean', 'median'],
    'Years_to_Unicorn': ['mean', 'median', 'std']
}).round(2)
vc_analysis.index = ['No Top VC', 'Has Top VC']
print(vc_analysis)

top_vc_speed = df[df['Has_Top_VC'] == 1]['Years_to_Unicorn'].dropna()
no_vc_speed = df[df['Has_Top_VC'] == 0]['Years_to_Unicorn'].dropna()

if len(top_vc_speed) > 0 and len(no_vc_speed) > 0:
    t_stat_rbv2, p_value_rbv2 = ttest_ind(top_vc_speed, no_vc_speed)
    cohens_d_rbv2 = cohens_d(top_vc_speed, no_vc_speed)
    
    print(f"\nEnhanced Statistical Analysis:")
    print(f"  T-statistic: {t_stat_rbv2:.4f}")
    print(f"  P-value: {p_value_rbv2:.4f}")
    print(f"  Effect Size (Cohen's d): {cohens_d_rbv2:.4f} ({interpret_cohens_d(cohens_d_rbv2)})")
    
    # ML Integration
    ml_importance = None
    if feature_importance is not None:
        vc_features = ['Has_Top_VC', 'VC_Quality_Score', 'Investors_x_Year', 'TopVC_x_Year']
        for feat in vc_features:
            if feat in feature_importance_dict:
                ml_importance = feature_importance_dict[feat]
                print(f"\n  ML Integration: '{feat}' importance = {ml_importance:.4f}")
                break
    
    rbv_vc_validated = p_value_rbv2 < 0.05
    framework_results['RBV_VC'] = {
        'validated': rbv_vc_validated,
        'p_value': p_value_rbv2,
        'effect_size': abs(cohens_d_rbv2),
        'effect_interpretation': interpret_cohens_d(cohens_d_rbv2),
        'ml_feature': 'Has_Top_VC' if feature_importance is not None else None,
        'ml_importance': ml_importance
    }
else:
    rbv_vc_validated = False
    p_value_rbv2 = 1.0
    cohens_d_rbv2 = 0.0
    framework_results['RBV_VC'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}

# ============================================================================
# NETWORK EFFECTS THEORY
# ============================================================================

print("\n" + "="*80)
print("PART 3: NETWORK EFFECTS THEORY")
print("="*80)

platform_keywords = ['marketplace', 'platform', 'social', 'network']
def is_platform(industry):
    ind_lower = str(industry).lower()
    return any(keyword in ind_lower for keyword in platform_keywords)

df['Is_Platform'] = df['Industry'].apply(is_platform).astype(int)
print(f"\nPlatform businesses: {df['Is_Platform'].sum()} companies")

if df['Is_Platform'].sum() > 0:
    network_stats = df.groupby('Is_Platform').agg({
        'Valuation ($B)': ['count', 'mean', 'median', 'std'],
        'Years_to_Unicorn': ['mean', 'median']
    }).round(2)
    network_stats.index = ['Non-Platform', 'Platform (Network Effects)']
    print(network_stats)
    
    platform_vals = df[df['Is_Platform'] == 1]['Valuation ($B)'].dropna()
    non_platform_vals = df[df['Is_Platform'] == 0]['Valuation ($B)'].dropna()
    
    if len(platform_vals) > 0 and len(non_platform_vals) > 0:
        t_stat_net, p_value_net = ttest_ind(platform_vals, non_platform_vals)
        cohens_d_net = cohens_d(platform_vals, non_platform_vals)
        
        print(f"\nEnhanced Statistical Analysis:")
        print(f"  T-statistic: {t_stat_net:.4f}")
        print(f"  P-value: {p_value_net:.4f}")
        print(f"  Effect Size (Cohen's d): {cohens_d_net:.4f} ({interpret_cohens_d(cohens_d_net)})")
        
        network_validated = p_value_net < 0.05
        framework_results['Network'] = {
            'validated': network_validated,
            'p_value': p_value_net,
            'effect_size': abs(cohens_d_net),
            'effect_interpretation': interpret_cohens_d(cohens_d_net),
            'ml_feature': None,
            'ml_importance': None
        }
    else:
        network_validated = False
        p_value_net = 1.0
        cohens_d_net = 0.0
        framework_results['Network'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}
else:
    network_validated = False
    p_value_net = 1.0
    cohens_d_net = 0.0
    framework_results['Network'] = {'validated': False, 'p_value': 1.0, 'effect_size': 0.0}

# ============================================================================
# ENHANCED VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING ENHANCED VISUALIZATIONS")
print("="*80)

os.makedirs('output/visualizations', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)
os.makedirs('output/data', exist_ok=True)

# Create comprehensive visualization dashboard
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
fig.suptitle('Enhanced Strategic Framework Analysis Dashboard', fontsize=18, fontweight='bold')

# Row 1: Porter's Five Forces with Box Plots
# Force 1: Rivalry
ax1 = fig.add_subplot(gs[0, 0])
rivalry_levels = [level for level in ['Low', 'Medium', 'High'] if level in df['Competitive_Rivalry'].unique()]
rivalry_data_list = [df[df['Competitive_Rivalry'] == level]['Years_to_Unicorn'].dropna() 
                     for level in rivalry_levels]
if rivalry_data_list and len(rivalry_data_list) == len(rivalry_levels):
    bp1 = ax1.boxplot(rivalry_data_list, labels=rivalry_levels, patch_artist=True)
    for patch, color in zip(bp1['boxes'], ['green', 'orange', 'red']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title('Force 1: Competitive Rivalry\n(Box Plot)', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Years to Unicorn')
    ax1.grid(axis='y', alpha=0.3)

# Force 2: Entry Barriers
ax2 = fig.add_subplot(gs[0, 1])
barriers_levels = [level for level in ['Low', 'Medium', 'High'] if level in df['Entry_Barriers'].unique()]
barriers_data_list = [df[df['Entry_Barriers'] == level]['Valuation ($B)'].dropna() 
                      for level in barriers_levels]
if barriers_data_list and len(barriers_data_list) == len(barriers_levels):
    bp2 = ax2.boxplot(barriers_data_list, labels=barriers_levels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['red', 'orange', 'green']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title('Force 2: Entry Barriers\n(Box Plot)', fontweight='bold', fontsize=10)
    ax2.set_ylabel('Valuation ($B)')
    ax2.grid(axis='y', alpha=0.3)

# Force 3: Supplier Power
ax3 = fig.add_subplot(gs[0, 2])
supplier_data_list = [df[df['Has_Top_VC'] == 0]['Valuation ($B)'].dropna(),
                      df[df['Has_Top_VC'] == 1]['Valuation ($B)'].dropna()]
bp3 = ax3.boxplot(supplier_data_list, labels=['No Top VC', 'Top VC'], patch_artist=True)
for patch, color in zip(bp3['boxes'], ['lightgray', 'darkblue']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_title('Force 3: Supplier Power\n(Box Plot)', fontweight='bold', fontsize=10)
ax3.set_ylabel('Valuation ($B)')
ax3.grid(axis='y', alpha=0.3)

# Force 4: Buyer Power
ax4 = fig.add_subplot(gs[0, 3])
buyer_data_list = [df[df['Is_B2B'] == 0]['Valuation ($B)'].dropna(),
                   df[df['Is_B2B'] == 1]['Valuation ($B)'].dropna()]
bp4 = ax4.boxplot(buyer_data_list, labels=['B2C', 'B2B'], patch_artist=True)
for patch, color in zip(bp4['boxes'], ['lightcoral', 'steelblue']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_title('Force 4: Buyer Power\n(Box Plot)', fontweight='bold', fontsize=10)
ax4.set_ylabel('Valuation ($B)')
ax4.grid(axis='y', alpha=0.3)

# Row 2: RBV and Network Effects
# Force 5: Substitutes
ax5 = fig.add_subplot(gs[1, 0])
substitute_data_list = [df[df['Is_Tech_Intensive'] == 0]['Years_to_Unicorn'].dropna(),
                        df[df['Is_Tech_Intensive'] == 1]['Years_to_Unicorn'].dropna()]
bp5 = ax5.boxplot(substitute_data_list, labels=['High Threat', 'Low Threat'], patch_artist=True)
for patch, color in zip(bp5['boxes'], ['orange', 'green']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_title('Force 5: Substitutes\n(Box Plot)', fontweight='bold', fontsize=10)
ax5.set_ylabel('Years to Unicorn')
ax5.grid(axis='y', alpha=0.3)

# RBV: Geographic
ax6 = fig.add_subplot(gs[1, 1])
geo_data_list = [df[df['Is_Tech_Hub'] == 0]['Valuation ($B)'].dropna(),
                 df[df['Is_Tech_Hub'] == 1]['Valuation ($B)'].dropna()]
bp6 = ax6.boxplot(geo_data_list, labels=['Non-Hub', 'Tech Hub'], patch_artist=True)
for patch, color in zip(bp6['boxes'], ['lightcoral', 'lightblue']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_title('RBV: Geographic Resource\n(Box Plot)', fontweight='bold', fontsize=10)
ax6.set_ylabel('Valuation ($B)')
ax6.grid(axis='y', alpha=0.3)

# RBV: VC Network
ax7 = fig.add_subplot(gs[1, 2])
vc_data_list = [df[df['Has_Top_VC'] == 0]['Years_to_Unicorn'].dropna(),
                df[df['Has_Top_VC'] == 1]['Years_to_Unicorn'].dropna()]
bp7 = ax7.boxplot(vc_data_list, labels=['No Top VC', 'Top VC'], patch_artist=True)
for patch, color in zip(bp7['boxes'], ['lightgray', 'gold']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax7.set_title('RBV: VC Network Resource\n(Box Plot)', fontweight='bold', fontsize=10)
ax7.set_ylabel('Years to Unicorn')
ax7.grid(axis='y', alpha=0.3)

# Network Effects
ax8 = fig.add_subplot(gs[1, 3])
if df['Is_Platform'].sum() > 0:
    network_data_list = [df[df['Is_Platform'] == 0]['Valuation ($B)'].dropna(),
                        df[df['Is_Platform'] == 1]['Valuation ($B)'].dropna()]
    bp8 = ax8.boxplot(network_data_list, labels=['Non-Platform', 'Platform'], patch_artist=True)
    for patch, color in zip(bp8['boxes'], ['gray', 'purple']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax8.set_title('Network Effects\n(Box Plot)', fontweight='bold', fontsize=10)
    ax8.set_ylabel('Valuation ($B)')
    ax8.grid(axis='y', alpha=0.3)
else:
    ax8.text(0.5, 0.5, 'No Platform Data', ha='center', va='center', fontsize=12)
    ax8.set_title('Network Effects', fontweight='bold', fontsize=10)

# Row 3: Effect Sizes Visualization
ax9 = fig.add_subplot(gs[2, :2])
framework_names = ['F1:Rivalry', 'F2:Entry', 'F3:Supplier', 'F4:Buyer', 'F5:Substitute', 
                   'RBV:Geo', 'RBV:VC', 'Network']
effect_sizes = [
    framework_results.get('Force1', {}).get('effect_size', 0),
    framework_results.get('Force2', {}).get('effect_size', 0),
    framework_results.get('Force3', {}).get('effect_size', 0),
    framework_results.get('Force4', {}).get('effect_size', 0),
    framework_results.get('Force5', {}).get('effect_size', 0),
    framework_results.get('RBV_Geo', {}).get('effect_size', 0),
    framework_results.get('RBV_VC', {}).get('effect_size', 0),
    framework_results.get('Network', {}).get('effect_size', 0)
]
colors_effect = ['green' if es > 0.5 else 'orange' if es > 0.2 else 'red' for es in effect_sizes]
ax9.barh(range(len(framework_names)), effect_sizes, color=colors_effect, edgecolor='black', alpha=0.7)
ax9.set_yticks(range(len(framework_names)))
ax9.set_yticklabels(framework_names)
ax9.set_xlabel('Effect Size')
ax9.set_title('Effect Sizes by Framework', fontweight='bold', fontsize=12)
ax9.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Small (0.2)')
ax9.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Medium (0.5)')
ax9.axvline(x=0.8, color='darkgreen', linestyle='--', alpha=0.5, label='Large (0.8)')
ax9.legend(fontsize=8)
ax9.grid(axis='x', alpha=0.3)

# Row 3: ML Integration Visualization
ax10 = fig.add_subplot(gs[2, 2:])
if feature_importance is not None:
    ml_features = []
    ml_importances = []
    for key, result in framework_results.items():
        if result.get('ml_importance') is not None:
            ml_features.append(key)
            ml_importances.append(result['ml_importance'])
    
    if ml_importances:
        ax10.barh(range(len(ml_features)), ml_importances, color='steelblue', edgecolor='black', alpha=0.7)
        ax10.set_yticks(range(len(ml_features)))
        ax10.set_yticklabels(ml_features)
        ax10.set_xlabel('ML Feature Importance')
        ax10.set_title('ML Model Feature Importance\n(Aligned with Frameworks)', fontweight='bold', fontsize=12)
        ax10.grid(axis='x', alpha=0.3)
    else:
        ax10.text(0.5, 0.5, 'No ML Integration Data', ha='center', va='center', fontsize=12)
        ax10.set_title('ML Integration', fontweight='bold', fontsize=12)
else:
    ax10.text(0.5, 0.5, 'ML Feature Importance\nNot Available', ha='center', va='center', fontsize=12)
    ax10.set_title('ML Integration', fontweight='bold', fontsize=12)

# Row 4: Validation Summary with P-values
ax11 = fig.add_subplot(gs[3, :2])
validation_scores = [
    framework_results.get('Force1', {}).get('validated', False),
    framework_results.get('Force2', {}).get('validated', False),
    framework_results.get('Force3', {}).get('validated', False),
    framework_results.get('Force4', {}).get('validated', False),
    framework_results.get('Force5', {}).get('validated', False),
    framework_results.get('RBV_Geo', {}).get('validated', False),
    framework_results.get('RBV_VC', {}).get('validated', False),
    framework_results.get('Network', {}).get('validated', False)
]
p_values = [
    framework_results.get('Force1', {}).get('p_value', 1.0),
    framework_results.get('Force2', {}).get('p_value', 1.0),
    framework_results.get('Force3', {}).get('p_value', 1.0),
    framework_results.get('Force4', {}).get('p_value', 1.0),
    framework_results.get('Force5', {}).get('p_value', 1.0),
    framework_results.get('RBV_Geo', {}).get('p_value', 1.0),
    framework_results.get('RBV_VC', {}).get('p_value', 1.0),
    framework_results.get('Network', {}).get('p_value', 1.0)
]
colors_val = ['green' if s else 'red' for s in validation_scores]
ax11.bar(range(len(framework_names)), [1 if s else 0 for s in validation_scores], 
        color=colors_val, edgecolor='black', alpha=0.7)
ax11.set_xticks(range(len(framework_names)))
ax11.set_xticklabels(framework_names, rotation=45, ha='right', fontsize=9)
ax11.set_ylabel('Validated (1=Yes, 0=No)')
ax11.set_title('Validation Summary', fontweight='bold', fontsize=12)
ax11.set_ylim(0, 1.2)
ax11.grid(axis='y', alpha=0.3)

# Row 4: P-values Visualization
ax12 = fig.add_subplot(gs[3, 2:])
ax12.barh(range(len(framework_names)), [-np.log10(p) if p > 0 else 0 for p in p_values], 
         color=colors_val, edgecolor='black', alpha=0.7)
ax12.set_yticks(range(len(framework_names)))
ax12.set_yticklabels(framework_names)
ax12.set_xlabel('-log10(P-value)')
ax12.set_title('Statistical Significance\n(-log10 P-value)', fontweight='bold', fontsize=12)
ax12.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')
ax12.legend(fontsize=8)
ax12.grid(axis='x', alpha=0.3)

plt.savefig('output/visualizations/enhanced_strategic_framework_dashboard.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: output/visualizations/enhanced_strategic_framework_dashboard.png")
plt.close()

# ============================================================================
# COMPREHENSIVE RESULTS TABLE
# ============================================================================

print("\n" + "="*80)
print("[SUCCESS] ENHANCED ANALYSIS COMPLETE")
print("="*80)

validation_results = pd.DataFrame({
    'Framework': [
        'Porter Force 1: Competitive Rivalry',
        'Porter Force 2: Threat of New Entrants',
        'Porter Force 3: Bargaining Power of Suppliers',
        'Porter Force 4: Bargaining Power of Buyers',
        'Porter Force 5: Threat of Substitutes',
        'RBV: Geographic Resource',
        'RBV: VC Network Resource',
        'Network Effects Theory'
    ],
    'Hypothesis': [
        'Low rivalry -> Faster growth',
        'High barriers -> Higher valuation',
        'VC supplier power -> Affects outcomes',
        'Buyer power -> Affects valuation',
        'Low substitutes -> Affects growth',
        'Tech hub -> Advantage',
        'Top VC -> Faster growth',
        'Platform/Network -> Higher valuation'
    ],
    'P-Value': [
        framework_results.get('Force1', {}).get('p_value', 1.0),
        framework_results.get('Force2', {}).get('p_value', 1.0),
        framework_results.get('Force3', {}).get('p_value', 1.0),
        framework_results.get('Force4', {}).get('p_value', 1.0),
        framework_results.get('Force5', {}).get('p_value', 1.0),
        framework_results.get('RBV_Geo', {}).get('p_value', 1.0),
        framework_results.get('RBV_VC', {}).get('p_value', 1.0),
        framework_results.get('Network', {}).get('p_value', 1.0)
    ],
    'Effect_Size': [
        framework_results.get('Force1', {}).get('effect_size', 0.0),
        framework_results.get('Force2', {}).get('effect_size', 0.0),
        framework_results.get('Force3', {}).get('effect_size', 0.0),
        framework_results.get('Force4', {}).get('effect_size', 0.0),
        framework_results.get('Force5', {}).get('effect_size', 0.0),
        framework_results.get('RBV_Geo', {}).get('effect_size', 0.0),
        framework_results.get('RBV_VC', {}).get('effect_size', 0.0),
        framework_results.get('Network', {}).get('effect_size', 0.0)
    ],
    'Effect_Interpretation': [
        framework_results.get('Force1', {}).get('effect_interpretation', 'N/A'),
        framework_results.get('Force2', {}).get('effect_interpretation', 'N/A'),
        framework_results.get('Force3', {}).get('effect_interpretation', 'N/A'),
        framework_results.get('Force4', {}).get('effect_interpretation', 'N/A'),
        framework_results.get('Force5', {}).get('effect_interpretation', 'N/A'),
        framework_results.get('RBV_Geo', {}).get('effect_interpretation', 'N/A'),
        framework_results.get('RBV_VC', {}).get('effect_interpretation', 'N/A'),
        framework_results.get('Network', {}).get('effect_interpretation', 'N/A')
    ],
    'ML_Feature': [
        framework_results.get('Force1', {}).get('ml_feature', 'N/A'),
        framework_results.get('Force2', {}).get('ml_feature', 'N/A'),
        framework_results.get('Force3', {}).get('ml_feature', 'N/A'),
        framework_results.get('Force4', {}).get('ml_feature', 'N/A'),
        framework_results.get('Force5', {}).get('ml_feature', 'N/A'),
        framework_results.get('RBV_Geo', {}).get('ml_feature', 'N/A'),
        framework_results.get('RBV_VC', {}).get('ml_feature', 'N/A'),
        framework_results.get('Network', {}).get('ml_feature', 'N/A')
    ],
    'ML_Importance': [
        framework_results.get('Force1', {}).get('ml_importance', None),
        framework_results.get('Force2', {}).get('ml_importance', None),
        framework_results.get('Force3', {}).get('ml_importance', None),
        framework_results.get('Force4', {}).get('ml_importance', None),
        framework_results.get('Force5', {}).get('ml_importance', None),
        framework_results.get('RBV_Geo', {}).get('ml_importance', None),
        framework_results.get('RBV_VC', {}).get('ml_importance', None),
        framework_results.get('Network', {}).get('ml_importance', None)
    ],
    'Status': [
        '[VALIDATED]' if framework_results.get('Force1', {}).get('validated', False) else '[NOT VALIDATED]',
        '[VALIDATED]' if framework_results.get('Force2', {}).get('validated', False) else '[NOT VALIDATED]',
        '[VALIDATED]' if framework_results.get('Force3', {}).get('validated', False) else '[NOT VALIDATED]',
        '[VALIDATED]' if framework_results.get('Force4', {}).get('validated', False) else '[NOT VALIDATED]',
        '[VALIDATED]' if framework_results.get('Force5', {}).get('validated', False) else '[NOT VALIDATED]',
        '[VALIDATED]' if framework_results.get('RBV_Geo', {}).get('validated', False) else '[NOT VALIDATED]',
        '[VALIDATED]' if framework_results.get('RBV_VC', {}).get('validated', False) else '[NOT VALIDATED]',
        '[VALIDATED]' if framework_results.get('Network', {}).get('validated', False) else '[NOT VALIDATED]'
    ]
})

print("\n", validation_results.to_string(index=False))

validation_results.to_csv('output/data/enhanced_theoretical_validation_results.csv', index=False)
print("\n[OK] Saved: output/data/enhanced_theoretical_validation_results.csv")

validated_count = sum(validation_results['Status'].str.contains('\\[VALIDATED\\]', na=False))
print(f"\n[Validation Score] {validated_count}/8 frameworks empirically supported ({validated_count/8*100:.1f}%)")

# ============================================================================
# STRATEGIC RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("STRATEGIC RECOMMENDATIONS")
print("="*80)

validated_frameworks = validation_results[validation_results['Status'].str.contains('VALIDATED', na=False)]

recommendations = []

if len(validated_frameworks) > 0:
    print("\n[VALIDATED FRAMEWORKS - ACTIONABLE INSIGHTS]")
    
    for idx, row in validated_frameworks.iterrows():
        framework = row['Framework']
        effect = row['Effect_Interpretation']
        ml_imp = row['ML_Importance']
        
        print(f"\n{framework}:")
        print(f"  Effect Size: {effect}")
        if pd.notna(ml_imp):
            print(f"  ML Confirmation: Feature importance = {ml_imp:.4f}")
        
        # Generate specific recommendations
        if 'Entry Barriers' in framework:
            recommendations.append({
                'framework': framework,
                'recommendation': 'Build high entry barriers through capital intensity and investor network',
                'action': 'Seek multiple rounds of funding from diverse investors to create competitive moat',
                'priority': 'HIGH'
            })
        elif 'VC Network' in framework:
            recommendations.append({
                'framework': framework,
                'recommendation': 'Partner with top-tier VCs for faster growth',
                'action': 'Target Sequoia, a16z, Tiger Global, or other top VCs for funding rounds',
                'priority': 'HIGH'
            })
else:
    print("\n[NO FRAMEWORKS VALIDATED]")
    print("  Consider alternative strategic approaches or data limitations")

# Create comprehensive report
summary_report = f"""
# Enhanced Strategic Framework Analysis Report

## Executive Summary

Comprehensive analysis of strategic frameworks with:
- **Effect Sizes**: Quantified magnitude of effects (Cohen's d, Eta-squared)
- **ML Integration**: Alignment with machine learning model findings
- **Enhanced Visualizations**: Box plots, effect sizes, statistical significance
- **Strategic Recommendations**: Actionable insights based on validated frameworks

**Validation Rate**: {validated_count}/8 frameworks ({validated_count/8*100:.1f}%)

## Detailed Framework Analysis

{validation_results.to_string(index=False)}

## Key Findings

### Validated Frameworks ({validated_count})

{chr(10).join(['- **' + row['Framework'] + '**: Effect size = ' + str(row['Effect_Size']) + ' (' + row['Effect_Interpretation'] + ')' + 
              (f', ML importance = {row["ML_Importance"]:.4f}' if pd.notna(row['ML_Importance']) else '') 
              for idx, row in validated_frameworks.iterrows()])}

### Non-Validated Frameworks ({8 - validated_count})

{chr(10).join(['- **' + row['Framework'] + '**: P-value = ' + f'{row["P-Value"]:.4f}' + ', Effect size = ' + f'{row["Effect_Size"]:.4f}' 
              for idx, row in validation_results[~validation_results['Status'].str.contains('VALIDATED', na=False)].iterrows()])}

## Strategic Recommendations

### High Priority Actions

{chr(10).join(['1. **' + rec['framework'] + '**' + chr(10) + 
              '   Recommendation: ' + rec['recommendation'] + chr(10) +
              '   Action: ' + rec['action'] 
              for rec in recommendations if rec['priority'] == 'HIGH'])}

### ML Model Alignment

The following frameworks align with ML model feature importance:
{chr(10).join(['- ' + row['Framework'] + ': ' + str(row['ML_Feature']) + ' (importance = ' + f'{row["ML_Importance"]:.4f}' + ')' 
              for idx, row in validation_results.iterrows() if pd.notna(row['ML_Importance'])])}

## Effect Size Interpretation

- **Large Effect (>= 0.8)**: Strong practical significance
- **Medium Effect (0.5-0.8)**: Moderate practical significance  
- **Small Effect (0.2-0.5)**: Weak but meaningful effect
- **Negligible (< 0.2)**: Minimal practical significance

## Statistical Significance vs. Practical Significance

Some frameworks may show statistical significance (p < 0.05) but small effect sizes, indicating:
- Effect is real but may not be practically meaningful
- Consider both p-value AND effect size when making strategic decisions

## Files Generated

- `output/visualizations/enhanced_strategic_framework_dashboard.png` - Comprehensive visualization dashboard
- `output/data/enhanced_theoretical_validation_results.csv` - Detailed results with effect sizes
- `output/reports/ENHANCED_STRATEGIC_FRAMEWORK_ANALYSIS.md` - This comprehensive report

---

**Analysis Date**: 2025
**Status**: Enhanced Analysis Complete
**Validation Rate**: {validated_count}/8 ({validated_count/8*100:.1f}%)
"""

with open('output/reports/ENHANCED_STRATEGIC_FRAMEWORK_ANALYSIS.md', 'w', encoding='utf-8') as f:
    f.write(summary_report)
print("[OK] Saved: output/reports/ENHANCED_STRATEGIC_FRAMEWORK_ANALYSIS.md")

print("\n" + "="*80)
print("[SUCCESS] ENHANCED STRATEGIC FRAMEWORK ANALYSIS COMPLETE")
print("="*80)
print("\n[OK] output/visualizations/enhanced_strategic_framework_dashboard.png")
print("[OK] output/data/enhanced_theoretical_validation_results.csv")
print("[OK] output/reports/ENHANCED_STRATEGIC_FRAMEWORK_ANALYSIS.md")
print("\n[SUCCESS] Ready for strategic decision-making!")

