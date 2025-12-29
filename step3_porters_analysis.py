"""
Step 3: Strategic Framework Analysis
OBJECTIVE 4: Theoretical Validation
Porter's Five Forces (All 5 Separate) + Resource-Based View + Network Effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle

plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

print("="*80)
print("OBJECTIVE 4: THEORETICAL VALIDATION")
print("Porter's Five Forces | Resource-Based View | Network Effects")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['df_full']
feature_importance = pd.read_csv('feature_importance_rankings.csv')

print(f"✓ Loaded {len(df)} companies")

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
# PORTER'S FIVE FORCES ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PART 1: PORTER'S FIVE FORCES FRAMEWORK (ALL 5 FORCES SEPARATE)")
print("="*80)

# ============================================================================
# FORCE 1: COMPETITIVE RIVALRY
# ============================================================================

print("\n" + "-"*80)
print("FORCE 1: COMPETITIVE RIVALRY")
print("-"*80)

# Count unicorns per industry (proxy for competition intensity)
industry_counts = df.groupby('Industry_Group').size().sort_values(ascending=False)
print("\nIndustry Distribution:")
for ind, count in industry_counts.items():
    print(f"  {ind:20s} {count:4d} companies")

# Classify rivalry intensity
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
    ('Median_Years', 'median')
]).round(2)
print(rivalry_stats)

# ANOVA test
groups = [df[df['Competitive_Rivalry'] == level]['Years_to_Unicorn'].dropna() 
          for level in ['Low', 'Medium', 'High'] if level in df['Competitive_Rivalry'].unique()]

if len(groups) >= 2:
    f_stat1, p_value1 = stats.f_oneway(*groups)
    print(f"\nStatistical Test (ANOVA):")
    print(f"  F-statistic: {f_stat1:.4f}")
    print(f"  P-value: {p_value1:.4f}")
    if p_value1 < 0.05:
        print(f"  ✅ SIGNIFICANT: Rivalry intensity affects growth speed")
        force1_validated = True
    else:
        print(f"  ⚠️  Not significant")
        force1_validated = False
else:
    force1_validated = False
    p_value1 = 1.0

# ============================================================================
# FORCE 2: THREAT OF NEW ENTRANTS (Barriers to Entry)
# ============================================================================

print("\n" + "-"*80)
print("FORCE 2: THREAT OF NEW ENTRANTS (BARRIERS TO ENTRY)")
print("-"*80)

# Proxy: High investor count = High capital intensity = High barriers
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
    ('Median_Val', 'median')
]).round(2)
print(barriers_stats)

# ANOVA test
groups_val = [df[df['Entry_Barriers'] == level]['Valuation ($B)'].dropna() 
              for level in ['Low', 'Medium', 'High'] if level in df['Entry_Barriers'].unique()]

if len(groups_val) >= 2:
    f_stat2, p_value2 = stats.f_oneway(*groups_val)
    print(f"\nStatistical Test (ANOVA):")
    print(f"  F-statistic: {f_stat2:.4f}")
    print(f"  P-value: {p_value2:.4f}")
    if p_value2 < 0.05:
        print(f"  ✅ SIGNIFICANT: Entry barriers affect valuation")
        force2_validated = True
    else:
        print(f"  ⚠️  Not significant")
        force2_validated = False
else:
    force2_validated = False
    p_value2 = 1.0

# ============================================================================
# FORCE 3: BARGAINING POWER OF SUPPLIERS
# ============================================================================

print("\n" + "-"*80)
print("FORCE 3: BARGAINING POWER OF SUPPLIERS")
print("-"*80)

# Proxy: Top-tier VCs have high bargaining power (they're selective)
# Companies with top VCs may face stronger supplier (VC) power
print("\nAnalyzing VC Bargaining Power:")
print("(Top-tier VCs = High supplier power; They choose selectively)")

supplier_stats = df.groupby('Has_Top_VC').agg({
    'Valuation ($B)': ['count', 'mean', 'median'],
    'Years_to_Unicorn': ['mean', 'median']
}).round(2)
supplier_stats.index = ['No Top VC (Low Power)', 'Top VC (High Power)']
print(supplier_stats)

# T-test: Does VC supplier power affect outcomes?
if df['Has_Top_VC'].sum() > 0:
    top_vc_vals = df[df['Has_Top_VC'] == 1]['Valuation ($B)'].dropna()
    no_vc_vals = df[df['Has_Top_VC'] == 0]['Valuation ($B)'].dropna()
    
    if len(top_vc_vals) > 0 and len(no_vc_vals) > 0:
        t_stat3, p_value3 = stats.ttest_ind(top_vc_vals, no_vc_vals)
        print(f"\nStatistical Test (T-Test on Valuation):")
        print(f"  T-statistic: {t_stat3:.4f}")
        print(f"  P-value: {p_value3:.4f}")
        if p_value3 < 0.05:
            print(f"  ✅ SIGNIFICANT: VC supplier power affects outcomes")
            force3_validated = True
        else:
            print(f"  ⚠️  Not significant")
            force3_validated = False
    else:
        force3_validated = False
        p_value3 = 1.0
else:
    force3_validated = False
    p_value3 = 1.0

# ============================================================================
# FORCE 4: BARGAINING POWER OF BUYERS (Customers)
# ============================================================================

print("\n" + "-"*80)
print("FORCE 4: BARGAINING POWER OF BUYERS (CUSTOMERS)")
print("-"*80)

# Proxy: Platform businesses have LOW buyer power (network effects create lock-in)
# B2B vs B2C: B2B customers typically have more bargaining power

# Identify B2B companies (Enterprise Tech, SaaS)
b2b_industries = ['Enterprise_Tech', 'AI_ML']
df['Is_B2B'] = df['Industry_Group'].isin(b2b_industries).astype(int)

print("\nBuyer Power Analysis:")
print("(B2B = High buyer power, B2C = Low buyer power)")

buyer_stats = df.groupby('Is_B2B').agg({
    'Valuation ($B)': ['count', 'mean', 'median'],
    'Years_to_Unicorn': ['mean', 'median']
}).round(2)
buyer_stats.index = ['B2C (Low Buyer Power)', 'B2B (High Buyer Power)']
print(buyer_stats)

# T-test
if df['Is_B2B'].sum() > 0 and df['Is_B2B'].sum() < len(df):
    b2b_vals = df[df['Is_B2B'] == 1]['Valuation ($B)'].dropna()
    b2c_vals = df[df['Is_B2B'] == 0]['Valuation ($B)'].dropna()
    
    if len(b2b_vals) > 0 and len(b2c_vals) > 0:
        t_stat4, p_value4 = stats.ttest_ind(b2b_vals, b2c_vals)
        print(f"\nStatistical Test (T-Test on Valuation):")
        print(f"  T-statistic: {t_stat4:.4f}")
        print(f"  P-value: {p_value4:.4f}")
        if p_value4 < 0.05:
            print(f"  ✅ SIGNIFICANT: Buyer power affects outcomes")
            force4_validated = True
        else:
            print(f"  ⚠️  Not significant")
            force4_validated = False
    else:
        force4_validated = False
        p_value4 = 1.0
else:
    force4_validated = False
    p_value4 = 1.0

# ============================================================================
# FORCE 5: THREAT OF SUBSTITUTES
# ============================================================================

print("\n" + "-"*80)
print("FORCE 5: THREAT OF SUBSTITUTES")
print("-"*80)

# Proxy: Tech-intensive, differentiated businesses have LOW substitute threat
# Commoditized businesses have HIGH substitute threat

print("\nSubstitute Threat Analysis:")
print("(High tech differentiation = Low substitute threat)")

substitute_stats = df.groupby('Is_Tech_Intensive').agg({
    'Years_to_Unicorn': ['count', 'mean', 'median'],
    'Valuation ($B)': ['mean', 'median']
}).round(2)
substitute_stats.index = ['High Substitute Threat', 'Low Substitute Threat']
print(substitute_stats)

# T-test on growth speed
if df['Is_Tech_Intensive'].sum() > 0 and df['Is_Tech_Intensive'].sum() < len(df):
    low_sub_vals = df[df['Is_Tech_Intensive'] == 1]['Years_to_Unicorn'].dropna()
    high_sub_vals = df[df['Is_Tech_Intensive'] == 0]['Years_to_Unicorn'].dropna()
    
    if len(low_sub_vals) > 0 and len(high_sub_vals) > 0:
        t_stat5, p_value5 = stats.ttest_ind(low_sub_vals, high_sub_vals)
        print(f"\nStatistical Test (T-Test on Growth Speed):")
        print(f"  T-statistic: {t_stat5:.4f}")
        print(f"  P-value: {p_value5:.4f}")
        if p_value5 < 0.05:
            print(f"  ✅ SIGNIFICANT: Substitute threat affects growth")
            force5_validated = True
        else:
            print(f"  ⚠️  Not significant")
            force5_validated = False
    else:
        force5_validated = False
        p_value5 = 1.0
else:
    force5_validated = False
    p_value5 = 1.0

# ============================================================================
# RESOURCE-BASED VIEW (RBV) ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PART 2: RESOURCE-BASED VIEW (RBV) FRAMEWORK")
print("="*80)

# RARE RESOURCE: Geographic Advantage
print("\nRARE RESOURCE: Geographic Location (Tech Hub)")
geo_analysis = df.groupby('Is_Tech_Hub').agg({
    'Valuation ($B)': ['count', 'mean', 'median'],
    'Years_to_Unicorn': ['mean', 'median']
}).round(2)
geo_analysis.index = ['Non-Tech Hub', 'Tech Hub']
print(geo_analysis)

tech_hub_vals = df[df['Is_Tech_Hub'] == 1]['Valuation ($B)'].dropna()
non_tech_vals = df[df['Is_Tech_Hub'] == 0]['Valuation ($B)'].dropna()

if len(tech_hub_vals) > 0 and len(non_tech_vals) > 0:
    t_stat_rbv1, p_value_rbv1 = stats.ttest_ind(tech_hub_vals, non_tech_vals)
    print(f"\nT-Test (Tech Hub Effect):")
    print(f"  P-value: {p_value_rbv1:.4f}")
    print(f"  {'✅ Significant advantage' if p_value_rbv1 < 0.05 else '⚠️  Not significant'}")
    rbv_geo_validated = p_value_rbv1 < 0.05
else:
    rbv_geo_validated = False
    p_value_rbv1 = 1.0

# VALUABLE RESOURCE: VC Network
print("\nVALUABLE RESOURCE: Top-Tier VC Network")
vc_analysis = df.groupby('Has_Top_VC').agg({
    'Valuation ($B)': ['count', 'mean', 'median'],
    'Years_to_Unicorn': ['mean', 'median']
}).round(2)
vc_analysis.index = ['No Top VC', 'Has Top VC']
print(vc_analysis)

top_vc_speed = df[df['Has_Top_VC'] == 1]['Years_to_Unicorn'].dropna()
no_vc_speed = df[df['Has_Top_VC'] == 0]['Years_to_Unicorn'].dropna()

if len(top_vc_speed) > 0 and len(no_vc_speed) > 0:
    t_stat_rbv2, p_value_rbv2 = stats.ttest_ind(top_vc_speed, no_vc_speed)
    print(f"\nT-Test (VC Network Effect on Growth):")
    print(f"  P-value: {p_value_rbv2:.4f}")
    print(f"  {'✅ Significant advantage' if p_value_rbv2 < 0.05 else '⚠️  Not significant'}")
    rbv_vc_validated = p_value_rbv2 < 0.05
else:
    rbv_vc_validated = False
    p_value_rbv2 = 1.0

# ============================================================================
# NETWORK EFFECTS THEORY
# ============================================================================

print("\n" + "="*80)
print("PART 3: NETWORK EFFECTS THEORY")
print("="*80)

# Platform businesses benefit from network effects
platform_keywords = ['marketplace', 'platform', 'social', 'network']

def is_platform(industry):
    ind_lower = str(industry).lower()
    return any(keyword in ind_lower for keyword in platform_keywords)

df['Is_Platform'] = df['Industry'].apply(is_platform).astype(int)

print(f"\nPlatform businesses: {df['Is_Platform'].sum()} companies")

if df['Is_Platform'].sum() > 0:
    network_stats = df.groupby('Is_Platform').agg({
        'Valuation ($B)': ['count', 'mean', 'median'],
        'Years_to_Unicorn': ['mean', 'median']
    }).round(2)
    network_stats.index = ['Non-Platform', 'Platform (Network Effects)']
    print(network_stats)
    
    platform_vals = df[df['Is_Platform'] == 1]['Valuation ($B)'].dropna()
    non_platform_vals = df[df['Is_Platform'] == 0]['Valuation ($B)'].dropna()
    
    if len(platform_vals) > 0 and len(non_platform_vals) > 0:
        t_stat_net, p_value_net = stats.ttest_ind(platform_vals, non_platform_vals)
        print(f"\nT-Test (Network Effects):")
        print(f"  P-value: {p_value_net:.4f}")
        print(f"  {'✅ Significant advantage' if p_value_net < 0.05 else '⚠️  Not significant'}")
        network_validated = p_value_net < 0.05
    else:
        network_validated = False
        p_value_net = 1.0
else:
    network_validated = False
    p_value_net = 1.0

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Strategic Framework Analysis: Porter\'s Five Forces + RBV + Network Effects', 
             fontsize=16, fontweight='bold')

# Plot 1: Force 1 - Rivalry
ax1 = axes[0, 0]
rivalry_data = df.groupby('Competitive_Rivalry')['Years_to_Unicorn'].mean().reindex(['Low', 'Medium', 'High'])
ax1.bar(range(len(rivalry_data)), rivalry_data.values, color=['green', 'orange', 'red'], edgecolor='black')
ax1.set_xticks(range(len(rivalry_data)))
ax1.set_xticklabels(rivalry_data.index)
ax1.set_title('Force 1: Competitive Rivalry', fontweight='bold', fontsize=10)
ax1.set_ylabel('Avg Years to Unicorn')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Force 2 - Entry Barriers
ax2 = axes[0, 1]
barriers_data = df.groupby('Entry_Barriers')['Valuation ($B)'].median().reindex(['Low', 'Medium', 'High'])
ax2.bar(range(len(barriers_data)), barriers_data.values, color=['red', 'orange', 'green'], edgecolor='black')
ax2.set_xticks(range(len(barriers_data)))
ax2.set_xticklabels(barriers_data.index)
ax2.set_title('Force 2: Threat of New Entrants', fontweight='bold', fontsize=10)
ax2.set_ylabel('Median Valuation ($B)')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Force 3 - Supplier Power
ax3 = axes[0, 2]
supplier_data = df.groupby('Has_Top_VC')['Valuation ($B)'].median()
ax3.bar(['No Top VC', 'Top VC'], supplier_data.values, color=['lightgray', 'darkblue'], edgecolor='black')
ax3.set_title('Force 3: Supplier Power (VCs)', fontweight='bold', fontsize=10)
ax3.set_ylabel('Median Valuation ($B)')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Force 4 - Buyer Power
ax4 = axes[1, 0]
buyer_data = df.groupby('Is_B2B')['Valuation ($B)'].median()
ax4.bar(['B2C', 'B2B'], buyer_data.values, color=['lightcoral', 'steelblue'], edgecolor='black')
ax4.set_title('Force 4: Buyer Power', fontweight='bold', fontsize=10)
ax4.set_ylabel('Median Valuation ($B)')
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Force 5 - Substitutes
ax5 = axes[1, 1]
substitute_data = df.groupby('Is_Tech_Intensive')['Years_to_Unicorn'].mean()
ax5.bar(['High Threat', 'Low Threat'], substitute_data.values, color=['orange', 'green'], edgecolor='black')
ax5.set_title('Force 5: Threat of Substitutes', fontweight='bold', fontsize=10)
ax5.set_ylabel('Avg Years to Unicorn')
ax5.grid(axis='y', alpha=0.3)

# Plot 6: RBV - Geographic
ax6 = axes[1, 2]
geo_data = df.groupby('Is_Tech_Hub')['Valuation ($B)'].median()
ax6.bar(['Non-Hub', 'Tech Hub'], geo_data.values, color=['lightcoral', 'lightblue'], edgecolor='black')
ax6.set_title('RBV: Geographic Resource', fontweight='bold', fontsize=10)
ax6.set_ylabel('Median Valuation ($B)')
ax6.grid(axis='y', alpha=0.3)

# Plot 7: RBV - VC Network
ax7 = axes[2, 0]
vc_res_data = df.groupby('Has_Top_VC')['Years_to_Unicorn'].mean()
ax7.bar(['No Top VC', 'Top VC'], vc_res_data.values, color=['lightgray', 'gold'], edgecolor='black')
ax7.set_title('RBV: VC Network Resource', fontweight='bold', fontsize=10)
ax7.set_ylabel('Avg Years to Unicorn')
ax7.grid(axis='y', alpha=0.3)

# Plot 8: Network Effects
ax8 = axes[2, 1]
if df['Is_Platform'].sum() > 0:
    platform_data = df.groupby('Is_Platform')['Valuation ($B)'].median()
    ax8.bar(['Non-Platform', 'Platform'], platform_data.values, color=['gray', 'purple'], edgecolor='black')
    ax8.set_title('Network Effects', fontweight='bold', fontsize=10)
    ax8.set_ylabel('Median Valuation ($B)')
    ax8.grid(axis='y', alpha=0.3)
else:
    ax8.text(0.5, 0.5, 'No Platform Data', ha='center', va='center')
    ax8.set_title('Network Effects', fontweight='bold', fontsize=10)

# Plot 9: Validation Summary
ax9 = axes[2, 2]
validation_scores = [
    force1_validated, force2_validated, force3_validated,
    force4_validated, force5_validated, rbv_geo_validated,
    rbv_vc_validated, network_validated
]
framework_names = ['F1:Rivalry', 'F2:Entry', 'F3:Supplier', 
                   'F4:Buyer', 'F5:Substitute', 'RBV:Geo', 
                   'RBV:VC', 'Network']
colors = ['green' if s else 'red' for s in validation_scores]
ax9.bar(range(len(framework_names)), [1 if s else 0 for s in validation_scores], 
        color=colors, edgecolor='black')
ax9.set_xticks(range(len(framework_names)))
ax9.set_xticklabels(framework_names, rotation=45, ha='right', fontsize=8)
ax9.set_title('Validation Summary', fontweight='bold', fontsize=10)
ax9.set_ylabel('Validated (1=Yes, 0=No)')
ax9.set_ylim(0, 1.2)
ax9.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('strategic_framework_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: strategic_framework_dashboard.png")
plt.close()

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("✅ OBJECTIVE 4: THEORETICAL VALIDATION COMPLETE")
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
        'Low rivalry → Faster growth',
        'High barriers → Higher valuation',
        'VC supplier power → Affects outcomes',
        'Buyer power → Affects valuation',
        'Low substitutes → Affects growth',
        'Tech hub → Advantage',
        'Top VC → Faster growth',
        'Platform/Network → Higher valuation'
    ],
    'P-Value': [
        p_value1, p_value2, p_value3, p_value4, p_value5,
        p_value_rbv1, p_value_rbv2, p_value_net
    ],
    'Status': [
        '✅ Validated' if force1_validated else '⚠️  Not Validated',
        '✅ Validated' if force2_validated else '⚠️  Not Validated',
        '✅ Validated' if force3_validated else '⚠️  Not Validated',
        '✅ Validated' if force4_validated else '⚠️  Not Validated',
        '✅ Validated' if force5_validated else '⚠️  Not Validated',
        '✅ Validated' if rbv_geo_validated else '⚠️  Not Validated',
        '✅ Validated' if rbv_vc_validated else '⚠️  Not Validated',
        '✅ Validated' if network_validated else '⚠️  Not Validated'
    ]
})

print("\n", validation_results.to_string(index=False))

validation_results.to_csv('theoretical_validation_results.csv', index=False)
print("\n✓ Saved: theoretical_validation_results.csv")

validated_count = sum(validation_results['Status'].str.contains('✅'))
print(f"\n📊 Validation Score: {validated_count}/8 frameworks empirically supported")

print("\n" + "="*80)
print("ALL 4 OBJECTIVES COMPLETED!")
print("="*80)
print("\n✓ strategic_framework_dashboard.png")
print("✓ theoretical_validation_results.csv")
print("\n🎉 Ready for final report and presentation!")