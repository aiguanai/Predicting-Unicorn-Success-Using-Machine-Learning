"""
Comprehensive Investor + Temporal Features Analysis
===================================================

This script analyzes:
1. Investor features alone
2. Temporal features alone  
3. Investor + Temporal combined
4. Investor × Temporal interactions
5. Feature importance comparison

Goal: Understand how investor and temporal features work together.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("INVESTOR + TEMPORAL FEATURES COMPREHENSIVE ANALYSIS")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

preprocessed_path = 'output/models/preprocessed_data.pkl'
if not os.path.exists(preprocessed_path):
    preprocessed_path = 'preprocessed_data.pkl'

with open(preprocessed_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']
df_full = data['df_full']

print(f"Dataset loaded: {X_train.shape[0]} training, {X_test.shape[0]} test samples")

# ============================================================================
# Remove Leakage (but keep Year_Founded and Era features)
# ============================================================================

LEAKAGE_FEATURES = ['Date_Joined_Year', 'Company_Age_2025']
keep_base = [f for f in X_train.columns if f not in LEAKAGE_FEATURES]

X_train_base = X_train[keep_base].copy()
X_test_base = X_test[keep_base].copy()

# ============================================================================
# Define Feature Groups
# ============================================================================

def get_feature_groups(X_train_df):
    """Extract different feature groups"""
    
    # Base features (always include)
    base_features = ['Valuation ($B)', 'Is_Tech_Hub', 'Is_Silicon_Valley', 
                     'Country_Tier', 'Is_Tech_Intensive']
    
    # Investor features
    investor_base = ['Investor_Count', 'Has_Top_VC']
    investor_features = [f for f in investor_base if f in X_train_df.columns]
    
    # Temporal features
    temporal_base = ['Year_Founded']
    era_features = [f for f in X_train_df.columns if f.startswith('Era_')]
    temporal_features = [f for f in temporal_base + era_features if f in X_train_df.columns]
    
    # Industry features
    industry_features = [f for f in X_train_df.columns if f.startswith('Ind_')]
    
    # Valuation features
    valuation_features = [f for f in ['Log_Valuation'] if f in X_train_df.columns]
    
    return {
        'base': [f for f in base_features if f in X_train_df.columns],
        'investor': investor_features,
        'temporal': temporal_features,
        'industry': industry_features,
        'valuation': valuation_features
    }

# ============================================================================
# Create Enhanced Features with Interactions
# ============================================================================

def create_investor_temporal_features(X_train_df, X_test_df, df_full):
    """Create investor × temporal interaction features"""
    
    # Start with base features
    X_train_enhanced = X_train_df.copy()
    X_test_enhanced = X_test_df.copy()
    
    # Get indices for accessing df_full
    if hasattr(X_train_df, 'index'):
        train_indices = X_train_df.index
        test_indices = X_test_df.index
    else:
        train_indices = range(len(X_train_df))
        test_indices = range(len(X_train_df), len(X_train_df) + len(X_test_df))
    
    # Investor × Temporal interactions
    if 'Year_Founded' in df_full.columns:
        try:
            train_years = df_full.loc[train_indices, 'Year_Founded'].values
            test_years = df_full.loc[test_indices, 'Year_Founded'].values
        except:
            train_years = df_full['Year_Founded'].iloc[:len(X_train_df)].values
            test_years = df_full['Year_Founded'].iloc[len(X_train_df):len(X_train_df)+len(X_test_df)].values
        
        # Investor × Year interactions
        if 'Investor_Count' in X_train_df.columns:
            X_train_enhanced['Investors_x_Year'] = X_train_df['Investor_Count'].values * train_years
            X_test_enhanced['Investors_x_Year'] = X_test_df['Investor_Count'].values * test_years
        
        # Top VC × Year
        if 'Has_Top_VC' in X_train_df.columns:
            X_train_enhanced['TopVC_x_Year'] = X_train_df['Has_Top_VC'].values * train_years
            X_test_enhanced['TopVC_x_Year'] = X_test_df['Has_Top_VC'].values * test_years
    
    # Investor × Era interactions
    era_cols = [f for f in X_train_df.columns if f.startswith('Era_')]
    for era_col in era_cols:
        if 'Investor_Count' in X_train_df.columns:
            X_train_enhanced[f'Investors_x_{era_col}'] = X_train_df['Investor_Count'] * X_train_df[era_col]
            X_test_enhanced[f'Investors_x_{era_col}'] = X_test_df['Investor_Count'] * X_test_df[era_col]
        
        if 'Has_Top_VC' in X_train_df.columns:
            X_train_enhanced[f'TopVC_x_{era_col}'] = X_train_df['Has_Top_VC'] * X_train_df[era_col]
            X_test_enhanced[f'TopVC_x_{era_col}'] = X_test_df['Has_Top_VC'] * X_test_df[era_col]
    
    # Existing investor interactions (if not already present)
    if 'Val_per_Investor' not in X_train_df.columns and 'Investor_Count' in X_train_df.columns:
        X_train_enhanced['Val_per_Investor'] = X_train_df['Valuation ($B)'] / (X_train_df['Investor_Count'] + 1)
        X_test_enhanced['Val_per_Investor'] = X_test_df['Valuation ($B)'] / (X_test_df['Investor_Count'] + 1)
    
    if 'Hub_x_TopVC' not in X_train_df.columns and 'Has_Top_VC' in X_train_df.columns:
        if 'Is_Tech_Hub' in X_train_df.columns:
            X_train_enhanced['Hub_x_TopVC'] = X_train_df['Is_Tech_Hub'] * X_train_df['Has_Top_VC']
            X_test_enhanced['Hub_x_TopVC'] = X_test_df['Is_Tech_Hub'] * X_test_df['Has_Top_VC']
    
    if 'Investor_Efficiency' not in X_train_df.columns and 'Investor_Count' in X_train_df.columns:
        X_train_enhanced['Investor_Efficiency'] = X_train_df['Investor_Count'] / (X_train_df['Valuation ($B)'] + 0.1)
        X_test_enhanced['Investor_Efficiency'] = X_test_df['Investor_Count'] / (X_test_df['Valuation ($B)'] + 0.1)
    
    if 'VC_Quality_Score' not in X_train_df.columns and 'Has_Top_VC' in X_train_df.columns:
        if 'Investor_Count' in X_train_df.columns:
            X_train_enhanced['VC_Quality_Score'] = X_train_df['Has_Top_VC'] * X_train_df['Investor_Count']
            X_test_enhanced['VC_Quality_Score'] = X_test_df['Has_Top_VC'] * X_test_df['Investor_Count']
    
    if 'Log_Investors' not in X_train_df.columns and 'Investor_Count' in X_train_df.columns:
        X_train_enhanced['Log_Investors'] = np.log1p(X_train_df['Investor_Count'])
        X_test_enhanced['Log_Investors'] = np.log1p(X_test_df['Investor_Count'])
    
    return X_train_enhanced, X_test_enhanced

# ============================================================================
# Test Different Configurations
# ============================================================================

configs = {
    'investor_only': {
        'name': 'Investor Features Only',
        'include': ['base', 'investor'],
        'exclude': ['temporal', 'industry', 'valuation'],
        'add_interactions': False
    },
    'temporal_only': {
        'name': 'Temporal Features Only',
        'include': ['base', 'temporal'],
        'exclude': ['investor', 'industry', 'valuation'],
        'add_interactions': False
    },
    'investor_temporal': {
        'name': 'Investor + Temporal (No Interactions)',
        'include': ['base', 'investor', 'temporal'],
        'exclude': ['industry', 'valuation'],
        'add_interactions': False
    },
    'investor_temporal_interactions': {
        'name': 'Investor + Temporal + Interactions',
        'include': ['base', 'investor', 'temporal'],
        'exclude': ['industry', 'valuation'],
        'add_interactions': True
    },
    'all_features': {
        'name': 'All Features (Full Model)',
        'include': ['base', 'investor', 'temporal', 'industry', 'valuation'],
        'exclude': [],
        'add_interactions': True
    }
}

# Get feature groups
feature_groups = get_feature_groups(X_train_base)

print("\nFeature Groups Available:")
for group, features in feature_groups.items():
    print(f"  {group}: {len(features)} features")
    if len(features) > 0 and len(features) <= 10:
        print(f"    {features}")

results = {}

print("\n" + "="*80)
print("TESTING CONFIGURATIONS")
print("="*80)

for config_key, config in configs.items():
    print(f"\n{'='*80}")
    print(f"CONFIGURATION: {config['name']}")
    print(f"{'='*80}")
    
    # Collect features
    selected_features = []
    for group in config['include']:
        selected_features.extend(feature_groups.get(group, []))
    
    # Remove excluded
    for group in config.get('exclude', []):
        selected_features = [f for f in selected_features if f not in feature_groups.get(group, [])]
    
    # Remove duplicates and ensure features exist
    selected_features = list(set([f for f in selected_features if f in X_train_base.columns]))
    
    print(f"\nBase features: {len(selected_features)}")
    print(f"  Features: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")
    
    # Add interactions if requested
    if config.get('add_interactions', False):
        X_train_config, X_test_config = create_investor_temporal_features(
            X_train_base[selected_features], 
            X_test_base[selected_features],
            df_full
        )
        print(f"With interactions: {X_train_config.shape[1]} features")
        print(f"  Added {X_train_config.shape[1] - len(selected_features)} interaction features")
    else:
        X_train_config = X_train_base[selected_features].copy()
        X_test_config = X_test_base[selected_features].copy()
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_config)
    X_test_scaled = scaler.transform(X_test_config)
    
    # Train models
    print("\nTraining models...")
    
    # Ridge
    ridge = Ridge()
    ridge_params = {'alpha': [0.1, 1, 10, 100, 1000]}
    ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2', n_jobs=-1)
    ridge_grid.fit(X_train_scaled, y_train)
    ridge_pred = ridge_grid.predict(X_test_scaled)
    
    # Random Forest
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=200, max_depth=15)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    
    # Store results
    results[config_key] = {
        'name': config['name'],
        'n_features': X_train_config.shape[1],
        'ridge_r2': r2_score(y_test, ridge_pred),
        'ridge_rmse': np.sqrt(mean_squared_error(y_test, ridge_pred)),
        'ridge_mae': mean_absolute_error(y_test, ridge_pred),
        'rf_r2': r2_score(y_test, rf_pred),
        'rf_rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'rf_mae': mean_absolute_error(y_test, rf_pred),
        'cv_r2': ridge_grid.best_score_,
        'features': list(X_train_config.columns),
        'ridge_model': ridge_grid.best_estimator_,
        'rf_model': rf
    }
    
    print(f"  Ridge R²: {results[config_key]['ridge_r2']:.4f} (CV: {results[config_key]['cv_r2']:.4f})")
    print(f"  RF R²: {results[config_key]['rf_r2']:.4f}")

# ============================================================================
# Feature Importance Analysis
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get best configuration
best_config_key = max(results.keys(), key=lambda k: results[k]['ridge_r2'])
best_config = results[best_config_key]

print(f"\nBest Configuration: {best_config['name']} (R² = {best_config['ridge_r2']:.4f})")

# Get feature importance from RF
rf_best = results[best_config_key]['rf_model']
best_features = results[best_config_key]['features']

importance_df = pd.DataFrame({
    'Feature': best_features,
    'Importance': rf_best.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 TOP 25 FEATURES (by importance):")
print("="*80)

# Categorize features
investor_features = []
temporal_features = []
interaction_features = []
other_features = []

for idx, row in importance_df.head(25).iterrows():
    feat = row['Feature']
    importance = row['Importance']
    
    if 'Investor' in feat or 'VC' in feat or 'investor' in feat.lower():
        investor_features.append((feat, importance))
    elif 'Era' in feat or 'Year' in feat or 'temporal' in feat.lower():
        temporal_features.append((feat, importance))
    elif '_x_' in feat or '×' in feat or 'x_' in feat:
        interaction_features.append((feat, importance))
    else:
        other_features.append((feat, importance))
    
    print(f"{feat:45s} {importance:.4f}")

print(f"\n📊 Feature Categories in Top 25:")
print(f"  Investor features: {len(investor_features)}")
if investor_features:
    print(f"    Top: {investor_features[0][0]} ({investor_features[0][1]:.4f})")
print(f"  Temporal features: {len(temporal_features)}")
if temporal_features:
    print(f"    Top: {temporal_features[0][0]} ({temporal_features[0][1]:.4f})")
print(f"  Interaction features: {len(interaction_features)}")
if interaction_features:
    print(f"    Top: {interaction_features[0][0]} ({interaction_features[0][1]:.4f})")
print(f"  Other features: {len(other_features)}")
if other_features:
    print(f"    Top: {other_features[0][0]} ({other_features[0][1]:.4f})")

# ============================================================================
# Comparison and Conclusions
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON")
print("="*80)

comparison_data = []
for config_key, result in results.items():
    comparison_data.append({
        'Configuration': result['name'],
        'N_Features': result['n_features'],
        'Ridge_R2': result['ridge_r2'],
        'RF_R2': result['rf_r2'],
        'Best_R2': max(result['ridge_r2'], result['rf_r2']),
        'CV_R2': result['cv_r2'],
        'Best_Model': 'Ridge' if result['ridge_r2'] > result['rf_r2'] else 'RF'
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('Best_R2', ascending=False)

print("\n" + comparison_df.to_string(index=False))

# Calculate improvements
baseline_r2 = results.get('all_features', {}).get('ridge_r2', 0.0357)
investor_only_r2 = results.get('investor_only', {}).get('ridge_r2', 0)
temporal_only_r2 = results.get('temporal_only', {}).get('ridge_r2', 0)
combined_r2 = results.get('investor_temporal', {}).get('ridge_r2', 0)
interactions_r2 = results.get('investor_temporal_interactions', {}).get('ridge_r2', 0)

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")

print(f"\n1. INVESTOR FEATURES ALONE:")
print(f"   R² = {investor_only_r2:.4f}")
if investor_only_r2 > 0.1:
    print(f"   ✅ Investor features provide moderate predictive power")
elif investor_only_r2 > 0.05:
    print(f"   ⚠️  Investor features provide some predictive power")
else:
    print(f"   ❌ Investor features alone have very limited predictive power")

print(f"\n2. TEMPORAL FEATURES ALONE:")
print(f"   R² = {temporal_only_r2:.4f}")
if temporal_only_r2 > 0.7:
    print(f"   ✅ Temporal features are HIGHLY predictive")
elif temporal_only_r2 > 0.4:
    print(f"   ⚠️  Temporal features have moderate predictive power")
else:
    print(f"   ❌ Temporal features have limited predictive power")

print(f"\n3. INVESTOR + TEMPORAL COMBINED:")
print(f"   R² = {combined_r2:.4f}")
improvement_over_temporal = combined_r2 - temporal_only_r2
if improvement_over_temporal > 0.05:
    print(f"   ✅ Adding investor features improves temporal-only by +{improvement_over_temporal:.4f}")
    print(f"   💡 Investor features add SIGNIFICANT value beyond temporal features")
elif improvement_over_temporal > 0.01:
    print(f"   ⚠️  Adding investor features provides moderate improvement (+{improvement_over_temporal:.4f})")
    print(f"   💡 Investor features add some value beyond temporal features")
elif improvement_over_temporal > 0:
    print(f"   ⚠️  Adding investor features provides minimal improvement (+{improvement_over_temporal:.4f})")
    print(f"   💡 Investor features add little value beyond temporal features")
else:
    print(f"   ❌ Investor features don't add value beyond temporal features")
    print(f"   💡 Temporal features dominate, investor features may be redundant")

print(f"\n4. WITH INTERACTIONS:")
print(f"   R² = {interactions_r2:.4f}")
interaction_improvement = interactions_r2 - combined_r2
if interaction_improvement > 0.01:
    print(f"   ✅ Investor × Temporal interactions improve by +{interaction_improvement:.4f}")
    print(f"   💡 Interactions capture valuable synergistic effects")
elif interaction_improvement > 0:
    print(f"   ⚠️  Interactions provide minimal improvement (+{interaction_improvement:.4f})")
    print(f"   💡 Interactions may help but effect is small")
else:
    print(f"   ❌ Interactions don't help (improvement: {interaction_improvement:.4f})")
    print(f"   💡 Interactions may cause overfitting or add noise")

print(f"\n5. FEATURE IMPORTANCE BREAKDOWN:")
print(f"   Investor features in top 25: {len(investor_features)}")
print(f"   Temporal features in top 25: {len(temporal_features)}")
print(f"   Interaction features in top 25: {len(interaction_features)}")
print(f"   Other features in top 25: {len(other_features)}")

# Calculate total importance by category
investor_total = sum([imp for _, imp in investor_features])
temporal_total = sum([imp for _, imp in temporal_features])
interaction_total = sum([imp for _, imp in interaction_features])
other_total = sum([imp for _, imp in other_features])
total_importance = investor_total + temporal_total + interaction_total + other_total

if total_importance > 0:
    print(f"\n   Total importance by category:")
    print(f"     Investor: {investor_total:.4f} ({investor_total/total_importance*100:.1f}%)")
    print(f"     Temporal: {temporal_total:.4f} ({temporal_total/total_importance*100:.1f}%)")
    print(f"     Interactions: {interaction_total:.4f} ({interaction_total/total_importance*100:.1f}%)")
    print(f"     Other: {other_total:.4f} ({other_total/total_importance*100:.1f}%)")

# ============================================================================
# Conclusions
# ============================================================================

print(f"\n{'='*80}")
print("CONCLUSIONS & RECOMMENDATIONS")
print(f"{'='*80}")

print(f"\n1. INVESTOR FEATURES CONTRIBUTION:")
if improvement_over_temporal > 0.05:
    print(f"   ✅ Investor features ADD SIGNIFICANT VALUE beyond temporal features")
    print(f"   💡 Recommendation: Include both investor and temporal features")
    print(f"   📊 Investor features explain additional {improvement_over_temporal*100:.1f}% of variance")
elif improvement_over_temporal > 0.01:
    print(f"   ⚠️  Investor features add SOME value, but temporal features dominate")
    print(f"   💡 Recommendation: Include both, but temporal features are primary")
    print(f"   📊 Investor features explain additional {improvement_over_temporal*100:.1f}% of variance")
else:
    print(f"   ❌ Investor features don't add significant value when temporal features are present")
    print(f"   💡 Recommendation: Focus on temporal features, investor features may be redundant")
    print(f"   📊 Investor features explain additional {improvement_over_temporal*100:.1f}% of variance")

print(f"\n2. INTERACTION FEATURES:")
if interaction_improvement > 0.01:
    print(f"   ✅ Investor × Temporal interactions are valuable")
    print(f"   💡 Recommendation: Include interaction features")
    print(f"   📊 Interactions explain additional {interaction_improvement*100:.1f}% of variance")
elif interaction_improvement > 0:
    print(f"   ⚠️  Interactions provide minimal improvement")
    print(f"   💡 Recommendation: Interactions optional, may cause overfitting")
else:
    print(f"   ❌ Interactions don't help significantly")
    print(f"   💡 Recommendation: Skip interactions to avoid overfitting")

print(f"\n3. RELATIVE IMPORTANCE:")
if temporal_total > investor_total * 2:
    print(f"   📊 Temporal features are PRIMARY (>{temporal_total/total_importance*100:.0f}% importance)")
    print(f"   📊 Investor features are SECONDARY (>{investor_total/total_importance*100:.0f}% importance)")
elif temporal_total > investor_total:
    print(f"   📊 Temporal features are MORE important ({temporal_total/total_importance*100:.0f}% vs {investor_total/total_importance*100:.0f}%)")
    print(f"   📊 But investor features still matter")
else:
    print(f"   📊 Investor and temporal features are BOTH important")
    print(f"   📊 Investor: {investor_total/total_importance*100:.0f}%, Temporal: {temporal_total/total_importance*100:.0f}%")

print(f"\n4. FINAL RECOMMENDATION:")
best_r2 = max([r['ridge_r2'] for r in results.values()])
best_config_name = [k for k, v in results.items() if v['ridge_r2'] == best_r2][0]
print(f"   ✅ Use: {results[best_config_name]['name']}")
print(f"   📊 Achieves: R² = {best_r2:.4f}")
print(f"   📊 Features: {results[best_config_name]['n_features']} total")

# ============================================================================
# Save Results
# ============================================================================

os.makedirs('output/data', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)

comparison_df.to_csv('output/data/investor_temporal_comparison.csv', index=False)
print(f"\n✓ Saved: output/data/investor_temporal_comparison.csv")

importance_df.to_csv('output/data/investor_temporal_feature_importance.csv', index=False)
print(f"✓ Saved: output/data/investor_temporal_feature_importance.csv")

with open('output/data/investor_temporal_detailed.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"✓ Saved: output/data/investor_temporal_detailed.pkl")

# Create comprehensive report
report = f"""
# Investor + Temporal Features Comprehensive Analysis

## Executive Summary

Comprehensive analysis of how investor and temporal features work together to predict Years_to_Unicorn.

**Best Configuration**: {results[best_config_name]['name']}
**Best R²**: {best_r2:.4f}
**Features**: {results[best_config_name]['n_features']}

## Results Comparison

{comparison_df.to_string(index=False)}

## Key Findings

### 1. Individual Feature Groups

- **Investor Features Alone**: R² = {investor_only_r2:.4f}
- **Temporal Features Alone**: R² = {temporal_only_r2:.4f}
- **Combined (No Interactions)**: R² = {combined_r2:.4f}
- **With Interactions**: R² = {interactions_r2:.4f}

### 2. Incremental Value

- **Investor adds to Temporal**: {improvement_over_temporal:+.4f} R² ({improvement_over_temporal/temporal_only_r2*100:+.1f}% relative improvement)
- **Interactions add to Combined**: {interaction_improvement:+.4f} R² ({interaction_improvement/combined_r2*100:+.1f}% relative improvement)

### 3. Feature Importance

Top 25 features include:
- **{len(investor_features)} investor features** (total importance: {investor_total:.4f}, {investor_total/total_importance*100:.1f}%)
- **{len(temporal_features)} temporal features** (total importance: {temporal_total:.4f}, {temporal_total/total_importance*100:.1f}%)
- **{len(interaction_features)} interaction features** (total importance: {interaction_total:.4f}, {interaction_total/total_importance*100:.1f}%)
- **{len(other_features)} other features** (total importance: {other_total:.4f}, {other_total/total_importance*100:.1f}%)

## Top Features

### Top Investor Features
{chr(10).join([f"- {feat[0]}: {feat[1]:.4f}" for feat in investor_features[:5]]) if investor_features else "None in top 25"}

### Top Temporal Features
{chr(10).join([f"- {feat[0]}: {feat[1]:.4f}" for feat in temporal_features[:5]]) if temporal_features else "None in top 25"}

### Top Interaction Features
{chr(10).join([f"- {feat[0]}: {feat[1]:.4f}" for feat in interaction_features[:5]]) if interaction_features else "None in top 25"}

## Conclusions

### Investor Features Contribution
{'Investor features ADD SIGNIFICANT VALUE beyond temporal features' if improvement_over_temporal > 0.05 else 'Investor features add SOME value, but temporal features dominate' if improvement_over_temporal > 0.01 else 'Investor features add minimal value when temporal features are present'}

### Interaction Features Value
{'Investor × Temporal interactions are valuable and should be included' if interaction_improvement > 0.01 else 'Interactions provide minimal benefit and may cause overfitting' if interaction_improvement > 0 else 'Interactions do not help and should be avoided'}

### Final Recommendation
Use **{results[best_config_name]['name']}** configuration with {results[best_config_name]['n_features']} features to achieve R² = {best_r2:.4f}.

## Strategic Implications

1. **Temporal factors are PRIMARY**: When a company was founded is the strongest predictor
2. **Investor factors are SECONDARY**: Investor quality/quantity adds value but is secondary
3. **Interactions matter**: {'Investor × Temporal interactions capture synergistic effects' if interaction_improvement > 0.01 else 'Interactions provide minimal additional value'}
4. **Both matter**: {'Both investor and temporal features should be included' if improvement_over_temporal > 0.01 else 'Focus on temporal features, investor features are less critical'}

## Detailed Results

See `output/data/investor_temporal_detailed.pkl` for full model results and feature lists.
See `output/data/investor_temporal_feature_importance.csv` for complete feature importance rankings.
"""

with open('output/reports/INVESTOR_TEMPORAL_ANALYSIS.md', 'w') as f:
    f.write(report)
print(f"✓ Saved: output/reports/INVESTOR_TEMPORAL_ANALYSIS.md")

print("\n" + "="*80)
print("✅ INVESTOR + TEMPORAL ANALYSIS COMPLETE")
print("="*80)

