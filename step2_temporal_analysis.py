"""
Comprehensive Temporal Features Analysis
========================================

This script tests different temporal feature configurations to determine
which approach works best for predicting Years_to_Unicorn.

Tests:
1. Baseline: No temporal features (current approach)
2. Era features only (categorical eras)
3. Year_Founded only (without Date_Joined_Year)
4. Market condition features (derived from Year_Founded)
5. Combined approaches

Compares all and provides conclusions.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("TEMPORAL FEATURES COMPREHENSIVE ANALYSIS")
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
# Define Temporal Feature Configurations
# ============================================================================

def get_temporal_config(config_name, X_train_df, X_test_df, df_full):
    """Get features for a specific temporal configuration"""
    
    # Base leakage features (always remove these)
    base_leakage = ['Date_Joined_Year']  # Always remove - direct calculation
    
    if config_name == "baseline":
        # No temporal features at all
        leakage = base_leakage + ['Year_Founded', 'Company_Age_2025',
                                  'Era_Pre-2000', 'Era_2000-2009', 'Era_2010-2014', 
                                  'Era_2015-2019', 'Era_2020+']
        description = "No temporal features (current baseline)"
        
    elif config_name == "era_only":
        # Only era categorical features
        leakage = base_leakage + ['Year_Founded', 'Company_Age_2025']
        # Keep Era_* features
        description = "Era categorical features only (Pre-2000, 2000-2009, etc.)"
        
    elif config_name == "year_founded_only":
        # Year_Founded but not Date_Joined_Year
        leakage = base_leakage + ['Company_Age_2025',
                                  'Era_Pre-2000', 'Era_2000-2009', 'Era_2010-2014', 
                                  'Era_2015-2019', 'Era_2020+']
        # Keep Year_Founded
        description = "Year_Founded only (without Date_Joined_Year)"
        
    elif config_name == "market_conditions":
        # Create market condition features from Year_Founded
        leakage = base_leakage + ['Year_Founded', 'Company_Age_2025',
                                  'Era_Pre-2000', 'Era_2000-2009', 'Era_2010-2014', 
                                  'Era_2015-2019', 'Era_2020+']
        description = "Market condition features (derived from Year_Founded)"
        
    elif config_name == "era_and_year":
        # Both Era and Year_Founded
        leakage = base_leakage + ['Company_Age_2025']
        # Keep Year_Founded and Era_* features
        description = "Both Era features and Year_Founded"
        
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    # Get available features
    available_features = [f for f in X_train_df.columns if f not in leakage]
    
    # For market_conditions, add derived features
    if config_name == "market_conditions":
        # Create market condition features
        if 'Year_Founded' in df_full.columns:
            train_years = df_full.loc[X_train_df.index, 'Year_Founded'] if hasattr(X_train_df, 'index') else df_full['Year_Founded'].iloc[:len(X_train_df)]
            test_years = df_full.loc[X_test_df.index, 'Year_Founded'] if hasattr(X_test_df, 'index') else df_full['Year_Founded'].iloc[len(X_train_df):len(X_train_df)+len(X_test_df)]
            
            # Create market condition features
            market_features_train = pd.DataFrame({
                'Is_Pre_DotCom': (train_years < 2000).astype(int).values,
                'Is_DotCom_Era': ((train_years >= 2000) & (train_years < 2010)).astype(int).values,
                'Is_Tech_Boom': ((train_years >= 2010) & (train_years < 2015)).astype(int).values,
                'Is_Peak_Unicorn': ((train_years >= 2015) & (train_years < 2020)).astype(int).values,
                'Is_Post_COVID': (train_years >= 2020).astype(int).values,
            }, index=X_train_df.index if hasattr(X_train_df, 'index') else range(len(X_train_df)))
            
            market_features_test = pd.DataFrame({
                'Is_Pre_DotCom': (test_years < 2000).astype(int).values,
                'Is_DotCom_Era': ((test_years >= 2000) & (test_years < 2010)).astype(int).values,
                'Is_Tech_Boom': ((test_years >= 2010) & (test_years < 2015)).astype(int).values,
                'Is_Peak_Unicorn': ((test_years >= 2015) & (test_years < 2020)).astype(int).values,
                'Is_Post_COVID': (test_years >= 2020).astype(int).values,
            }, index=X_test_df.index if hasattr(X_test_df, 'index') else range(len(X_test_df)))
            
            # Combine with available features
            X_train_config = pd.concat([X_train_df[available_features], market_features_train], axis=1)
            X_test_config = pd.concat([X_test_df[available_features], market_features_test], axis=1)
        else:
            X_train_config = X_train_df[available_features]
            X_test_config = X_test_df[available_features]
    else:
        X_train_config = X_train_df[available_features]
        X_test_config = X_test_df[available_features]
    
    return X_train_config, X_test_config, description, available_features

# ============================================================================
# Test Each Configuration
# ============================================================================

configs = ["baseline", "era_only", "year_founded_only", "market_conditions", "era_and_year"]
results = {}

print("\n" + "="*80)
print("TESTING TEMPORAL FEATURE CONFIGURATIONS")
print("="*80)

for config_name in configs:
    print(f"\n{'='*80}")
    print(f"CONFIGURATION: {config_name.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    
    # Get features for this configuration
    X_train_config, X_test_config, description, available_features = get_temporal_config(
        config_name, X_train, X_test, df_full
    )
    
    print(f"\n{description}")
    print(f"Features: {X_train_config.shape[1]} total")
    
    # Feature selection (same as improved script)
    print("\nApplying feature selection...")
    
    # Correlation filtering
    feature_corr = X_train_config.corrwith(y_train).abs()
    high_corr_features = feature_corr[feature_corr > 0.05].index.tolist()
    
    # Mutual Information
    mi_selector = SelectKBest(score_func=mutual_info_regression, k='all')
    mi_selector.fit(X_train_config, y_train)
    mi_scores = pd.Series(mi_selector.scores_, index=X_train_config.columns).sort_values(ascending=False)
    mi_threshold = np.percentile(mi_scores, 25)
    high_mi_features = mi_scores[mi_scores > mi_threshold].index.tolist()
    
    # F-statistic
    f_selector = SelectKBest(score_func=f_regression, k='all')
    f_selector.fit(X_train_config, y_train)
    f_scores = pd.Series(f_selector.scores_, index=X_train_config.columns).sort_values(ascending=False)
    f_threshold = np.percentile(f_scores, 25)
    high_f_features = f_scores[f_scores > f_threshold].index.tolist()
    
    # Combine methods
    set_corr = set(high_corr_features)
    set_mi = set(high_mi_features)
    set_f = set(high_f_features)
    
    selected_features_set = (set_corr & set_mi) | (set_corr & set_f) | (set_mi & set_f)
    
    if len(selected_features_set) < 5:
        selected_features_set = set_corr | set_mi | set_f
    
    # Ensure key features
    key_features = ['Valuation ($B)', 'Investor_Count', 'Has_Top_VC', 'Is_Tech_Hub', 
                    'Country_Tier', 'Is_Tech_Intensive']
    for feat in key_features:
        if feat in X_train_config.columns:
            selected_features_set.add(feat)
    
    selected_features = list(selected_features_set)
    print(f"Selected features: {len(selected_features)}/{X_train_config.shape[1]}")
    
    # Filter and scale
    X_train_selected = X_train_config[selected_features].copy()
    X_test_selected = X_test_config[selected_features].copy()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
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
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=5)
    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)
    
    # Store results
    results[config_name] = {
        'description': description,
        'n_features': len(selected_features),
        'ridge': {
            'test_r2': r2_score(y_test, ridge_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, ridge_pred)),
            'test_mae': mean_absolute_error(y_test, ridge_pred),
            'cv_r2': ridge_grid.best_score_,
            'alpha': ridge_grid.best_params_['alpha']
        },
        'rf': {
            'test_r2': r2_score(y_test, rf_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'test_mae': mean_absolute_error(y_test, rf_pred)
        },
        'gb': {
            'test_r2': r2_score(y_test, gb_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
            'test_mae': mean_absolute_error(y_test, gb_pred)
        },
        'selected_features': selected_features
    }
    
    print(f"\nResults:")
    print(f"  Ridge R²: {results[config_name]['ridge']['test_r2']:.4f}")
    print(f"  RF R²: {results[config_name]['rf']['test_r2']:.4f}")
    print(f"  GB R²: {results[config_name]['gb']['test_r2']:.4f}")

# ============================================================================
# Comparison and Analysis
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON")
print("="*80)

# Create comparison DataFrame
comparison_data = []
for config_name, result in results.items():
    comparison_data.append({
        'Configuration': config_name.replace('_', ' ').title(),
        'Description': result['description'],
        'N_Features': result['n_features'],
        'Ridge_R2': result['ridge']['test_r2'],
        'RF_R2': result['rf']['test_r2'],
        'GB_R2': result['gb']['test_r2'],
        'Best_R2': max(result['ridge']['test_r2'], result['rf']['test_r2'], result['gb']['test_r2']),
        'Best_Model': ['Ridge', 'RF', 'GB'][np.argmax([
            result['ridge']['test_r2'], 
            result['rf']['test_r2'], 
            result['gb']['test_r2']
        ])]
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Best_R2', ascending=False)

print("\n" + comparison_df.to_string(index=False))

# Find best configuration
best_config = comparison_df.iloc[0]
baseline_r2 = comparison_df[comparison_df['Configuration'] == 'Baseline']['Best_R2'].values[0]

print(f"\n{'='*80}")
print("KEY FINDINGS")
print(f"{'='*80}")

print(f"\n🏆 BEST CONFIGURATION: {best_config['Configuration']}")
print(f"   Best R²: {best_config['Best_R2']:.4f} ({best_config['Best_Model']})")
print(f"   Improvement over baseline: {best_config['Best_R2'] - baseline_r2:+.4f}")

print(f"\n📊 Performance Summary:")
for idx, row in comparison_df.iterrows():
    improvement = row['Best_R2'] - baseline_r2
    pct_improvement = (improvement / baseline_r2 * 100) if baseline_r2 > 0 else 0
    status = "✅" if improvement > 0 else "❌" if improvement < 0 else "➡️"
    print(f"   {status} {row['Configuration']:20s}: R² = {row['Best_R2']:.4f} ({improvement:+.4f}, {pct_improvement:+.1f}%)")

# ============================================================================
# Detailed Analysis
# ============================================================================

print(f"\n{'='*80}")
print("DETAILED ANALYSIS")
print(f"{'='*80}")

# Analyze which temporal features matter
print("\n1. ERA FEATURES ANALYSIS:")
era_config = results.get('era_only', {})
baseline_config = results.get('baseline', {})
if era_config and baseline_config:
    era_improvement = era_config['ridge']['test_r2'] - baseline_config['ridge']['test_r2']
    print(f"   Era features improve R² by: {era_improvement:+.4f}")
    if era_improvement > 0:
        print(f"   ✅ Era features help capture temporal patterns")
    else:
        print(f"   ❌ Era features don't improve performance")

print("\n2. YEAR_FOUNDED ANALYSIS:")
year_config = results.get('year_founded_only', {})
if year_config and baseline_config:
    year_improvement = year_config['ridge']['test_r2'] - baseline_config['ridge']['test_r2']
    print(f"   Year_Founded improves R² by: {year_improvement:+.4f}")
    if year_improvement > 0.01:
        print(f"   ✅ Year_Founded provides useful signal (but may have leakage risk)")
    elif year_improvement > 0:
        print(f"   ⚠️  Year_Founded provides minimal improvement")
    else:
        print(f"   ❌ Year_Founded doesn't help")

print("\n3. MARKET CONDITIONS ANALYSIS:")
market_config = results.get('market_conditions', {})
if market_config and baseline_config:
    market_improvement = market_config['ridge']['test_r2'] - baseline_config['ridge']['test_r2']
    print(f"   Market condition features improve R² by: {market_improvement:+.4f}")
    if market_improvement > 0:
        print(f"   ✅ Market condition features capture era effects effectively")
    else:
        print(f"   ❌ Market condition features don't improve performance")

print("\n4. COMBINED APPROACH:")
combined_config = results.get('era_and_year', {})
if combined_config and baseline_config:
    combined_improvement = combined_config['ridge']['test_r2'] - baseline_config['ridge']['test_r2']
    print(f"   Combined (Era + Year) improves R² by: {combined_improvement:+.4f}")
    if combined_improvement > max(era_config.get('ridge', {}).get('test_r2', 0) - baseline_config['ridge']['test_r2'],
                                   year_config.get('ridge', {}).get('test_r2', 0) - baseline_config['ridge']['test_r2']):
        print(f"   ✅ Combining features provides synergy")
    else:
        print(f"   ⚠️  Combining features doesn't provide additional benefit")

# ============================================================================
# Conclusions
# ============================================================================

print(f"\n{'='*80}")
print("CONCLUSIONS & RECOMMENDATIONS")
print(f"{'='*80}")

print("\n1. TEMPORAL FEATURES IMPACT:")
best_improvement = best_config['Best_R2'] - baseline_r2
if best_improvement > 0.01:
    print(f"   ✅ Temporal features provide meaningful improvement (+{best_improvement:.4f} R²)")
    print(f"   💡 Recommendation: Use {best_config['Configuration']} configuration")
elif best_improvement > 0:
    print(f"   ⚠️  Temporal features provide minimal improvement (+{best_improvement:.4f} R²)")
    print(f"   💡 Recommendation: Consider using temporal features, but don't expect major gains")
else:
    print(f"   ❌ Temporal features don't improve performance")
    print(f"   💡 Recommendation: Stick with baseline (no temporal features)")

print("\n2. LEAKAGE RISK ASSESSMENT:")
if 'year' in best_config['Configuration'].lower():
    print(f"   ⚠️  Best configuration uses Year_Founded - monitor for leakage")
    print(f"   💡 Ensure Date_Joined_Year is NOT in features")
else:
    print(f"   ✅ Best configuration avoids direct temporal leakage")

print("\n3. INTERPRETABILITY:")
if 'era' in best_config['Configuration'].lower() or 'market' in best_config['Configuration'].lower():
    print(f"   ✅ Temporal features provide interpretable era effects")
    print(f"   💡 Can analyze: 'Companies founded in X era have different growth patterns'")
else:
    print(f"   ⚠️  No temporal interpretability with current best configuration")

print("\n4. FINAL RECOMMENDATION:")
if best_improvement > 0.005:
    rec_text = f"Use {best_config['Configuration']} configuration"
else:
    rec_text = "Stick with baseline - temporal features don't help significantly"
print(f"   {rec_text}")

# ============================================================================
# Save Results
# ============================================================================

os.makedirs('output/data', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)

# Save comparison
comparison_df.to_csv('output/data/temporal_features_comparison.csv', index=False)
print(f"\n✓ Saved: output/data/temporal_features_comparison.csv")

# Save detailed results
with open('output/data/temporal_features_detailed.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"✓ Saved: output/data/temporal_features_detailed.pkl")

# Create summary report
impact_text = 'Significant' if best_improvement > 0.01 else 'Minimal' if best_improvement > 0 else 'None'
leakage_text = 'Low' if 'year' not in best_config['Configuration'].lower() else 'Medium - monitor Year_Founded usage'
if best_improvement > 0.01:
    rec_text = 'Use temporal features - they provide meaningful improvement'
elif best_improvement > 0:
    rec_text = 'Temporal features provide minimal benefit - consider other approaches'
else:
    rec_text = "Temporal features don't help - focus on other feature engineering"

report = f"""
# Temporal Features Analysis Report

## Executive Summary

Comprehensive analysis of temporal feature configurations for predicting Years_to_Unicorn.

**Best Configuration**: {best_config['Configuration']}
**Best R²**: {best_config['Best_R2']:.4f} ({best_config['Best_Model']})
**Improvement over Baseline**: {best_improvement:+.4f} ({best_improvement/baseline_r2*100:+.1f}%)

## Results Comparison

{comparison_df.to_string(index=False)}

## Key Findings

1. **Temporal Features Impact**: {impact_text}
2. **Best Approach**: {best_config['Configuration']}
3. **Leakage Risk**: {leakage_text}

## Recommendations

{rec_text}

## Detailed Results

See `output/data/temporal_features_detailed.pkl` for full model results and feature lists.
"""

with open('output/reports/TEMPORAL_FEATURES_ANALYSIS.md', 'w') as f:
    f.write(report)
print(f"✓ Saved: output/reports/TEMPORAL_FEATURES_ANALYSIS.md")

print("\n" + "="*80)
print("✅ TEMPORAL FEATURES ANALYSIS COMPLETE")
print("="*80)

