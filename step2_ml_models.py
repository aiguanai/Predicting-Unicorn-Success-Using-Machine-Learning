"""
Step 2: ML Models for Growth Speed Prediction & Feature Importance
OBJECTIVES 2 & 3
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print("OBJECTIVES 2 & 3: GROWTH SPEED PREDICTION + FEATURE IMPORTANCE")
print("="*80)

# ============================================================================
# LOAD PREPROCESSED DATA
# ============================================================================

print("\nLoading preprocessed data...")
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
feature_names = data['feature_names']

print(f"✓ Data loaded")
print(f"  Training: {len(X_train)} samples")
print(f"  Test: {len(X_test)} samples")
print(f"  Features: {len(feature_names)}")

# ============================================================================
# OBJECTIVE 2: GROWTH SPEED PREDICTION (REGRESSION)
# ============================================================================

print("\n" + "="*80)
print("OBJECTIVE 2: GROWTH SPEED PREDICTION")
print("="*80)

# ============================================================================
# MODEL 1: LINEAR REGRESSION (Baseline)
# ============================================================================

print("\n" + "-"*80)
print("Model 1: Linear Regression (Baseline)")
print("-"*80)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr_train = lr.predict(X_train)
y_pred_lr_test = lr.predict(X_test)

r2_lr_train = r2_score(y_train, y_pred_lr_train)
r2_lr_test = r2_score(y_test, y_pred_lr_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr_test))
mae_lr = mean_absolute_error(y_test, y_pred_lr_test)

print(f"\nPerformance:")
print(f"  Train R²: {r2_lr_train:.4f}")
print(f"  Test R²: {r2_lr_test:.4f}")
print(f"  Test RMSE: {rmse_lr:.2f} years")
print(f"  Test MAE: {mae_lr:.2f} years")

cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
print(f"  5-Fold CV R²: {cv_scores_lr.mean():.4f} (±{cv_scores_lr.std():.4f})")

# ============================================================================
# MODEL 2: RANDOM FOREST REGRESSOR (Primary Model)
# ============================================================================

print("\n" + "-"*80)
print("Model 2: Random Forest Regressor (Primary Model)")
print("-"*80)

print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf_train = rf.predict(X_train)
y_pred_rf_test = rf.predict(X_test)

r2_rf_train = r2_score(y_train, y_pred_rf_train)
r2_rf_test = r2_score(y_test, y_pred_rf_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
mae_rf = mean_absolute_error(y_test, y_pred_rf_test)

print(f"\nPerformance:")
print(f"  Train R²: {r2_rf_train:.4f}")
print(f"  Test R²: {r2_rf_test:.4f}")
print(f"  Test RMSE: {rmse_rf:.2f} years")
print(f"  Test MAE: {mae_rf:.2f} years")

cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"  5-Fold CV R²: {cv_scores_rf.mean():.4f} (±{cv_scores_rf.std():.4f})")

# Model comparison
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'Test R²': [r2_lr_test, r2_rf_test],
    'Test RMSE': [rmse_lr, rmse_rf],
    'Test MAE': [mae_lr, mae_rf],
    'CV R² Mean': [cv_scores_lr.mean(), cv_scores_rf.mean()]
})

print("\n", comparison.to_string(index=False))

best_model = 'Random Forest' if r2_rf_test > r2_lr_test else 'Linear Regression'
best_r2 = max(r2_rf_test, r2_lr_test)

print(f"\n🏆 Best Model: {best_model}")
print(f"   Test R²: {best_r2:.4f}")

if best_r2 >= 0.60:
    print(f"   ✅ OBJECTIVE 2 ACHIEVED: R² exceeds 0.60 target!")
else:
    print(f"   ⚠️  R² below 0.60 target, but model still provides insights")

# ============================================================================
# OBJECTIVE 3: FACTOR IMPORTANCE RANKING
# ============================================================================

print("\n" + "="*80)
print("OBJECTIVE 3: FACTOR IMPORTANCE RANKING")
print("="*80)

print("\nExtracting feature importance from Random Forest...")

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 TOP 20 MOST IMPORTANT FEATURES:")
print("="*80)
for idx, row in feature_importance.head(20).iterrows():
    print(f"{idx+1:2d}. {row['Feature']:40s} {row['Importance']:.4f}")

# Group importance by category
print("\n📊 IMPORTANCE BY CATEGORY:")
print("="*80)

# Calculate category totals
category_importance = {}
for idx, row in feature_importance.iterrows():
    feat = row['Feature']
    imp = row['Importance']
    
    if feat.startswith('Ind_'):
        category = 'Industry'
    elif feat.startswith('Era_'):
        category = 'Founding Era'
    elif feat.startswith('ValCat_'):
        category = 'Valuation Category'
    elif feat in ['Valuation ($B)', 'Log_Valuation']:
        category = 'Valuation'
    elif feat in ['Is_Tech_Hub', 'Is_Silicon_Valley', 'Country_Tier']:
        category = 'Geography'
    elif feat in ['Investor_Count', 'Has_Top_VC']:
        category = 'Investors'
    elif feat in ['Year_Founded', 'Date_Joined_Year', 'Company_Age_2025']:
        category = 'Temporal'
    else:
        category = 'Other'
    
    category_importance[category] = category_importance.get(category, 0) + imp

category_df = pd.DataFrame({
    'Category': list(category_importance.keys()),
    'Total_Importance': list(category_importance.values())
}).sort_values('Total_Importance', ascending=False)

for idx, row in category_df.iterrows():
    pct = row['Total_Importance'] * 100
    print(f"{row['Category']:25s} {row['Total_Importance']:.4f} ({pct:.1f}%)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: Top 15 Features
plt.figure(figsize=(12, 8))
top15 = feature_importance.head(15)
plt.barh(range(len(top15)), top15['Importance'], color='steelblue')
plt.yticks(range(len(top15)), top15['Feature'])
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 15 Most Important Features for Predicting Growth Speed\n(Years to Unicorn)', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('top15_features.png', dpi=300, bbox_inches='tight')
print("✓ Saved: top15_features.png")
plt.close()

# Plot 2: Category Importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(category_df)), category_df['Total_Importance'], color='coral')
plt.xticks(range(len(category_df)), category_df['Category'], rotation=45, ha='right')
plt.ylabel('Total Importance', fontsize=12, fontweight='bold')
plt.title('Feature Importance by Category', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('category_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: category_importance.png")
plt.close()

# Plot 3: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf_test, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Years to Unicorn', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Years to Unicorn', fontsize=12, fontweight='bold')
plt.title(f'Actual vs Predicted: Random Forest\n(R² = {r2_rf_test:.3f}, RMSE = {rmse_rf:.2f} years)', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: actual_vs_predicted.png")
plt.close()

# Plot 4: Residuals
residuals = y_test - y_pred_rf_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf_test, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Years to Unicorn', fontsize=12, fontweight='bold')
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
plt.title('Residual Plot: Checking Model Fit', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: residual_plot.png")
plt.close()

# Plot 5: Model Comparison
plt.figure(figsize=(8, 6))
models = comparison['Model']
r2_scores = comparison['Test R²']
colors = ['lightcoral' if r2 < 0.6 else 'lightgreen' for r2 in r2_scores]
bars = plt.bar(range(len(models)), r2_scores, color=colors, edgecolor='black', linewidth=1.5)
plt.xticks(range(len(models)), models)
plt.ylabel('R² Score', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.axhline(y=0.6, color='blue', linestyle='--', alpha=0.7, label='Target (R²=0.60)', linewidth=2)
plt.legend(fontsize=10)
plt.ylim(0, max(r2_scores) * 1.2)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, r2) in enumerate(zip(bars, r2_scores)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save models
models_dict = {
    'linear_regression': lr,
    'random_forest': rf,
    'best_model': rf if r2_rf_test > r2_lr_test else lr,
    'best_model_name': best_model
}

with open('trained_models.pkl', 'wb') as f:
    pickle.dump(models_dict, f)
print("✓ Saved: trained_models.pkl")

# Save feature importance
feature_importance.to_csv('feature_importance_rankings.csv', index=False)
print("✓ Saved: feature_importance_rankings.csv")

category_df.to_csv('category_importance.csv', index=False)
print("✓ Saved: category_importance.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted_LR': y_pred_lr_test,
    'Predicted_RF': y_pred_rf_test,
    'Residual_RF': residuals.values,
    'Abs_Error_RF': np.abs(residuals.values)
})
predictions_df.to_csv('predictions.csv', index=False)
print("✓ Saved: predictions.csv")

# Save model performance summary
comparison.to_csv('model_performance.csv', index=False)
print("✓ Saved: model_performance.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ OBJECTIVES 2 & 3 COMPLETED")
print("="*80)

print(f"\n🎯 OBJECTIVE 2: Growth Speed Prediction")
print(f"   Best Model: {best_model}")
print(f"   Test R²: {best_r2:.4f} {'✅ (Target: >0.60)' if best_r2 >= 0.60 else '⚠️  (Below 0.60 target)'}")
print(f"   Test RMSE: {rmse_rf if best_model == 'Random Forest' else rmse_lr:.2f} years")
print(f"   Test MAE: {mae_rf if best_model == 'Random Forest' else mae_lr:.2f} years")
print(f"\n   Interpretation:")
print(f"   - Model explains {best_r2*100:.1f}% of variance in growth speed")
print(f"   - Average prediction error: ±{(mae_rf if best_model == 'Random Forest' else mae_lr):.1f} years")

print(f"\n🎯 OBJECTIVE 3: Factor Importance Ranking")
print(f"   Top 5 Predictive Factors:")
for i, row in feature_importance.head(5).iterrows():
    print(f"   {i+1}. {row['Feature']} (Importance: {row['Importance']:.4f})")

print(f"\n   Key Insights:")
print(f"   - {category_df.iloc[0]['Category']} features are most important ({category_df.iloc[0]['Total_Importance']*100:.1f}%)")
print(f"   - {category_df.iloc[1]['Category']} features rank second ({category_df.iloc[1]['Total_Importance']*100:.1f}%)")

print(f"\n📊 Generated Outputs:")
print(f"   ✓ top15_features.png (visualization)")
print(f"   ✓ category_importance.png (visualization)")
print(f"   ✓ actual_vs_predicted.png (visualization)")
print(f"   ✓ residual_plot.png (visualization)")
print(f"   ✓ model_comparison.png (visualization)")
print(f"   ✓ feature_importance_rankings.csv (data)")
print(f"   ✓ category_importance.csv (data)")
print(f"   ✓ predictions.csv (data)")
print(f"   ✓ model_performance.csv (data)")
print(f"   ✓ trained_models.pkl (saved models)")

print(f"\n🎯 Next Step:")
print(f"   Run 'step3_porters_analysis.py' for Objective 4 (Theoretical Validation)")

print("\n" + "="*80)