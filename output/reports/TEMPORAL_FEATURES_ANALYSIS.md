
# Temporal Features Analysis Report

## Executive Summary

Comprehensive analysis of temporal feature configurations for predicting Years_to_Unicorn.

**Best Configuration**: Era And Year
**Best R²**: 0.8475 (Ridge)
**Improvement over Baseline**: +0.8118 (+2275.3%)

## Results Comparison

    Configuration                                               Description  N_Features  Ridge_R2     RF_R2     GB_R2  Best_R2 Best_Model
     Era And Year                        Both Era features and Year_Founded          15  0.847458  0.816824  0.825881 0.847458      Ridge
Year Founded Only              Year_Founded only (without Date_Joined_Year)          10  0.841114  0.817276  0.822398 0.841114      Ridge
         Era Only Era categorical features only (Pre-2000, 2000-2009, etc.)          14  0.736713  0.721755  0.687999 0.736713      Ridge
Market Conditions     Market condition features (derived from Year_Founded)          15  0.735328  0.721892  0.680174 0.735328      Ridge
         Baseline                   No temporal features (current baseline)          10  0.035677 -0.092352 -0.131456 0.035677      Ridge

## Key Findings

1. **Temporal Features Impact**: Significant
2. **Best Approach**: Era And Year
3. **Leakage Risk**: Medium - monitor Year_Founded usage

## Recommendations

Use temporal features - they provide meaningful improvement

## Detailed Results

See `output/data/temporal_features_detailed.pkl` for full model results and feature lists.
