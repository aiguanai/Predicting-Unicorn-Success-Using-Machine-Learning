
# Investor + Temporal Features Comprehensive Analysis

## Executive Summary

Comprehensive analysis of how investor and temporal features work together to predict Years_to_Unicorn.

**Best Configuration**: All Features (Full Model)
**Best R²**: 0.8549
**Features**: 32

## Results Comparison

                        Configuration  N_Features  Ridge_R2     RF_R2  Best_R2    CV_R2 Best_Model
            All Features (Full Model)          32  0.854948  0.820785 0.854948 0.819561      Ridge
   Investor + Temporal + Interactions          27  0.841032  0.820301 0.841032 0.808034      Ridge
Investor + Temporal (No Interactions)          12  0.840547  0.818879 0.840547 0.806479      Ridge
               Temporal Features Only          10  0.836832  0.824088 0.836832 0.802884      Ridge
               Investor Features Only           7  0.036162 -0.102731 0.036162 0.033183      Ridge

## Key Findings

### 1. Individual Feature Groups

- **Investor Features Alone**: R² = 0.0362
- **Temporal Features Alone**: R² = 0.8368
- **Combined (No Interactions)**: R² = 0.8405
- **With Interactions**: R² = 0.8410

### 2. Incremental Value

- **Investor adds to Temporal**: +0.0037 R² (+0.4% relative improvement)
- **Interactions add to Combined**: +0.0005 R² (+0.1% relative improvement)

### 3. Feature Importance

Top 25 features include:
- **13 investor features** (total importance: 0.1131, 11.3%)
- **2 temporal features** (total importance: 0.7997, 80.2%)
- **0 interaction features** (total importance: 0.0000, 0.0%)
- **10 other features** (total importance: 0.0840, 8.4%)

## Top Features

### Top Investor Features
- Investors_x_Year: 0.0310
- Investor_Efficiency: 0.0236
- Val_per_Investor: 0.0205
- Investors_x_Era_2010-2014: 0.0104
- TopVC_x_Year: 0.0064

### Top Temporal Features
- Year_Founded: 0.7878
- Era_2010-2014: 0.0120

### Top Interaction Features
None in top 25

## Conclusions

### Investor Features Contribution
Investor features add minimal value when temporal features are present

### Interaction Features Value
Interactions provide minimal benefit and may cause overfitting

### Final Recommendation
Use **All Features (Full Model)** configuration with 32 features to achieve R² = 0.8549.

## Strategic Implications

1. **Temporal factors are PRIMARY**: When a company was founded is the strongest predictor
2. **Investor factors are SECONDARY**: Investor quality/quantity adds value but is secondary
3. **Interactions matter**: Interactions provide minimal additional value
4. **Both matter**: Focus on temporal features, investor features are less critical

## Detailed Results

See `output/data/investor_temporal_detailed.pkl` for full model results and feature lists.
See `output/data/investor_temporal_feature_importance.csv` for complete feature importance rankings.
