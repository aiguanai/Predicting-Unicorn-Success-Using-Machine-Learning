# Complete Feature Documentation for ML Models

This document provides a comprehensive overview of all features used in the machine learning models for predicting unicorn company growth speed.

## Feature Categories Overview

The models use **45+ features** organized into 5 main categories:
1. **Temporal Features** (84.5% importance) - PRIMARY
2. **Investor Features** (8.0% importance) - SECONDARY  
3. **Geographic Features** (2.2% importance)
4. **Industry Features** (1.7% importance)
5. **Valuation Features** (3.3% importance)

---

## 1. Temporal Features (84.5% Total Importance)

### Base Temporal Features

#### `Year_Founded`
- **Type**: Numeric (integer)
- **Description**: The year the company was founded
- **Importance**: **78.68%** (most important single feature)
- **Range**: 1950-2023
- **Usage**: Direct input to model; captures market timing effects

#### `Era_Pre-2000`
- **Type**: Binary (0 or 1, one-hot encoded)
- **Description**: Company founded before year 2000
- **Usage**: Categorical era representation

#### `Era_2000-2009`
- **Type**: Binary (0 or 1, one-hot encoded)
- **Description**: Company founded between 2000-2009
- **Usage**: Categorical era representation

#### `Era_2010-2014`
- **Type**: Binary (0 or 1, one-hot encoded)
- **Description**: Company founded between 2010-2014
- **Usage**: Categorical era representation (often fastest growth era)

#### `Era_2015-2019`
- **Type**: Binary (0 or 1, one-hot encoded)
- **Description**: Company founded between 2015-2019
- **Usage**: Categorical era representation

#### `Era_2020+`
- **Type**: Binary (0 or 1, one-hot encoded)
- **Description**: Company founded in 2020 or later
- **Usage**: Categorical era representation

### Temporal Interaction Features

#### `Investors_x_Year`
- **Type**: Numeric
- **Description**: Interaction between investor count and founding year
- **Formula**: `Investor_Count × Year_Founded`
- **Purpose**: Captures how investor effects vary by founding era

#### `TopVC_x_Year`
- **Type**: Numeric
- **Description**: Interaction between top VC presence and founding year
- **Formula**: `Has_Top_VC × Year_Founded`
- **Purpose**: Captures how VC quality effects vary by founding era

#### `Investors_x_Era_*`
- **Type**: Numeric (multiple features)
- **Description**: Interaction between investor count and each era category
- **Examples**: `Investors_x_Era_2010-2014`, `Investors_x_Era_2015-2019`
- **Formula**: `Investor_Count × Era_*`
- **Purpose**: Captures era-specific investor effects

#### `TopVC_x_Era_*`
- **Type**: Numeric (multiple features)
- **Description**: Interaction between top VC presence and each era category
- **Examples**: `TopVC_x_Era_2010-2014`, `TopVC_x_Era_2015-2019`
- **Formula**: `Has_Top_VC × Era_*`
- **Purpose**: Captures era-specific VC quality effects

---

## 2. Investor Features (8.0% Total Importance)

### Base Investor Features

#### `Investor_Count`
- **Type**: Numeric (integer)
- **Description**: Number of investors backing the company
- **Calculation**: Count of commas in "Select Investors" field + 1
- **Range**: 0-50+ investors
- **Importance**: Validated in Porter's Five Forces (Entry Barriers)

#### `Has_Top_VC`
- **Type**: Binary (0 or 1)
- **Description**: Whether company has backing from top-tier VCs
- **Top VCs**: Sequoia, Andreessen Horowitz (a16z), Tiger Global, SoftBank, Accel, Benchmark, Insight Partners, General Catalyst, Lightspeed, Greylock, Kleiner Perkins, Khosla
- **Importance**: Validated in RBV framework (VC Network Resource)
- **Effect**: Companies with top VCs reach unicorn status 1-2 years faster

### Derived Investor Features

#### `Log_Investors`
- **Type**: Numeric
- **Description**: Log-transformed investor count (log(1 + Investor_Count))
- **Purpose**: Handles skewed distribution of investor counts

#### `Investors_Squared`
- **Type**: Numeric
- **Description**: Squared investor count (Investor_Count²)
- **Purpose**: Captures non-linear effects of investor count

#### `Val_per_Investor`
- **Type**: Numeric
- **Description**: Valuation per investor (capital efficiency metric)
- **Formula**: `Valuation ($B) / (Investor_Count + 1)`
- **Purpose**: Measures how efficiently capital is deployed

#### `Investor_Efficiency`
- **Type**: Numeric
- **Description**: Investor efficiency metric (inverse of capital efficiency)
- **Formula**: `Investor_Count / (Valuation ($B) + 0.1)`
- **Purpose**: Alternative measure of investor efficiency

#### `Val_x_Investors`
- **Type**: Numeric
- **Description**: Interaction between valuation and investor count
- **Formula**: `Valuation ($B) × Investor_Count`
- **Purpose**: Captures valuation-investor relationships

#### `VC_Quality_Score`
- **Type**: Numeric
- **Description**: Combined VC quality indicator
- **Formula**: `Has_Top_VC × Investor_Count`
- **Purpose**: Combines VC quality with investor count

---

## 3. Geographic Features (2.2% Total Importance)

### Base Geographic Features

#### `Is_Tech_Hub`
- **Type**: Binary (0 or 1)
- **Description**: Whether company is located in a major tech hub
- **Tech Hubs**: San Francisco, Palo Alto, Mountain View, Menlo Park, San Jose, New York, Beijing, Shenzhen, Bangalore, London, Tel Aviv, Boston, Seattle, Los Angeles
- **Purpose**: Captures advantages of being in innovation clusters

#### `Is_Silicon_Valley`
- **Type**: Binary (0 or 1)
- **Description**: Whether company is located in Silicon Valley specifically
- **Cities**: San Francisco, Palo Alto, Mountain View, Menlo Park, San Jose
- **Purpose**: Captures premium of being in the world's leading tech hub

#### `Country_Tier`
- **Type**: Numeric (1, 2, or 3)
- **Description**: Country classification based on startup ecosystem strength
- **Tier 1** (Value: 1): United States, China
- **Tier 2** (Value: 2): India, United Kingdom, Germany, Israel, Singapore, South Korea, Japan, Canada, France, Sweden
- **Tier 3** (Value: 3): All other countries
- **Purpose**: Captures country-level advantages

### Derived Geographic Features

#### `Geo_Advantage`
- **Type**: Numeric
- **Description**: Combined geographic advantage score
- **Formula**: `Is_Tech_Hub + Is_Silicon_Valley + (4 - Country_Tier)`
- **Purpose**: Single metric capturing all geographic advantages
- **Range**: 0-6 (higher = better geographic position)

### Geographic Interaction Features

#### `Hub_x_TopVC`
- **Type**: Binary (0 or 1)
- **Description**: Interaction between tech hub location and top VC backing
- **Formula**: `Is_Tech_Hub × Has_Top_VC`
- **Purpose**: Captures synergy between location and investor quality

#### `Valley_x_TopVC`
- **Type**: Binary (0 or 1)
- **Description**: Interaction between Silicon Valley location and top VC backing
- **Formula**: `Is_Silicon_Valley × Has_Top_VC`
- **Purpose**: Captures premium of Silicon Valley + top VC combination

#### `CountryTier_x_TopVC`
- **Type**: Numeric
- **Description**: Interaction between country tier and top VC backing
- **Formula**: `Country_Tier × Has_Top_VC`
- **Purpose**: Captures how VC quality effects vary by country tier

#### `CountryTier_x_Investors`
- **Type**: Numeric
- **Description**: Interaction between country tier and investor count
- **Formula**: `Country_Tier × Investor_Count`
- **Purpose**: Captures how investor count effects vary by country tier

---

## 4. Industry Features (1.7% Total Importance)

### Base Industry Features

#### `Industry_Group` (One-Hot Encoded)
- **Type**: Binary (0 or 1, multiple features)
- **Description**: Industry category classification
- **Categories** (one-hot encoded as `Ind_*`):
  - `Ind_Fintech`: Financial technology, payments, banking, insurance
  - `Ind_Enterprise_Tech`: Software, SaaS, enterprise tech, data, cloud, cybersecurity
  - `Ind_AI_ML`: Artificial intelligence, machine learning
  - `Ind_Healthcare`: Health, biotech, medical, pharma
  - `Ind_E-commerce`: E-commerce, retail, marketplace, consumer
  - `Ind_Media`: Media, entertainment, gaming, content
  - `Ind_Mobility`: Transportation, mobility, logistics, delivery, automotive
  - `Ind_Other`: All other industries (baseline, dropped in one-hot encoding)

#### `Is_Tech_Intensive`
- **Type**: Binary (0 or 1)
- **Description**: Whether company is in a high-tech industry
- **High-Tech Industries**: Enterprise_Tech, AI_ML, Fintech
- **Purpose**: Captures technology intensity effects

### Industry Interaction Features

#### `Hub_x_Fintech`
- **Type**: Binary (0 or 1)
- **Description**: Interaction between tech hub location and fintech industry
- **Formula**: `Is_Tech_Hub × Ind_Fintech`
- **Purpose**: Captures fintech hub advantages

#### `Hub_x_EnterpriseTech`
- **Type**: Binary (0 or 1)
- **Description**: Interaction between tech hub location and enterprise tech industry
- **Formula**: `Is_Tech_Hub × Ind_Enterprise_Tech`
- **Purpose**: Captures enterprise tech hub advantages

#### `TechIntensive_x_TopVC`
- **Type**: Binary (0 or 1)
- **Description**: Interaction between tech-intensive industry and top VC backing
- **Formula**: `Is_Tech_Intensive × Has_Top_VC`
- **Purpose**: Captures synergy between tech intensity and VC quality

---

## 5. Valuation Features (3.3% Total Importance)

### Base Valuation Features

#### `Valuation ($B)`
- **Type**: Numeric (float)
- **Description**: Company valuation in billions of dollars
- **Range**: $1B - $500B+
- **Purpose**: Direct measure of company size/value

#### `Log_Valuation`
- **Type**: Numeric
- **Description**: Natural log of valuation
- **Formula**: `log(Valuation ($B))`
- **Purpose**: Handles skewed distribution of valuations

### Derived Valuation Features

#### `Log_Val`
- **Type**: Numeric
- **Description**: Log(1 + valuation) transformation
- **Formula**: `log1p(Valuation ($B))`
- **Purpose**: Alternative log transformation (handles zero values)

#### `Val_Squared`
- **Type**: Numeric
- **Description**: Squared valuation term
- **Formula**: `Valuation ($B)²`
- **Purpose**: Captures non-linear valuation effects

#### `Valuation_Category` (One-Hot Encoded)
- **Type**: Binary (0 or 1, multiple features)
- **Description**: Categorical valuation classification
- **Categories** (one-hot encoded as `ValCat_*`):
  - `ValCat_Small`: < $5B
  - `ValCat_Medium`: $5B - $10B
  - `ValCat_Large`: $10B - $50B
  - `ValCat_Mega`: $50B+ (baseline, dropped in one-hot encoding)

---

## Features Removed (Data Leakage Prevention)

These features are **excluded** from the model to prevent data leakage:

### `Date_Joined_Year`
- **Reason**: Directly reveals when company became unicorn (part of target calculation)
- **Status**: Removed

### `Company_Age_2025`
- **Reason**: Directly computes target variable (Years_to_Unicorn = Company_Age_2025 - Years_to_Unicorn)
- **Status**: Removed

---

## Feature Selection Process

The model uses intelligent feature selection to remove noise:

1. **Correlation Filtering**: Keeps features with correlation > 0.01 with target
2. **Mutual Information**: Keeps top 90% of features by mutual information score
3. **F-Statistic**: Keeps top 90% of features by F-statistic score
4. **Combined Selection**: Features selected by at least 2 methods are kept
5. **Key Feature Protection**: Essential features (Year_Founded, Era_*, Investor_Count, etc.) are always included

---

## Feature Importance Summary

Based on the final Ridge Regression model:

| Feature Category | Total Importance | Key Features |
|-----------------|------------------|--------------|
| **Temporal** | **84.5%** | Year_Founded (78.68%), Era_* features |
| **Investor** | **8.0%** | Investor_Count, Has_Top_VC, interactions |
| **Valuation** | **3.3%** | Valuation ($B), Log_Valuation |
| **Geographic** | **2.2%** | Is_Tech_Hub, Country_Tier |
| **Industry** | **1.7%** | Industry_Group, Is_Tech_Intensive |

---

## Model Performance with All Features

**Best Model**: Ridge Regression
- **R² Score**: 0.8545 (85.45% variance explained)
- **RMSE**: 1.94 years
- **MAE**: 1.45 years
- **CV R²**: 0.8158 (excellent generalization)

---

## Notes

- All numeric features are standardized (StandardScaler) before model training
- Categorical features are one-hot encoded with first category dropped
- Interaction features capture non-linear relationships between base features
- Feature selection ensures only predictive features are included while maintaining model interpretability

