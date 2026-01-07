# Strategic Management in Unicorn Companies: Lessons and Insights

A comprehensive data science project analyzing strategic factors that influence unicorn company growth speed and success, using advanced machine learning and strategic management frameworks.

## 📋 Project Overview

This project investigates what makes unicorn companies successful by:
1. **Data Preprocessing**: Cleaning and engineering comprehensive features from unicorn company data
2. **Machine Learning**: Building production-ready models to predict time to unicorn status (R² = 0.8545)
3. **Strategic Analysis**: Empirically validating Porter's Five Forces, Resource-Based View, and Network Effects theories with effect sizes and ML integration

## 🎯 Objectives

1. **Data Augmentation**: Systematically web scrape founding years for 1,295 companies from multiple sources to enable growth analysis
2. **Growth Speed Prediction**: Estimate time-to-unicorn (years from founding to $1B valuation) through regression analysis
3. **Factor Importance Ranking**: Quantify which variables most strongly predict success
4. **Theoretical Validation**: Interpret findings through Porter's Five Forces, Resource-Based View, and network effects frameworks

## 📁 Project Structure

```
Unicorn/
├── step1_preprocessing.py              # Data cleaning and feature engineering
├── step2_ml_models_final.py           # Final production ML model (R² = 0.8545)
├── step2_temporal_analysis.py         # Temporal features impact analysis
├── step2_investor_temporal_analysis.py # Investor + Temporal comprehensive analysis
├── step3_porters_analysis.py          # Enhanced strategic framework analysis
├── generate_impact_reports.py         # Generate targeted reports for stakeholders
├── unicorn_scraper.py                 # Web scraper for founding years
├── validate_data.py                   # Data quality validation
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── output/                            # All generated outputs
│   ├── models/                        # Saved model files
│   │   ├── final_production_model.pkl # Production-ready model (R² = 0.8545)
│   │   └── preprocessed_data.pkl     # Preprocessed datasets
│   ├── visualizations/                # Charts and graphs
│   │   ├── final_model_results.png    # Final model comprehensive dashboard
│   │   └── enhanced_strategic_framework_dashboard.png # Enhanced framework analysis
│   ├── data/                          # Processed data files
│   │   ├── final_model_comparison.csv # Model performance comparison
│   │   ├── final_feature_importance.csv # Feature importance rankings
│   │   ├── enhanced_theoretical_validation_results.csv # Framework validation
│   │   └── [other analysis results]
│   └── reports/                       # Comprehensive analysis reports
│       ├── FINAL_MODEL_DOCUMENTATION.md # Complete model documentation
│       ├── ENHANCED_STRATEGIC_FRAMEWORK_ANALYSIS.md # Framework analysis
│       ├── ENTREPRENEUR_INSIGHTS.md   # Insights for entrepreneurs
│       ├── INVESTOR_FRAMEWORK.md      # Investment framework
│       └── [other detailed reports]
```


## 🚀 Quick Start


Follow these steps to set up and run the complete analysis pipeline.

### Prerequisites

- Python 3.8+ (check with `python --version`)
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aiguanai/Predicting-Unicorn-Success-Using-Machine-Learning.git
   cd Unicorn
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate.ps1
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup

Place `CB-Insights_Global-Unicorn-Club_2025.xlsx` in the project root directory.

> **Note:** `unicorn_data_augmented.xlsx` will be created automatically during preprocessing if missing.

### Run Analysis Pipeline

Execute these scripts in order:

**Step 1: Preprocess Data**
```bash
python step1_preprocessing.py
```
- Cleans data and engineers 45+ features
- Creates train/test splits (80/20)
- Output: `output/models/preprocessed_data.pkl`

**Step 2: Train ML Models**
```bash
python step2_ml_models_final.py
```
- Trains 6 ML algorithms with hyperparameter tuning
- Best model: Ridge Regression (R² = 0.8545, RMSE = 1.94 years)
- Output: `output/models/final_production_model.pkl`

**Step 3: Strategic Analysis**
```bash
python step3_porters_analysis.py
```
- Validates Porter's Five Forces, RBV, and Network Effects frameworks
- Generates visualizations and reports
- Output: `output/reports/` and `output/visualizations/`

**Optional: Generate Impact Reports**
```bash
python generate_impact_reports.py
```
- Creates stakeholder-specific reports (Entrepreneurs, Investors, Researchers)

## 📊 Output Files

All outputs are organized in the `output/` directory:

### Models (`output/models/`)
- `final_production_model.pkl` - **Production-ready model** (R² = 0.8545, RMSE = 1.94 years)
- `preprocessed_data.pkl` - Preprocessed datasets with train/test splits

### Visualizations (`output/visualizations/`)
- `final_model_results.png` - Comprehensive model performance dashboard
- `enhanced_strategic_framework_dashboard.png` - Enhanced framework analysis with effect sizes

### Data (`output/data/`)
- `final_model_comparison.csv` - Model performance metrics comparison
- `final_feature_importance.csv` - Complete feature importance rankings
- `final_predictions.csv` - Model predictions on test set
- `enhanced_theoretical_validation_results.csv` - Framework validation with effect sizes

### Reports (`output/reports/`)
- `FINAL_MODEL_DOCUMENTATION.md` - Complete production model documentation
- `ENHANCED_STRATEGIC_FRAMEWORK_ANALYSIS.md` - Enhanced framework analysis
- `ENTREPRENEUR_INSIGHTS.md` - Insights for entrepreneurs
- `INVESTOR_FRAMEWORK.md` - Investment framework and criteria
- `FINAL_REPORT.md` - Executive summary and key findings

## 🔍 Key Features Engineered

### Temporal Features (84.5% importance - PRIMARY)
- `Year_Founded`: Year company was founded (most important feature: 78.68% importance)
- `Era_Pre-2000`, `Era_2000-2009`, `Era_2010-2014`, `Era_2015-2019`, `Era_2020+`: Founding era categories
- `Investors_x_Year`: Investor count × Year interaction
- `TopVC_x_Year`: Top VC × Year interaction

### Investor Features (8.0% importance - SECONDARY)
- `Investor_Count`: Number of investors
- `Has_Top_VC`: Has top-tier VC backing (Sequoia, a16z, Tiger Global, etc.)
- `Val_per_Investor`: Valuation per investor (capital efficiency)
- `Investor_Efficiency`: Investor efficiency metric

### Geographic Features (2.2% importance)
- `Is_Tech_Hub`: Located in major tech hub (SF, NYC, Beijing, etc.)
- `Is_Silicon_Valley`: Located in Silicon Valley
- `Country_Tier`: Country classification (Tier 1: US/China, Tier 2: Major economies, Tier 3: Others)

### Industry Features (1.7% importance)
- `Industry_Group`: Categorized industries (Fintech, Enterprise Tech, AI/ML, Healthcare, etc.)
- `Is_Tech_Intensive`: High-tech industry indicator

### Valuation Features (3.3% importance)
- `Valuation ($B)`: Company valuation in billions
- `Log_Valuation`: Log-transformed valuation

## 📈 Model Performance

### Final Production Model

**Best Model**: Ridge Regression
- **R² Score**: 0.8545 (85.45% variance explained)
- **RMSE**: 1.94 years
- **MAE**: 1.45 years
- **CV R²**: 0.8158 (excellent generalization)
- **Overfitting Gap**: -0.0387 (minimal overfitting)

### Model Comparison

The project trains and compares multiple algorithms:
- **Ridge Regression**: L2 regularization (BEST: R² = 0.8545)
- **Ensemble**: Voting regressor of multiple models (R² = 0.8435)
- **XGBoost**: Optimized gradient boosting (R² = 0.8350)
- **Random Forest**: Ensemble tree-based model (R² = 0.8246)
- **Gradient Boosting**: Sequential tree boosting (R² = 0.8165)

## 🎓 Strategic Frameworks Tested

### Validation Results

**2/8 frameworks empirically validated** (25.0%)

**Validated Frameworks**:
1. **Porter Force 2: Threat of New Entrants (Entry Barriers)**
   - P-value: < 0.00001
   - Effect Size: 0.0152 (small)
   - Companies with more investors (higher capital intensity) achieve higher valuations

2. **RBV: VC Network Resource**
   - P-value: 0.0002
   - Effect Size: 0.2123 (small)
   - Top-tier VC backing significantly accelerates growth (1-2 years faster on average)

**Not Validated Frameworks**:
- Porter Force 1: Competitive Rivalry
- Porter Force 3: Supplier Power
- Porter Force 4: Buyer Power
- Porter Force 5: Substitutes
- RBV: Geographic Resource
- Network Effects Theory

## 📝 Key Findings

### Machine Learning Insights

1. **Temporal Features Dominate**: When a company was founded is the single most important factor (78.68% importance)
2. **Investor Features Add Value**: Top VC backing and investor count matter, but timing matters more (8.0% importance)
3. **Excellent Model Performance**: R² = 0.8545 demonstrates strong predictive power
4. **Good Generalization**: CV R² = 0.8158 shows model generalizes well

### Strategic Management Insights

1. **Entry Barriers Matter**: High capital intensity (many investors) creates competitive advantage
2. **VC Network Is Valuable**: Top-tier VC backing accelerates growth speed
3. **Timing Is Critical**: Founding era is more important than investor quality
4. **Limited Framework Validation**: Only 2/8 frameworks validated, suggesting unicorn success is complex

## 🎯 Strategic Recommendations

Based on validated frameworks and ML insights:

### For Entrepreneurs
1. **Prioritize Timing**: Start your company in favorable market eras (2010-2014, 2015-2019)
2. **Build Entry Barriers**: Secure multiple rounds of funding to create competitive moat
3. **Target Top VCs**: Partner with Sequoia, a16z, Tiger Global for faster growth
4. **Consider Location**: Tech hubs provide some advantage, but less than timing
5. **Industry Selection**: Healthcare & Life Sciences (6.5 years) and Industrials (6.9 years) show fastest paths

### For Investors
1. **Validated Investment Criteria**: 
   - Entry barriers (4+ investors)
   - Top-tier VC backing
2. **Portfolio Construction**: Use validated frameworks for due diligence
3. **Timing Matters**: Consider founding era when evaluating companies

### For Researchers
1. **Web Scraping Methodology**: Repeatable framework for enhancing incomplete datasets
2. **Feature Engineering**: Temporal features are critical for growth prediction
3. **Framework Validation**: Empirical testing of strategic management theories at scale

## 🛠️ Dependencies

See `requirements.txt` for full list. Key packages:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0
- xgboost >= 2.0.0
- requests >= 2.31.0 (for web scraping)
- beautifulsoup4 >= 4.12.0 (for web scraping)
- openpyxl >= 3.1.0 (for Excel files)

## 🔬 Technical Details

- **Dataset**: 1,271 unicorn companies (after quality filtering from 1,290)
- **Train-Test Split**: 80/20 with random seed 42 for reproducibility
- **Temporal Leakage Prevention**: Date_Joined_Year and Company_Age_2025 removed (direct leakage)
- **Safe Temporal Features**: Year_Founded and Era_* features are safe and highly predictive
- **Feature Selection**: Intelligent selection using correlation, mutual information, and F-statistic
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Statistical Significance**: Tested at α = 0.05 with effect sizes (Cohen's d, Eta-squared)
- **Data Quality Filters**: Missing target variables, invalid growth duration (>50 years), missing critical features

## 📚 Documentation

Comprehensive documentation is available in `output/reports/`:
- **FINAL_MODEL_DOCUMENTATION.md**: Complete model usage guide
- **ENHANCED_STRATEGIC_FRAMEWORK_ANALYSIS.md**: Detailed framework analysis
- **ENTREPRENEUR_INSIGHTS.md**: Insights and recommendations for entrepreneurs
- **INVESTOR_FRAMEWORK.md**: Investment framework and validated criteria
- **FINAL_REPORT.md**: Executive summary and key findings

## 🏆 Project Highlights

- ✅ **Production-Ready Model**: R² = 0.8545, RMSE = 1.94 years
- ✅ **Comprehensive Analysis**: 45 features across 5 categories
- ✅ **Enhanced Framework Validation**: Effect sizes and ML integration
- ✅ **Extensive Documentation**: Multiple detailed reports
- ✅ **Clean Codebase**: Organized structure with best practices

## 📊 Dataset Information

- **Source**: CB Insights Global Unicorn Club 2025
- **Original Size**: 1,290 companies
- **Final Size**: 1,271 companies (after quality filtering)
- **Data Enhancement**: Web scraping from Crunchbase, Wikipedia, Google Search, LinkedIn, company websites
- **Geographic Coverage**: Global (US, China, India, UK, and others)
- **Industries**: Fintech, Enterprise Tech, AI/ML, Healthcare, E-commerce, Media, Mobility, and more



## 📄 License

This project is for educational/research purposes.

## 🙏 Acknowledgments

- **CB Insights** - For providing the Global Unicorn Club 2025 dataset
- **Open-Source Community** - For tools and libraries that enabled this research


---

**Last Updated**: 2025  
**Status**: Production Ready  
**Model Performance**: Excellent (R² = 0.8545)
