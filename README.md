# 🛡️ TransactIQ — ML-Powered Sanctions Detection & Transaction Screening

> Graduate Project | BA775 Financial Analytics | Boston University

TransactIQ is a multi-page Streamlit application that screens financial transactions against global sanctions lists using machine learning. It layers ML-based composite scoring on top of fuzzy entity matching to reduce false positives while maintaining high detection rates — built for compliance officers, AML analysts, and financial crime investigators.

---

## Features

- **Transaction Screening** — Upload a CSV/XLSX or use built-in synthetic demo data. Runs each transaction through a full ML pipeline and returns color-coded risk levels (High / Grey / Low)
- **Risk Dashboard** — Interactive world choropleth heatmap, risk score histogram with Altman Z-Score zone analogy, and top flagged transactions table
- **Benford's Law Analysis** — Per-counterparty first-digit frequency analysis with chi-squared test to detect fabricated or manipulated transaction amounts
- **Model Performance** — Feature importance, logistic regression coefficients, ROC curves, and confusion matrices for both models
- **Counterparty Lookup** — Live financial data via yfinance with Du Pont ratio analysis and revenue-vs-transaction-volume mismatch detection
- **Data Formatter** — Reformat any external dataset by mapping its columns to the required schema, then load directly into the screener

---

## Course Concept Connections (BA775)

| Concept | Where It Appears | Lecture |
|---|---|---|
| Logit Regression | Model 1 for sanctions probability; coefficients on Model Performance page | Lecture 3 |
| Benford's Law | Dedicated analysis page; deviation used as ML feature; synthetic data generation | Lecture 4 |
| Altman Z-Score Zones | Risk score histogram High/Grey/Low zones | Lectures 3-4 |
| Financial Ratios / Du Pont | Counterparty Lookup page ratio analysis | Lecture 2 |
| Beneish M-Score | Overall fraud detection approach using ratio anomalies | Lecture 4 |
| yfinance | Live counterparty financial data retrieval | Lecture 3 |
| CAPM / Beta | Beta calculation on Counterparty Lookup page | Lecture 5 |

---

## Tech Stack

| Category | Libraries |
|---|---|
| Framework | Streamlit |
| ML | scikit-learn (LogisticRegression), XGBoost |
| Data | pandas, numpy |
| Visualization | Plotly, matplotlib, seaborn |
| Fuzzy Matching | rapidfuzz |
| Financial Data | yfinance |
| Statistics | scipy |
| File Handling | openpyxl |

---

## Project Structure

```
sanctions_app/
├── app.py                          # Main entry point, sidebar nav, shared state
├── pages/
│   ├── 1_Transaction_Screening.py  # Core screening page
│   ├── 2_Risk_Dashboard.py         # Heatmap, histogram, top flagged table
│   ├── 3_Benfords_Analysis.py      # Benford's Law digit analysis
│   ├── 4_Model_Performance.py      # Model metrics, feature importance
│   └── 5_Counterparty_Lookup.py   # yfinance counterparty financial lookup
├── data/
│   ├── generate_synthetic_data.py  # Generates 10,000 synthetic transactions
│   ├── load_sanctions_lists.py     # OFAC, UN, OpenSanctions loaders
│   └── world_bank.py               # World Bank governance score fetcher
├── models/
│   ├── train.py                    # Training pipeline (Logit + XGBoost)
│   ├── predict.py                  # Inference pipeline
│   ├── features.py                 # Feature engineering (9 features)
│   └── saved/                      # Trained model files (auto-generated)
├── utils/
│   ├── fuzzy_match.py              # Entity name matching logic
│   ├── benford.py                  # Benford's Law analysis utilities
│   └── constants.py                # Sanctioned country codes, thresholds
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- [Homebrew](https://brew.sh/) (Mac) — required for XGBoost dependency

### Installation

**1. Install OpenMP (required for XGBoost on Mac)**
```bash
brew install libomp
```

**2. Clone the repository**
```bash
git clone <your-repo-url>
cd sanctions_app
```

**3. Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**4. Install dependencies**
```bash
python3 -m pip install -r requirements.txt
```

**5. Run the app**
```bash
streamlit run app.py
```

On first launch the app will automatically generate 10,000 synthetic transactions and train both ML models. This takes approximately 1-2 minutes. Models are saved to `models/saved/` and reloaded on subsequent runs.

---

## ML Pipeline

### Features (9 engineered per transaction)

| Feature | Description |
|---|---|
| `sanctions_list_match_score` | Fuzzy match score of counterparty name against OFAC + UN + OpenSanctions |
| `country_risk_score` | World Bank governance composite (inverted — higher = riskier) |
| `sanctioned_country_flag` | Binary flag if receiver country is on OFAC/UN sanctioned list |
| `transaction_amount_log` | Log-transformed transaction amount |
| `currency_mismatch` | Binary flag if currency differs from expected local currency |
| `amount_to_revenue_ratio` | Transaction amount relative to counterparty revenue |
| `benford_deviation` | Chi-squared statistic of counterparty first-digit distribution |
| `transaction_frequency` | Normalized transaction count per counterparty |
| `high_risk_geo_interaction` | `country_risk_score × transaction_amount_log` interaction term |

### Models

- **Logistic Regression** — `sklearn.linear_model.LogisticRegression` with L2 regularization and balanced class weights
- **XGBoost** — `XGBClassifier` with 200 estimators, max depth 5, AUCPR evaluation metric
- **Composite Score** — `0.4 × logit_proba + 0.6 × xgb_proba` (XGBoost weighted higher for non-linear interaction capture)

### Risk Zones (Altman Z-Score Analogy)

| Zone | Score Range | Altman Z Analogy |
|---|---|---|
| 🔴 High Risk | > 0.70 | Z < 1.80 (Distress Zone) |
| 🟡 Grey Zone | 0.40 – 0.70 | 1.80 < Z < 2.99 (Grey Zone) |
| 🟢 Low Risk | < 0.40 | Z > 2.99 (Safe Zone) |

*Thresholds are adjustable via the sidebar.*

---

## Data Sources

All data sources are free and require no API keys.

| Source | Purpose | Offline Fallback |
|---|---|---|
| OFAC SDN List (treasury.gov) | Primary sanctions entity list | 50 hardcoded entities |
| UN Consolidated Sanctions | Supplementary entity list | Hardcoded subset |
| OpenSanctions | Aggregated watchlist data | Hardcoded subset |
| World Bank API | Country governance risk scores | Hardcoded scores for 60+ countries |
| yfinance | Counterparty financial statements | Graceful error handling |

The app works fully offline using synthetic demo data and hardcoded fallback data.

---

## Expected Data Schema

When uploading your own transaction data, the following columns are required:

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | Unique transaction identifier (auto-generated if missing) |
| `date` | date | Transaction date (YYYY-MM-DD) |
| `sender_name` | string | Sending entity name |
| `sender_country` | string | Sender ISO-2 country code (e.g. US, GB) |
| `receiver_name` | string | Receiving entity name |
| `receiver_country` | string | Receiver ISO-2 country code |
| `amount` | float | Transaction amount |
| `currency` | string | Currency code (e.g. USD, EUR) |

Use the **Format Data** tab on the Transaction Screening page to reformat datasets that use different column names.

---

## Offline Mode

TransactIQ is designed to work fully offline:
- Sanctions list downloads fail gracefully, falling back to 50 hardcoded known entities
- World Bank API calls fall back to hardcoded governance scores for 60+ countries
- yfinance failures are caught and displayed as warnings
- All ML models are trained locally on synthetic data

---

*Built with Python, Streamlit, scikit-learn, and XGBoost.*
