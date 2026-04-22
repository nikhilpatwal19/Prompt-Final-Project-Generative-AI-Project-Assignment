# 🏦 AI Financial Fragility Detector

**Early Warning Signals in Regional Banks**

An AI-powered system that detects structural financial weakness in regional banks before stress becomes visible in market prices. Combines quantitative financial modeling, NLP analysis of SEC filings, and three generative AI components.

> *"A bank does not collapse in silence. Its ratios drift. Its language tightens. Its balance sheet narrows. The signals are there. This system does not claim prophecy — it claims measurement."*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Generative AI Components](#generative-ai-components)
- [System Architecture](#system-architecture)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Key Results](#key-results)
- [Technologies Used](#technologies-used)
- [Ethical Considerations](#ethical-considerations)
- [Author](#author)

---

## Overview

Traditional LLMs can summarize a 10-K filing, but they cannot:
- Normalize financial stress across institutions
- Quantify structural vulnerability over time
- Detect deteriorating liquidity patterns
- Convert risk-language density into measurable exposure
- Produce a comparable fragility index

This system addresses that gap by combining **structured financial data** (FDIC quarterly reports) with **unstructured text analysis** (SEC 10-K filings) to produce a ranked fragility index for 19 regional banks across 2018–2023.

---

## Features

- **Financial Ratio Engine** — Computes liquidity, leverage, interest coverage, and deposit risk ratios from FDIC data
- **NLP Stress Scoring** — Analyzes 10-K Risk Factors using hedging term density, VADER sentiment, and section growth
- **RAG Knowledge Base** — ChromaDB vector database of chunked SEC filings with multi-strategy retrieval
- **LLM Risk Scoring** — Prompt-engineered multi-dimensional fragility assessment via Groq/Llama
- **Synthetic Data Augmentation** — 80 LLM-generated bank profiles across 5 fragility archetypes
- **Data-Driven Weights** — Logistic Regression-derived fragility index replacing hand-picked weights
- **ML Classification** — 3 models (Logistic Regression, Gradient Boosting, Random Forest) with full evaluation
- **Backtesting Framework** — Validates whether the index detects known failures before they happen
- **Interactive Dashboards** — 3 Plotly HTML dashboards + 11 static visualizations

---

## Generative AI Components

### 1. Retrieval-Augmented Generation (RAG)
- **Knowledge Base:** 10-K Risk Factors and MD&A sections for 19 banks
- **Chunking:** 500-word overlapping chunks with 100-word overlap
- **Embedding:** all-MiniLM-L6-v2 sentence transformer (local)
- **Vector DB:** ChromaDB with metadata filtering
- **Enhanced Retrieval:** 3 strategies (bank-specific, general fallback, ratio-informed) achieving 100% coverage

### 2. Prompt Engineering
- System prompt defining a financial risk analyst persona
- Structured context injection of financial ratios
- Multi-dimensional JSON scoring (liquidity, leverage, deposit risk, interest rate, textual stress)
- Comparative analysis prompts (failed vs stable banks)
- Edge case handling with JSON parsing fallbacks

### 3. Synthetic Data Generation
- Addresses severe class imbalance (only 3–4 real failures)
- 5 fragility archetypes: Liquidity Crunch, Deposit Flight, Leverage Overload, Silent Deterioration, Interest Rate Squeeze
- 80 synthetic profiles (50 distressed, 30 stable)
- Distribution-validated against real data
- Training set expanded from 112 → 192 observations

---

## System Architecture

```
FDIC API ─────────┐
                   ├──▶ Financial Ratios ──┐
SEC EDGAR ────────┤                         ├──▶ Fragility Index ──▶ Rankings
  (10-K text) ────┤──▶ NLP Scoring ────────┤                         │
                   │──▶ RAG + LLM Scoring ──┘                         │
Yahoo Finance ────┘                                                    │
                                                                       ▼
Synthetic Data ──▶ Augmented Training ──▶ ML Models ──▶ Evaluation
```

**Pipeline Stages:**
1. Data Collection (FDIC API, SEC EDGAR, Yahoo Finance)
2. Financial Ratio Engineering (4 core + supplementary ratios)
3. NLP Textual Stress Analysis (VADER, hedging density, section growth)
4. RAG Knowledge Base (ChromaDB vector search)
5. LLM Prompt-Engineered Scoring (Groq/Llama 3.3-70B)
6. Synthetic Data Generation (5 archetypes, 80 profiles)
7. ML Classification & Evaluation (3 models, cross-validated)

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Jupyter Notebook or VS Code with Jupyter extension
- ~2GB disk space for dependencies and data

### 1. Clone the Repository

```bash
git clone https://github.com/nikhilpatwal19/Final-Project-Generative-AI-Project-Assignment.git
cd Final-Project-Generative-AI-Project-Assignment
```

### 2. Install Dependencies

```bash
pip install pandas numpy requests beautifulsoup4 nltk scikit-learn matplotlib seaborn yfinance plotly lxml html5lib tqdm openpyxl groq chromadb sentence-transformers
```

### 3. API Keys

**Groq API (free):** Get a key at https://console.groq.com
- Used for LLM scoring and synthetic data generation
- Free tier: 30 requests/minute, 14,400/day

**SEC EDGAR:** No key needed, but update the `User-Agent` header in Cell 2 with your name and email:
```python
EDGAR_HEADERS = {
    'User-Agent': 'Your Name your_email@university.edu',
}
```

### 4. Configure API Key

In Cell 17 of the notebook, paste your Groq API key:
```python
GROQ_API_KEY = "your_groq_api_key_here"
```

---

## How to Run

1. Open `AI_Financial_Fragility_Detector.ipynb` in Jupyter or VS Code
2. Update the EDGAR User-Agent (Cell 2) and Groq API key (Cell 17)
3. Run all cells sequentially (Cell 1 through Cell 23)
4. Total runtime: ~15–20 minutes (mostly API calls with rate limiting)
5. Results will be saved to `data/` and `outputs/` directories

### Run Tests

After running the notebook, validate all outputs with the test suite (26 tests across 3 suites):

```bash
python tests/run_all_tests.py
```

Individual suites can also be run separately:
```bash
python tests/test_ratios.py          # Financial ratio validation (7 tests)
python tests/test_synthetic_data.py  # Synthetic data checks (8 tests)
python tests/test_pipeline.py        # RAG, scoring & output tests (11 tests)
```

### Live Demo

The Streamlit app is deployed at: **[aifinancialfragilitydetector.streamlit.app](https://aifinancialfragilitydetector.streamlit.app/)**

To run locally:
```bash
streamlit run app.py
```

### Cell Overview

| Cell | Purpose |
|------|---------|
| 1 | Install libraries |
| 2 | Import and configure |
| 3 | Define 19 target banks |
| 4 | Pull FDIC financial data |
| 5 | Compute vulnerability ratios |
| 6 | Visualize ratio trends |
| 7 | Pull 10-K text from SEC EDGAR |
| 8 | NLP textual stress scoring |
| 9 | Stock price drawdown analysis |
| 10 | Merge all data sources |
| 11 | Build fragility index (V1) |
| 12 | Visualize fragility scores |
| 13 | Train ML models |
| 14 | Backtesting & evaluation |
| 15 | Interactive Plotly dashboards |
| 16 | Export final summary |
| 17 | Setup Groq + ChromaDB |
| 18 | RAG knowledge base & assessment |
| 19 | Prompt engineering scoring |
| 20 | Synthetic data generation |
| 21 | Data-driven weights (V2) |
| 22 | Enhanced RAG (full coverage) |
| 23 | Comprehensive model evaluation |

---

## Project Structure

```
ai-financial-fragility-detector/
├── AI_Financial_Fragility_Detector.ipynb   # Main notebook (44 cells)
├── README.md                                # This file
├── app.py                                   # Streamlit web application
├── requirements.txt                         # Python dependencies
├── AI_Financial_Fragility_Detector_Report.pdf  # Project report
├── index.html                               # GitHub Pages web page
├── .gitignore
│
├── tests/                                   # Test scripts
│   ├── run_all_tests.py                    # Run all 26 tests at once
│   ├── test_ratios.py                      # Financial ratio validation (7 tests)
│   ├── test_synthetic_data.py              # Synthetic data checks (8 tests)
│   └── test_pipeline.py                    # RAG, scoring & output tests (11 tests)
│
├── examples/                                # Curated example outputs
│   ├── example_fragility_rankings.txt      # Final bank rankings by fragility
│   ├── example_rag_assessment.txt          # Sample RAG assessment (Signature Bank)
│   ├── example_llm_scores.csv             # All LLM fragility scores
│   └── example_synthetic_profiles.csv     # Sample synthetic bank profiles
│
├── data/                                    # Generated data files
│   ├── fdic_financials_raw.csv             # Raw FDIC quarterly data
│   ├── fdic_financials_with_ratios.csv     # Data with computed ratios
│   ├── edgar_10k_text_raw.csv              # Extracted 10-K text
│   ├── edgar_textual_stress_scores.csv     # NLP stress scores
│   ├── stock_drawdowns.csv                 # Stock drawdown analysis
│   ├── merged_feature_matrix.csv           # Combined feature set
│   ├── fragility_scored.csv                # Final fragility scores (V1 + V2)
│   ├── rag_risk_assessments.csv            # Original RAG assessments
│   ├── enhanced_rag_assessments.csv        # Enhanced RAG (100% coverage)
│   ├── llm_fragility_scores.csv            # LLM prompt-engineered scores
│   ├── synthetic_bank_profiles.csv         # 80 synthetic profiles
│   └── augmented_training_data.csv         # Real + synthetic training set
│
├── outputs/                                 # Visualizations & dashboards
│   ├── ratio_trends.png                    # Financial ratio trends
│   ├── fragility_heatmap.png               # Fragility score heatmap
│   ├── fragility_components.png            # Score component breakdown
│   ├── fragility_timeseries.png            # Fragility over time
│   ├── feature_importance.png              # ML feature importance
│   ├── llm_vs_rulebased.png               # LLM vs rule-based comparison
│   ├── real_vs_synthetic_distributions.png # Synthetic data validation
│   ├── original_vs_augmented_importance.png# Training comparison
│   ├── learned_weights_comparison.png      # V1 vs V2 weights
│   ├── comprehensive_model_evaluation.png  # ROC + confusion + CV
│   ├── precision_recall_curves.png         # PR curves
│   ├── fragility_dashboard.html            # Interactive timeline
│   ├── fragility_breakdown.html            # Interactive components
│   └── fragility_vs_roa.html              # Fragility vs profitability
│
└── Datasets_Prompt_Final_Project/           # Source datasets
    ├── Financial_4_21_2026.csv             # FDIC quarterly financials
    ├── Financial_4_21_2026 (1).csv         # FDIC quarterly financials
    ├── institutions.csv                     # FDIC bank list
    ├── institutions_definitions.csv         # Field definitions
    ├── download-data.csv                    # FDIC failed bank list
    └── All Financial Reports.xlsx           # FDIC API reference URLs
```

---

## Data Sources

| Source | Type | Access | Cost |
|--------|------|--------|------|
| FDIC BankFind API | Quarterly financials | Free, no key | Free |
| SEC EDGAR | 10-K filings (text) | Free, User-Agent required | Free |
| Yahoo Finance | Stock prices | Free via yfinance | Free |
| FDIC Failed Bank List | Ground truth labels | Free CSV download | Free |
| Groq API | LLM inference | Free tier (14,400 req/day) | Free |

---

## Key Results

- **19 banks analyzed** across 2018–2023 (112 real observations)
- **Signature Bank** correctly ranked 3rd most fragile (V2 score: 0.671)
- **First Republic Bank** correctly ranked 4th (V2 score: 0.640)
- **80 synthetic profiles** generated across 5 fragility archetypes
- **100% RAG coverage** with enhanced 3-strategy retrieval
- **17 banks** scored by LLM with structured prompt engineering
- **44 notebook cells**, zero errors, 14 visualizations generated

---

## Technologies Used

- **Python 3.12** — Core language
- **pandas, numpy** — Data manipulation
- **scikit-learn** — ML models and evaluation
- **matplotlib, seaborn, plotly** — Visualization
- **NLTK (VADER)** — Sentiment analysis
- **BeautifulSoup** — HTML/filing parsing
- **ChromaDB** — Vector database for RAG
- **sentence-transformers** — Text embedding
- **Groq API (Llama 3.3-70B)** — LLM inference
- **yfinance** — Stock price data
- **requests** — API calls
- **reportlab** — PDF report generation

---

## Ethical Considerations

- All data from public regulatory filings — no private data used
- Synthetic data explicitly labeled (`is_synthetic=True`)
- Not investment advice — academic exercise only
- Fragility index measures vulnerability, not predicting collapse
- LLM outputs may contain biases from training data
- All code documented and reproducible

---

## Author

**Nikhil Patwal**
Northeastern University
April 2026

---

## License

This project is an academic submission. All code is original. External libraries and data sources are cited above.
