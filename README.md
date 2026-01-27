# Summit County Housing: Analytics & ML Portfolio
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![SQL](https://img.shields.io/badge/SQL-Advanced-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-green)

A production-grade analytics platform that combines **Advanced SQL Engineering** with **Deep Learning** to analyze and predict real estate trends in Summit County, CO.

This project demonstrates the full Data Science lifecycle:
1.  **Ingestion**: Scraping and validating property records (~40k rows).
2.  **Engineering**: Building a normalized SQLite warehouse with complex window functions.
3.  **Modeling**: Training Neural Networks and Gradient Boosting models for price prediction.
4.  **Deployment**: Serving insights via an interactive Streamlit dashboard.

## 🚀 Key Technical Features

### 1. Advanced Machine Learning & Interpretability
Beyond simple regression, this project leverages modern ML techniques to understand market drivers:
-   **Neural Network (PyTorch)**: Custom feed-forward architecture for high-accuracy price prediction (`src/summit_housing/models.py`).
-   **Explainable AI (XAI)**:
    -   **SHAP Values**: Deconstructs individual predictions (e.g., "This home is $50k cheaper because of its distance to Breckenridge").
    -   **Partial Dependence Plots (PDP)**: Visualizes non-linear relationships between features like Square Footage and Price.
-   **Geospatial Feature Engineering**: Calculates haversine distances to major ski resorts (Breckenridge, Keystone, Copper) to quantify location value.

### 2. Advanced SQL Analytics
Logic is pushed to the database layer to ensure performance and demonstrate SQL mastery:
-   **Window Functions**: Uses `LAG()` and `LEAD()` to calculate appreciation rates and holding periods.
-   **Trend Smoothing**: Uses `AVG() OVER (ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)` for moving averages.
-   **Complex Unpivoting**: Transforms denormalized raw CSV logs into a clean transaction stream using CTEs and `UNION ALL`.

### 3. Resilient Data Pipeline (The "DLQ" Pattern)
Real-world data is messy. The pipeline (`src/summit_housing/ingestion.py`) implements a **Dead Letter Queue** pattern:
-   **Validation**: Every record is checked against strict **Pydantic** models.
-   **Fault Tolerance**: The pipeline fails gracefully. Malformed records (bad dates, mixed types) are quarantined in `data/rejected_records.csv` for audit, ensuring good data isn't blocked.

### 4. Production Engineering
-   **Dockerized**: Zero-dependency deployment.
-   **Scraper**: Asynchronous data collection with built-in rate limiting and error handling (`src/summit_housing/scraper_v2.py`).
-   **Maintainability**: Comprehensive `Makefile`, `pytest` suite, and type hinting.

## 🛠️ Quick Start

### Option A: Docker (Recommended)
You don't need Python installed. Just Docker.

```bash
docker-compose up --build
```
*App will be available at http://localhost:8501*

### Option B: Local Development
Requires Python 3.9+

```bash
# 1. Setup Virtual Environment & Dependencies
make setup

# 2. Run ETL Pipeline (Builds/Resets DB)
make ingest

# 3. Launch Dashboard
make run

# (Optional) Run Scraper to get fresh data
# make scrape
```

## 📂 Project Structure

```text
├── data/                  # Raw CSVs, SQLite DB, and quarantined records
├── models/                # Saved PyTorch (.pth) and Scikit-Learn pipelines
├── src/
│   └── summit_housing/
│       ├── dashboard/     # Streamlit application
│       ├── ingestion.py   # ETL pipeline (CSV -> SQLite)
│       ├── ml.py          # Training and Inference logic
│       ├── models.py      # PyTorch Model Definitions
│       ├── queries.py     # SQL Business Logic
│       └── scraper_v2.py  # Data collector
├── tests/                 # Pytest suite
├── Dockerfile             # Container definition
└── Makefile               # Task runner
```

## 📊 Dashboard Modules

The dashboard is structured into narrative tabs:
1.  **Data & Limits**: Dataset overview, bias analysis (survivorship bias), and quality checks.
2.  **Market Context**: Macro-trends (Mortgage Rates vs. Price), buyer demographics, and supply vs. demand.
3.  **Drivers (Feature Importance)**: SHAP analysis and correlation matrices showing what drives value.
4.  **Price Predictor**: A "What-If" simulator letting users adjust property attributes to see predicted prices in real-time.
