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
3.  **Modeling**: Unified GBM and Neural Network training with temporal cross-validation.
4.  **MLOps**: Automated "Champion/Challenger" registry and centralized YAML-driven configuration.
5.  **Productization**: Uncertainty estimation (Quantile Regression) and luxury segment calibration.
6.  **Deployment**: Serving insights via an interactive Streamlit dashboard.

## 🚀 Key Technical Features

### 1. Advanced Machine Learning & Interpretability
Beyond simple regression, this project leverages modern ML techniques to understand market drivers:
-   **Neural Network (PyTorch)**: Custom feed-forward architecture for high-accuracy price prediction (`src/summit_housing/models.py`).
-   **Explainable AI (XAI)**:
    -   **SHAP Values**: Deconstructs individual predictions (e.g., "This home is $50k cheaper because of its distance to Breckenridge").
    -   **Partial Dependence Plots (PDP)**: Visualizes non-linear relationships between features for global model understanding.
-   **MLOps Maturity**:
    -   **Automated Tournament**: Run `make tournament` to clear old artifacts, perform a 5-run parameter sweep, and promote the best models automatically.
    -   **Centralized Configuration**: All features and hyperparameters (including monotonicity constraints) managed via [ml_config.yaml](file:///Users/brian/Documents/summit/config/ml_config.yaml).
    -   **Champion Registry**: The system tracks the "current best" in `models/champion_registry.json`. New models are only promoted if they reduce the Mean Absolute Error (MAE) compared to the current champion.
-   **Productization**:
    -   **Confidence Intervals**: Predicts a "Likely Range" (10th-90th percentile) using Quantile Regression.
    -   **Luxury Calibration**: Focused sample weighting to reduce high-variance errors in the \$2M+ segment.
-   **Validation Rigor**: Implements **Walk-Forward Cross-Validation** to prevent temporal leakage. You can run a standalone evaluation without promoting a model using:
    ```bash
    python src/summit_housing/ml/pipeline.py eval gbm  # or nn
    ```

### 2. Advanced SQL Analytics
Logic is pushed to the database layer to ensure performance and demonstrate SQL mastery:
-   **Window Functions**: Uses `LAG()` and `LEAD()` to calculate appreciation rates and holding periods.
-   **Trend Smoothing**: Uses `AVG() OVER (ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)` for moving averages.
-   **Complex Unpivoting**: Transforms denormalized raw CSV logs into a clean transaction stream using CTEs and `UNION ALL`.

### 3. Resilient Data Pipeline (The "DLQ" Pattern)
Real-world data is messy. The pipeline (`src/summit_housing/ingestion.py`) implements a **Dead Letter Queue** pattern:
-   **Validation**: Every record is checked against strict **Pydantic** models.
-   **Fault Tolerance**: The pipeline fails gracefully. Malformed records are quarantined in `data/rejected_records.csv`.
-   **Stateful Scraping**: The scraper (`scraper_v2.py`) is resumable. It automatically detects existing records in the output and skips them, allowing for graceful recovery from network failures.

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

# 3. Trigger Training (Optional - Current Champions are provided)
make tournament # Purge and sweep for best parameters (Recommended)
make train-gbm
make train-nn

# 4. Launch Dashboard
make run
```

## 📂 Project Structure

```text
├── data/                  # Raw CSVs, SQLite DB, and quarantined records
├── models/                # Saved PyTorch (.pth) and Scikit-Learn pipelines
├── notebooks/             # Jupyter notebooks for exploratory analysis
├── scripts/               # Utility and debugging scripts
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

## 🛠️ Developer Reference

### Scraper CLI Arguments
While `make scrape` uses defaults, you can customize the collection:
```bash
python src/summit_housing/scraper_v2.py --workers 20 --input data/custom_ids.csv --output data/new_collection.csv
```
*Note: Progress is automatically tracked via `data/metadata.json`.*

### ML Configuration Schema
The `config/ml_config.yaml` controls the engine. Key sections include:
- **Monotonicity**: Set to `1` (positive), `-1` (negative), or `0` (none). This forces the Neural Network to respect physical laws (e.g., more square footage cannot decrease the price).
- **Temporal Split**: `test_months` defines the "hold-out" period for the walk-forward evaluation.

### Model Registry
The "Champion" system sources its live models from `models/champion_registry.json`. To manually override a champion, update this file with the desired `run_id`.
