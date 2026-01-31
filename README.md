# Summit County Housing: Analytics & ML Portfolio
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)

![SQL](https://img.shields.io/badge/SQL-Advanced-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-green)

A production-grade analytics platform that combines **Advanced SQL Engineering** with **Deep Learning** to analyze and predict real estate trends in Summit County, CO.

This project demonstrates the full Data Science lifecycle:
1.  **Ingestion**: Scraping and validating property records (~40k rows).
2.  **Engineering**: Building a normalized SQLite warehouse with complex window functions.
3.  **Modeling**: Unified GBM and Neural Network training with temporal cross-validation.
4.  **MLOps**: Automated "Champion/Challenger" registry and centralized YAML-driven configuration.
5.  **Productization**: Uncertainty estimation (Quantile Regression) and luxury segment calibration.
6.  **Deployment**: Serving insights via a static web dashboard.

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

### Prerequisites
- **Python 3.9+** (check with `python3 --version`)
- **Git** (to clone the repository)
- **Docker** (optional, for containerized deployment)

### Option A: One-Command Start (Easiest) ⭐
Perfect for first-time users. This script handles everything automatically.

```bash
git clone <your-repo-url>
cd summit-housing
./quickstart.sh
```

The script will:
1. Create a virtual environment
2. Install all dependencies
3. Build the database (if needed)
4. Launch the static dashboard at http://localhost:8000

### Option B: Docker (Zero Python Setup)
You don't need Python installed. Just Docker.

```bash
docker-compose up --build
```
*App will be available at http://localhost:8000*

### Option C: Manual Setup (For Developers)
For those who want more control over the process.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd summit-housing

# 2. Setup Virtual Environment & Dependencies
make setup

# 3. Build the Database (First time only)
make ingest

# 4. (Optional) Train Models - Pre-trained models are included
make tournament  # Full parameter sweep
# OR
make train-gbm   # Train just GBM
make train-nn    # Train just Neural Network

# 5. Launch Dashboard
make serve-static
```

**Dashboard will be available at:** http://localhost:8000

---

## 🌐 Static Site Export

This project exports data to a **static web dashboard** for public viewing.

**Live Dashboard:** [brian.fishman.info/projects/summit](https://brian.fishman.info/projects/summit/)


### Exporting Data

The backend generates JSON files that power the static dashboard:

```bash
# Export all data (analytics, ML metrics, PDP, SHAP, etc.)
make export
```

**Export Destination:**
1. Checks `SUMMIT_EXPORT_PATH` environment variable
2. Tries `../brian.fishman.info/public/projects/summit/data/` (side-by-side repos)
3. Falls back to hardcoded path

### Custom Export Path

```bash
export SUMMIT_EXPORT_PATH="/custom/path/to/data"
make export
```

### Viewing the Static Site Locally

```bash
# Serve the static dashboard
make serve-static
# Opens at http://localhost:8000
```

### Architecture & Deployment
The system is designed to be self-contained while supporting external portfolio integration.

**1. Internal Viewer (Self-Contained)**
The `static_dashboard/` directory contains a full copy of the frontend.
```bash
make export        # Populates static_dashboard/data
make serve-static  # Launches local viewer
```

**2. External Portfolio Integration**
If the [brian.fishman.info](https://github.com/fishish1/brian.fishman.info) repo is cloned side-by-side, `make export` **automatically syncs** the data there as well.

```
summit-data-science/          ← Backend Repo
├── static_dashboard/         ← Internal Viewer (Auto-updated)
└── scripts/export...         ← Syncs to sibling repo if present
```

**Benefits:**
- **Zero Dependencies:** Reviewers can run the full dashboard without cloning the portfolio.
- **Portfolio Sync:** updates the live site automatically when present.


---

## 📸 Project Screenshots

The project includes an automated system to capture high-quality desktop and mobile screenshots for portfolio use.

- **Command**: `make screenshots`
- **Output**: `screenshots/` directory
- **Details**: See [SCREENSHOT_SYSTEM.md](SCREENSHOT_SYSTEM.md) for full documentation and integration guides.

---

## 📂 Project Structure

```text
├── data/                  # Raw CSVs, SQLite DB, and quarantined records
├── models/                # Saved PyTorch (.pth) and Scikit-Learn pipelines
├── notebooks/             # Jupyter notebooks for exploratory analysis
├── scripts/               # Utility and debugging scripts
├── src/
│   └── summit_housing/
│       ├── dashboard/     # Streamlit application (Deprecated)
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
