# Contributing to Summit Housing Analytics

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### 1. Fork and Clone
```bash
git clone https://github.com/YOUR_USERNAME/summit-housing.git
cd summit-housing
```

### 2. Set Up Development Environment
```bash
make setup
```

This will:
- Create a virtual environment in `.venv/`
- Install all dependencies including dev tools (pytest, ruff, black)

### 3. Verify Installation
```bash
# Run tests
make test

# Check code quality
make lint
```

## Project Structure

```
summit-housing/
├── src/summit_housing/       # Main application code
│   ├── dashboard/            # Streamlit UI components (Deprecated)
│   ├── ml/                   # Machine learning models
│   ├── queries.py            # SQL analytics layer
│   ├── ingestion.py          # ETL pipeline
│   └── scraper_v2.py         # Data collection
├── data/                     # Raw data and SQLite DB
├── models/                   # Trained model artifacts
├── config/                   # YAML configuration files
├── scripts/                  # Utility scripts
└── tests/                    # Test suite
```

## Common Development Tasks

### Running the Dashboard Locally
```bash
make serve-static
```

### Rebuilding the Database
```bash
make ingest
```

### Training Models
```bash
# Train a single model
make train-gbm
make train-nn

# Run full tournament (parameter sweep)
make tournament
```

### Running Tests
```bash
make test
```

### Code Quality
We use `ruff` for linting and `black` for formatting:
```bash
make lint
```

## Making Changes

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
make test
make lint
```

### 4. Commit and Push
```bash
git add .
git commit -m "Description of your changes"
git push origin feature/your-feature-name
```

### 5. Create a Pull Request
Open a PR on GitHub with a clear description of your changes.

## Code Style Guidelines

- **Python**: Follow PEP 8 (enforced by ruff)
- **Line Length**: 88 characters (black default)
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Encouraged for new code

## Configuration

All ML configuration is centralized in `config/ml_config.yaml`:
- Feature definitions
- Model hyperparameters
- Training settings
- Monotonicity constraints

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Documentation improvements

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
