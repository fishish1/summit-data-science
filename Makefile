.PHONY: setup test ingest run clean lint export serve-static screenshots

VENV = .venv/bin
PYTHON = python3

setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv .venv
	@echo "Installing dependencies..."
	$(VENV)/pip install --upgrade pip
	$(VENV)/pip install -e .[dev]
	@echo "✅ Setup complete! Run './quickstart.sh' to start the dashboard."

test:
	$(VENV)/pytest -v

lint:
	$(VENV)/ruff check .

ingest:
	$(VENV)/python -m summit_housing.ingestion

run:
	$(VENV)/streamlit run src/summit_housing/dashboard/Introduction.py


scrape:
	$(VENV)/python src/summit_housing/scraper_v2.py --workers 10

train-gbm:
	$(VENV)/python src/summit_housing/ml/pipeline.py train gbm

train-nn:
	$(VENV)/python src/summit_housing/ml/pipeline.py train nn

tournament:
	$(VENV)/python src/summit_housing/ml/tournament.py --runs 5

export:
	@echo "📊 Exporting data and models to static site..."
	$(VENV)/python scripts/prepare_mortgage_data.py
	$(VENV)/python scripts/export_extra_data.py
	$(VENV)/python scripts/export_to_onnx.py
	@echo "✅ All exports complete!"
	@echo ""
	@echo "To view the static site:"
	@echo "  make serve-static"

serve-static:
	@echo "Serving static dashboard at http://localhost:8000"
	@if [ -d "static_dashboard" ]; then \
		cd static_dashboard && python3 -m http.server 8000; \
	else \
		echo "❌ 'static_dashboard' directory not found. Run 'make export' first."; \
	fi

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf src/__pycache__
	rm -rf tests/__pycache__

screenshots:
	@echo "📸 Capturing portfolio screenshots..."
	@./scripts/capture_portfolio_screenshots.sh

