.PHONY: setup test ingest run clean lint export serve-static

VENV = .venv/bin
PYTHON = python3

setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv .venv
	@echo "Installing dependencies..."
	$(VENV)/pip install --upgrade pip
	$(VENV)/pip install -e .[dev]
	@echo "✅ Setup complete! Run 'make run' to start the dashboard."

test:
	$(VENV)/pytest -v

lint:
	$(VENV)/ruff check .

ingest:
	$(VENV)/python -m summit_housing.ingestion

run:
	$(VENV)/streamlit run src/summit_housing/dashboard/Introduction.py

view-static:
	@echo "Serving static dashboard at http://localhost:8000"
	@cd static_dashboard && python3 -m http.server 8000

scrape:
	$(VENV)/python src/summit_housing/scraper_v2.py --workers 10

train-gbm:
	$(VENV)/python src/summit_housing/ml/pipeline.py train gbm

train-nn:
	$(VENV)/python src/summit_housing/ml/pipeline.py train nn

tournament:
	$(VENV)/python src/summit_housing/ml/tournament.py --runs 5

export:
	@echo "📊 Exporting data to static site..."
	$(VENV)/python scripts/export_extra_data.py
	@echo "✅ Export complete!"
	@echo ""
	@echo "To view the static site:"
	@echo "  make serve-static"

serve-static:
	@echo "🌐 Starting static site server..."
	@echo "Finding portfolio directory..."
	@if [ -d "../brian.fishman.info/public/projects/summit" ]; then \
		cd ../brian.fishman.info/public/projects/summit && \
		echo "📍 Serving from: $$(pwd)" && \
		echo "🚀 Open http://localhost:8000 in your browser" && \
		echo "" && \
		$(PYTHON) -m http.server 8000; \
	else \
		echo "❌ Portfolio directory not found at ../brian.fishman.info/public/projects/summit"; \
		echo "   Make sure repos are cloned side-by-side, or set SUMMIT_EXPORT_PATH"; \
	fi

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
