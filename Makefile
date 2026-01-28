.PHONY: setup test ingest run clean lint

VENV = .venv/bin

setup:
	$(VENV)/pip install -e .[dev]

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

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
