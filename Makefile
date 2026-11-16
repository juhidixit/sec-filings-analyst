PY ?= python

.PHONY: install data build test lint eval api app docker

install:            ## install package with app, LLM and dev extras
	$(PY) -m pip install -e ".[app,llm,dev]"

data:               ## download the 9 corpus filings from SEC EDGAR (needs SEC_USER_AGENT)
	filings-analyst fetch-manifest

build:              ## parse data/raw/*.html into data/processed/
	filings-analyst build

test:
	$(PY) -m pytest --cov=filings_analyst --cov-report=term-missing

lint:
	ruff check src app eval tests
	ruff format --check src app eval tests
	mypy src

eval:               ## reproduce eval/results/results.md
	$(PY) eval/run_eval.py

api:
	uvicorn filings_analyst.api:app --reload --port 8000

app:
	streamlit run app/streamlit_app.py

docker:
	docker compose up --build
