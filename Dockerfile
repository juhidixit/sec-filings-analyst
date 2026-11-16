FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app,llm]"

COPY app ./app
COPY data/processed ./data/processed
COPY .streamlit ./.streamlit

RUN useradd --create-home analyst && chown -R analyst /app
USER analyst

ENV FILINGS_CORPUS=/app/data/processed
EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "filings_analyst.api:app", "--host", "0.0.0.0", "--port", "8000"]
