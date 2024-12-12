.PHONY: install run parse_documents

install:
	poetry install --no-cache

run:
	python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

parse_documents:
	python parse_documents.py