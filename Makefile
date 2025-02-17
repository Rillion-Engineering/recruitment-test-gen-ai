.PHONY: install run parse_documents

install:
	uv install

run:
	PYTHONPATH=$(pwd)/app python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8003

parse_documents:
	python parse_documents.py