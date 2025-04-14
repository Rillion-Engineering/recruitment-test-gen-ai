.PHONY: install run ingestion_pipeline

install:
	uv install

run:
	PYTHONPATH=$(pwd)/app python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8003

ingestion_pipeline:
	python ingestion_pipeline.py