.PHONY: install train run-producer run-consumer test lint clean

PYTHON := python3
PIP := pip3
SRC_DIR := src
TEST_DIR := tests

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	mkdir -p models data/output data/checkpoints

train:
	$(PYTHON) -m src.detector.trainer

run-producer:
	$(PYTHON) -m src.producer.taxi_producer

run-consumer:
	$(PYTHON) -m src.consumer.spark_consumer

test:
	pytest $(TEST_DIR)/ -v --cov=$(SRC_DIR) --cov-report=term-missing

lint:
	ruff check $(SRC_DIR)/ $(TEST_DIR)/
	mypy $(SRC_DIR)/ --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	rm -rf data/output/ data/checkpoints/
