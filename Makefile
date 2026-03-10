.PHONY: test lint typecheck run-all clean

test:
	python -m pytest tests/ -v

lint:
	python -m flake8 src/ experiments/ tests/ --max-line-length=100 --ignore=E402

typecheck:
	python -m mypy src/ --ignore-missing-imports

run-all:
	@echo "=== Experiment 0: Baseline ==="
	python experiments/exp0_baseline.py
	@echo ""
	@echo "=== Experiment 1: Features ==="
	python experiments/exp1_features.py
	@echo ""
	@echo "=== Experiment 2: Clustering ==="
	python experiments/exp2_clustering.py
	@echo ""
	@echo "=== Experiment 3: Supervised ==="
	python experiments/exp3_supervised.py
	@echo ""
	@echo "=== Experiment 4: Embeddings ==="
	python experiments/exp4_embeddings.py
	@echo ""
	@echo "=== Experiment 7: Four-way ==="
	python experiments/exp7_fourway.py
	@echo ""
	@echo "=== Experiment 8: Jataka ==="
	python experiments/exp8_jataka.py
	@echo ""
	@echo "=== Misclassification Analysis ==="
	python experiments/exp_misclassification_analysis.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .pytest_cache
