.PHONY: data test p1 p2 p3 all notebooks
data:
	python -m common.synthetic data/synthetic
test:
	python -m pytest tests -q
p1:
	python -m project1_matching.src.pipeline configs/project1.yaml
p2:
	python -m project2_feedback_nlp.src.pipeline configs/project2.yaml
p3:
	python -m project3_allocation.src.pipeline configs/project3.yaml
all: data test p1 p2 p3
notebooks:
	python scripts/build_notebooks.py --execute
