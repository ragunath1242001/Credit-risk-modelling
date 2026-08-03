install:
	python -m pip install .
test:
	python -m pytest -q
train:
	python scripts/train_pd.py
reports:
	python scripts/generate_reports.py
