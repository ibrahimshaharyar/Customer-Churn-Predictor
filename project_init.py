import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_files = [
    # CI/CD
    ".github/workflows/ci.yml",

    # Configs
    "configs/config.yaml",
    "configs/params.yaml",

    # Data
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/features/.gitkeep",

    # Notebooks
    "notebooks/data_exploration.ipynb",
    "notebooks/feature_engineering.ipynb",
    "notebooks/model_experiments.ipynb",

    # Artifacts
    "artifacts/models/.gitkeep",
    "artifacts/encoders/.gitkeep",
    "artifacts/metrics/.gitkeep",

    # Source code
    "src/data_ingestion/__init__.py",
    "src/data_validation/__init__.py",
    "src/data_preprocessing/__init__.py",
    "src/feature_engineering/__init__.py",
    "src/models/__init__.py",
    "src/serving/__init__.py",
    "src/utils/__init__.py",

    # API
    "app/main.py",

    # Root files
    "requirements.txt",
    "Dockerfile",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    if not filedir.exists():
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

    if not filepath.exists():
        filepath.touch()
        logging.info(f"Created file: {filepath}")
    else:
        logging.info(f"Already exists: {filepath}")
