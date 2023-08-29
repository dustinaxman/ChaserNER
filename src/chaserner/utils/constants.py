from pathlib import Path


(Path(__file__).parents[3]/'logs').mkdir(parents=True, exist_ok=True)
LOGGING_DIR_PATH = Path(__file__).parents[3]/'logs'
