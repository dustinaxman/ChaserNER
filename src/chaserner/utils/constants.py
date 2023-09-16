from pathlib import Path
#LOGGING_DIR_PATH = Path(__file__).parents[3]/'logs'

LOGGING_DIR_PATH = Path().home()/'logs'
LOGGING_DIR_PATH.mkdir(parents=True, exist_ok=True)