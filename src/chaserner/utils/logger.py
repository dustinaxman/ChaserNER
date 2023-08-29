import logging
import datetime
from chaserner.utils.constants import LOGGING_DIR_PATH

# Create a custom logger
logger = logging.getLogger(__name__)

# Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

now = datetime.datetime.now()
logfile_name = now.strftime("%Y-%m-%d_%H-%M-%S.log")
logging_file_path = LOGGING_DIR_PATH/logfile_name

# Create a file handler to log messages to a file
file_handler = logging.FileHandler(logging_file_path)
file_handler.setLevel(logging.DEBUG)

# Create a console handler to print messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)