import logging
import os
from datetime import datetime
import os

# Log file name
LOG_FILE_NAME = f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.log"

#Log Directory
LOG_FILE_DIR = os.path.join(os.getcwd(), "logs")

#Create folder if not available
os.makedirs(LOG_FILE_DIR, exist_ok=True)

#log file path
LOG_FILE_PATH = os.path.join(LOG_FILE_NAME, LOG_FILE_DIR)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(levelname)s - %(message)s",
    level = logging.INFO
)