# Logger is a way to track what's happending in program during execution by recording(logging) messages, errors and other information.
# It's is like keeping a diary of our program execution.

import logging
import os
from datetime import datetime

# Create log-file name(with Timestamp).
# eg: "02_11_2024_15_30_45.log"
# Create log directory path
logs_dir = os.path.join(os.getcwd(), "logs")

# Create the directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)  

# CREATE LOG FILE NAME.
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(logs_dir, LOG_FILE)

# Basic Configuration for logging(recording) information during execution. 
logging.basicConfig(
    # Parameters.
    filename = logs_path, # Where the logs should be saved(messages are written in this file).
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s", # Fomrate fot the log-message.
    level = logging.INFO # SET LOGGING LEVEL TO "INFO".
)


 