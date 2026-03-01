"""
src/utils/logger.py
-------------------
Shared logging configuration for all pipeline scripts.
"""

# -----
# Libraries
import logging
from datetime import datetime
from pathlib import Path

# -----
# Constants
LOG_DIR = "artifacts/logs"


# -----
# Logger setup function
def setup_logger(script_name: str, log_dir: str = LOG_DIR) -> logging.Logger:
    """
    Configure and return a logger that writes to both a log file and the console.

    Parameters:
    ---
    script_name: str
        Name used for the log filename (e.g. 'prep', 'train').
    log_dir: str
        Directory where log files will be saved.

    Returns:
    ---
    logging.Logger
        Configured logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{log_dir}/{script_name}_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(script_name)
