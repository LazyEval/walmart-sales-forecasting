from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.resolve()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
MODELS_DIR = Path(BASE_DIR, "models")
LOGS_DIR = Path(BASE_DIR, "logs")

# Data
RAW_DATA_DIR = Path(DATA_DIR, "raw")
PROCESSED_DATA_DIR = Path(DATA_DIR, "processed")
TEST_DATA_DIR = Path(DATA_DIR, "test")

# Stores
EXPERIMENTS_STORE = Path(MODELS_DIR, "experiments")
TENSORBOARD_STORE = Path(MODELS_DIR, "tensorboard")

# Constants for default values
SEED = 3
