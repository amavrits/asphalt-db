from dotenv import load_dotenv
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).parent
dotenv_path = SCRIPT_DIR.parent / ".env"
load_dotenv(dotenv_path)

DB_CONFIG = {
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}
