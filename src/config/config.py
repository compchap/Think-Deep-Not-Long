# config.py
import os
from dotenv import load_dotenv

# Load the .env file once when this module is imported
load_dotenv()

# Export specific variables as Python constants
HF_TOKEN = os.getenv("HF_TOKEN")
DEBUG_MODE = os.getenv("DEBUG", "False") == "True"
