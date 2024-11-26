# config.py
from dotenv import load_dotenv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / '.env'
load_dotenv(env_path)

class Config:
    def __init__(self):
        self.mongodb_uri = os.getenv('MONGODB_URI')
        print(f"MongoDB URI: {self.mongodb_uri}")
        
        self.openai = {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'base_url': os.getenv('OPENAI_BASE_URL'),
            'model': os.getenv('OPENAI_MODEL')
        }
        
        self.port = int(os.getenv('PORT', 3000))
        self.max_file_size = int(os.getenv('MAX_FILE_SIZE', 104857600))
        self.rate_limit_window = int(os.getenv('RATE_LIMIT_WINDOW', 60))
        self.rate_limit_max_requests = int(os.getenv('RATE_LIMIT_MAX_REQUESTS', 3))