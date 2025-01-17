# config.py
from dotenv import load_dotenv
import os
from pathlib import Path
import logging
import sys

# Configure logger
logger = logging.getLogger(__name__)

# Set up base directory and env path
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / '.env'
load_dotenv(env_path)

class Config:
    def __init__(self):
        """Initialize configuration settings."""
        logger.info("Initializing application configuration")
        
        # OpenAI Configuration
        self.openai = {
            'api_key': self._get_env('OPENAI_API_KEY'),
            'base_url': self._get_env('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            'model': self._get_env('OPENAI_MODEL', 'gpt-4o')
        }
        
        # Google Configuration
        self.google = {
            'api_key': os.getenv('GOOGLE_API_KEY'),
            'model': os.getenv('GOOGLE_MODEL', 'gemini-1.5-flash')
        }

        self.llm_type=os.getenv('LLM_TYPE')
        self.embedding_type=os.getenv('EMBEDDING_TYPE')
        
        # Pinecone Configuration
        self.pinecone = {
            'api_key': self._get_env('PINECONE_API_KEY'),
            'environment': self._get_env('PINECONE_ENVIRONMENT'),
            'index_name': self._get_env('PINECONE_INDEX_NAME', 'saanvi-production')
        }
        
        # Server Configuration
        self.port = int(self._get_env('PORT', '8000'))
        self.max_file_size = int(self._get_env('MAX_FILE_SIZE', '104857600'))  # 100MB default
        
        # Rate Limiting
        self.rate_limit_window = int(self._get_env('RATE_LIMIT_WINDOW', '60'))
        self.rate_limit_max_requests = int(self._get_env('RATE_LIMIT_MAX_REQUESTS', '3'))
        
        # CORS Configuration
        self.cors_origins = self._get_env('CORS_ORIGINS', '*').split(',')
        
        # File Upload Configuration
        self.upload_folder = self._get_env('UPLOAD_FOLDER', '/tmp/uploads')
        self.allowed_extensions = {'txt', 'pdf', 'doc', 'docx', 'csv'}
        
        # Environment
        self.flask_env = self._get_env('FLASK_ENV', 'production')
        self.debug = self.flask_env == 'development'
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("Configuration initialization completed")

    def _get_env(self, key: str, default: str = None) -> str:
        """
        Get environment variable with logging.
        
        Args:
            key: Environment variable key
            default: Default value if key not found
            
        Returns:
            str: Environment variable value or default
        """
        value = os.getenv(key, default)
        if value is None:
            logger.warning(f"Environment variable {key} not set")
            if key in ['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'PINECONE_API_KEY', 'PINECONE_ENVIRONMENT']:
                logger.error(f"{key} is required!")
                sys.exit(1)
        return value

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        try:
            # Create uploads directory
            Path(self.upload_folder).mkdir(parents=True, exist_ok=True)
            logger.debug("Required directories created successfully")
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}", exc_info=True)
            raise

    def validate(self):
        """Validate the configuration settings."""
        required_settings = {
            'OPENAI_API_KEY': self.openai['api_key'],
            'GOOGLE_API_KEY': self.google['api_key'],
            'PINECONE_API_KEY': self.pinecone['api_key'],
            'PINECONE_ENVIRONMENT': self.pinecone['environment']
        }
        
        missing_settings = [key for key, value in required_settings.items() if not value]
        if missing_settings:
            error_msg = f"Missing required configuration: {', '.join(missing_settings)}"
            logger.error(error_msg)
            sys.exit(1)
        
        logger.info("Configuration validation successful")