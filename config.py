from dotenv import load_dotenv
import os
from pathlib import Path
import logging

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
            'model': self._get_env('OPENAI_MODEL', 'gpt-3.5-turbo')
        }
        logger.debug("OpenAI configuration loaded")

        # ChromaDB Configuration
        self.chroma_settings = {
            'persist_directory': self._get_env('CHROMA_PERSIST_DIRECTORY', 
                                             str(BASE_DIR / 'chroma_db')),
            'anonymized_telemetry': False
        }
        logger.debug(f"ChromaDB persist directory set to: {self.chroma_settings['persist_directory']}")

        # Server Configuration
        self.port = int(self._get_env('PORT', '3000'))
        self.max_file_size = int(self._get_env('MAX_FILE_SIZE', '104857600'))  # 100MB default
        
        # Rate Limiting
        self.rate_limit_window = int(self._get_env('RATE_LIMIT_WINDOW', '60'))
        self.rate_limit_max_requests = int(self._get_env('RATE_LIMIT_MAX_REQUESTS', '3'))
        
        # File Upload Configuration
        self.upload_folder = self._get_env('UPLOAD_FOLDER', str(BASE_DIR / 'uploads'))
        self.allowed_extensions = {
            'txt', 'pdf', 'doc', 'docx', 'csv'
        }
        
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
            if key == 'OPENAI_API_KEY':
                logger.error("OpenAI API key is required!")
                raise ValueError("OPENAI_API_KEY environment variable is required")
        return value

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        try:
            # Create ChromaDB directory
            Path(self.chroma_settings['persist_directory']).mkdir(parents=True, exist_ok=True)
            
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
            'CHROMA_PERSIST_DIRECTORY': self.chroma_settings['persist_directory']
        }
        
        missing_settings = [key for key, value in required_settings.items() if not value]
        
        if missing_settings:
            error_msg = f"Missing required configuration: {', '.join(missing_settings)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation successful")