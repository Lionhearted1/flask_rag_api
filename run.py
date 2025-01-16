# run.py
from app import create_app, init_logging
from flask_cors import CORS
from config import Config
import logging

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)

# Create Flask application
app = create_app()

# Initialize CORS
config = Config()
CORS(app, resources={
    r"/api/*": {
        "origins": config.cors_origins,
        "methods": ["GET", "POST", "OPTIONS", "DELETE", "PUT"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

if __name__ == '__main__':
    # Production-ready server configuration
    app.run(
        host='0.0.0.0',
        port=config.port,
        debug=config.debug
    )