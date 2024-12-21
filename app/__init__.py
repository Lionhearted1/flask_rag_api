# app/__init__.py
import logging
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import Config
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
import os

# Configure logger
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://"
)

def create_app(config_class=Config):
    """
    Create and configure the Flask application.
    """
    logger.info("Creating Flask application")
    
    # Initialize Flask app
    app = Flask(__name__)
    
    try:
        # Load configuration
        logger.debug("Loading configuration")
        config = config_class()
        app.config.from_object(config)
        
        # Store config instance
        app.config_instance = config
        
        # Initialize rate limiter
        logger.debug("Initializing rate limiter")
        limiter.init_app(app)
        
        # Initialize HuggingFace embeddings
        logger.debug("Initializing HuggingFace embeddings")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB
        logger.debug("Initializing ChromaDB")
        persist_dir = config.chroma_settings['persist_directory']
        
        # Ensure the persist directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        # Create settings for both client and vectorstore
        chroma_settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True
        )
        
        try:
            # First try to create a new persistent client
            app.chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=chroma_settings
            )
        except ValueError as e:
            if "already exists" in str(e):
                logger.info("Connecting to existing ChromaDB instance")
                # If database exists, use the existing settings
                app.chroma_client = chromadb.PersistentClient(
                    path=persist_dir
                )
            else:
                raise
        
        # Create or get collection
        app.chroma_collection = app.chroma_client.get_or_create_collection("documents")
        
        # Initialize vectorstore with the existing client
        app.vectorstore = Chroma(
            client=app.chroma_client,
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
        # Register blueprints
        logger.debug("Registering blueprints")
        from app.routes import api
        app.register_blueprint(api.bp)
        
        # Register error handlers
        @app.errorhandler(500)
        def handle_500(error):
            logger.error(f"Internal Server Error: {error}", exc_info=True)
            return {"error": "Internal Server Error"}, 500
        
        @app.teardown_appcontext
        def cleanup(exc):
            """Clean up resources on app context teardown."""
            logger.debug("Cleaning up application resources")
            if hasattr(app, 'vectorstore'):
                app.vectorstore.persist()
        
        logger.info("Flask application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Error creating Flask application: {str(e)}", exc_info=True)
        raise

def init_logging():
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set log levels for some verbose libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('flask').setLevel(logging.INFO)