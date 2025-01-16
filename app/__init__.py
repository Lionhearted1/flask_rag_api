import logging
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import Config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from pinecone import Pinecone, ServerlessSpec

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
        
        # Initialize Google's embedding model
        logger.debug("Initializing Google embeddings")
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=config.google['api_key'],
            model="models/embedding-001"  # Google's text embedding model
        )
        
        # Create an instance of Pinecone
        logger.debug("Initializing Pinecone")
        pc = Pinecone(
            api_key=config.pinecone['api_key']
        )
        
        # Get or create Pinecone index
        index_name = config.pinecone['index_name']
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=768,  # Dimension for Google's embedding model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
        
        # Initialize vectorstore
        from langchain_pinecone import PineconeVectorStore
        app.vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            text_key="text"
        )
        
        # Store the index_name in the app context for easy access
        app.pinecone_index_name = index_name
        
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
    logging.getLogger('pinecone').setLevel(logging.WARNING)
    logging.getLogger('flask').setLevel(logging.INFO)
