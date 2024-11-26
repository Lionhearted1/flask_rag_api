from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pymongo import MongoClient
from config import Config

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="memory://"
)

def create_app(config_class=Config):
    app = Flask(__name__)
    config = config_class()  # Instantiate Config class
    app.config.from_object(config)
    
    # Initialize extensions
    limiter.init_app(app)
    
    # Initialize MongoDB
    app.mongodb_client = MongoClient(config.mongodb_uri)
    app.db = app.mongodb_client.get_default_database()
    
    # Store config instance
    app.config_instance = config
    
    # Register blueprints
    from app.routes import api
    app.register_blueprint(api.bp)
    
    return app