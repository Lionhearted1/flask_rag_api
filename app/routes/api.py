# app/routes/api.py
from flask import Blueprint, request, jsonify, current_app
from app.services.document_processor import DocumentProcessor
from app.services.chat_service import ChatService
from app.utils.errors import AppError
from werkzeug.utils import secure_filename
import logging

# Configure logger
logger = logging.getLogger(__name__)

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.errorhandler(AppError)
def handle_error(error):
    """Handle application-specific errors."""
    logger.error(f"API Error: {error}")
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@bp.route('/documents', methods=['POST'])
def upload_document():
    """Upload and process a new document."""
    logger.info("Received document upload request")
    
    if 'file' not in request.files:
        logger.warning("No file provided in request")
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if not file.filename:
        logger.warning("No filename in request")
        return jsonify({'error': 'No file selected'}), 400
        
    if file.content_length and file.content_length > current_app.config_instance.max_file_size:
        logger.warning(f"File too large: {file.content_length} bytes")
        return jsonify({'error': 'File too large'}), 413
        
    try:
        processor = DocumentProcessor()
        result = processor.process_file(file)
        logger.info(f"Successfully processed document: {file.filename}")
        return jsonify(result), 201
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/documents', methods=['GET'])
def list_documents():
    """List all documents in the Pinecone index."""
    try:
        processor = DocumentProcessor()
        documents = processor.get_all_documents()
        
        # Process the list of documents
        formatted_documents = []
        for doc in documents:
            formatted_documents.append({
                'id': doc['id'],
                'metadata': doc['metadata'],
                'score': doc['score']
            })
        
        return jsonify(formatted_documents), 200
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route('/documents/<id>', methods=['DELETE'])
def delete_document(id):
    """Delete a specific document."""
    logger.info(f"Received request to delete document: {id}")
    try:
        processor = DocumentProcessor()
        processor.delete_document(id)
        logger.info(f"Successfully deleted document: {id}")
        return '', 204
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/documents/<id>', methods=['PUT'])
def update_document(id):
    """Update document metadata."""
    logger.info(f"Received request to update document: {id}")
    try:
        data = request.get_json()
        if not data:
            logger.warning("No data provided in update request")
            return jsonify({'error': 'No data provided'}), 400
            
        processor = DocumentProcessor()
        updated_metadata = processor.update_document(id, data)
        
        logger.info(f"Successfully updated document: {id}")
        return jsonify({
            'id': id,
            'metadata': updated_metadata
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating document: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@bp.route('/chat', methods=['POST'])
def chat():
    """Process a chat message."""
    logger.info("Received chat request")
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning("No message provided in chat request")
            return jsonify({'error': 'Message is required'}), 400
            
        chat_service = ChatService(current_app.config_instance)
        response = chat_service.chat(data['message'])
        
        logger.info("Successfully processed chat message")
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500