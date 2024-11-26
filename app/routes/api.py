# app/routes/api.py
from flask import Blueprint, request, jsonify, current_app
from app.services.document_processor import DocumentProcessor
from app.services.chat_service import ChatService
from app.utils.errors import AppError
from werkzeug.utils import secure_filename
from bson import ObjectId

bp = Blueprint('api', __name__, url_prefix='/api')

@bp.errorhandler(AppError)
def handle_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@bp.route('/documents', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    if file.content_length and file.content_length > current_app.config.max_file_size:
        return jsonify({'error': 'File too large'}), 413
    
    processor = DocumentProcessor(current_app.db)
    result = processor.process_file(file)
    return jsonify(result), 201

@bp.route('/documents', methods=['GET'])
def list_documents():
    documents = current_app.db.documents.find({}, {'embedding': 0})
    return jsonify([{**doc, '_id': str(doc['_id'])} for doc in documents])

@bp.route('/documents/<id>', methods=['DELETE'])
def delete_document(id):
    result = current_app.db.documents.delete_one({'_id': ObjectId(id)})
    if result.deleted_count == 0:
        return jsonify({'error': 'Document not found'}), 404
    return '', 204

@bp.route('/documents/<id>', methods=['PUT'])
def update_document(id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        result = current_app.db.documents.update_one(
            {'_id': ObjectId(id)},
            {'$set': {'metadata.fileName': data.get('fileName')}}
        )
        
        if result.matched_count == 0:
            return jsonify({'error': 'Document not found'}), 404
            
        return jsonify({'message': 'Document updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Message is required'}), 400
    
    chat_service = ChatService(current_app.db, current_app.config_instance)
    response = chat_service.chat(data['message'])
    return jsonify({'response': response})