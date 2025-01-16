import logging
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.errors import DocumentProcessingError
import tempfile
import os
from flask import current_app
import pinecone
import uuid

# Configure logger
logger = logging.getLogger(__name__)

from flask import current_app

class DocumentProcessor:
    def __init__(self):
        """Initialize DocumentProcessor."""
        logger.info("Initializing DocumentProcessor")
        try:
            logger.debug("Setting up text splitter")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            logger.info("DocumentProcessor initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DocumentProcessor: {str(e)}", exc_info=True)
            raise

    @property
    def vectorstore(self):
        """Get the vectorstore from the current Flask application context."""
        return current_app.vectorstore

    def get_all_documents(self):
        """Get all documents from Pinecone with pagination."""
        try:
            # Access the index_name from the Flask app context
            index_name = current_app.pinecone_index_name
            index = pinecone.Index(index_name)
            
            # Use sparse vectors query to get all documents
            results = []
            batch_size = 1000
            next_page_token = None
            
            while True:
                response = index.query(
                    vector=[0.0] * 768,  # Dimension for Google's embedding model
                    top_k=batch_size,
                    include_metadata=True,
                    page_size=batch_size,
                    next_page_token=next_page_token
                )
                
                if not response.matches:
                    break
                    
                results.extend(response.matches)
                next_page_token = getattr(response, 'next_page_token', None)
                
                if not next_page_token:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}", exc_info=True)
            raise DocumentProcessingError(str(e))

    def process_file(self, file):
        """Process an uploaded file and store its chunks in Pinecone."""
        logger.info(f"Processing file: {file.filename} (type: {file.content_type})")
        temp_dir = None
        temp_path = None
        
        try:
            # Create temporary directory and save file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.filename)
            file.save(temp_path)
            
            # Load and process document
            documents = self._load_document(temp_path, file.content_type)
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata to chunks
            doc_ids = []
            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata.update({
                    'fileName': file.filename,
                    'contentType': file.content_type,
                    'doc_id': doc_id,
                    'text': chunk.page_content  # Required for Pinecone
                })
            
            # Add documents to Pinecone
            logger.debug(f"Adding {len(chunks)} chunks to Pinecone")
            self.vectorstore.add_documents(chunks)
            
            logger.info(f"Successfully processed and stored {len(chunks)} chunks")
            return {
                'fileName': file.filename,
                'chunks': len(chunks),
                'ids': doc_ids
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
            raise DocumentProcessingError(str(e))
            
        finally:
            # Cleanup temporary files
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def delete_document(self, doc_id):
        """Delete a document from Pinecone."""
        try:
            index = pinecone.Index(self.vectorstore.index_name)
            index.delete(ids=[doc_id])
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            raise DocumentProcessingError(str(e))

    def update_document(self, doc_id, metadata):
        """Update document metadata in Pinecone."""
        try:
            index = pinecone.Index(self.vectorstore.index_name)
            # Get existing vector
            vector_data = index.fetch([doc_id])
            
            if not vector_data.vectors:
                raise DocumentProcessingError("Document not found")
            
            existing_vector = vector_data.vectors[doc_id]
            
            # Update metadata while preserving the vector and existing metadata
            updated_metadata = {**existing_vector.metadata, **metadata}
            
            # Ensure text field is preserved
            if 'text' not in updated_metadata and 'text' in existing_vector.metadata:
                updated_metadata['text'] = existing_vector.metadata['text']
            
            index.upsert([(
                doc_id,
                existing_vector.values,
                updated_metadata
            )])
            
            return updated_metadata
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}", exc_info=True)
            raise DocumentProcessingError(str(e))

#_load_document method:
    def _load_document(self, file_path, content_type):
        """Load a document based on its content type."""
        logger.debug(f"Loading document with content type: {content_type}")
        
        try:
            if content_type == 'text/csv':
                logger.debug("Using CSV loader with pandas")
                import pandas as pd
                
                # Read CSV using pandas, letting it auto-detect headers
                try:
                    # First try reading with headers
                    df = pd.read_csv(file_path)
                    has_headers = True
                except pd.errors.EmptyDataError:
                    logger.error("Empty CSV file")
                    raise DocumentProcessingError("CSV file is empty")
                except Exception:
                    # If failed, try reading without headers
                    try:
                        df = pd.read_csv(file_path, header=None)
                        # Generate column names
                        df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]
                        has_headers = False
                    except Exception as e:
                        logger.error(f"Failed to read CSV: {str(e)}")
                        raise
                
                logger.debug(f"CSV columns: {df.columns.tolist()}")
                documents = []
                
                # Process each row into a document
                for idx, row in df.iterrows():
                    # Format each value with proper rounding for floats
                    formatted_data = {}
                    for col, val in row.items():
                        if pd.isna(val):
                            formatted_data[col] = "N/A"
                        elif isinstance(val, float):
                            formatted_data[col] = f"{val:.3f}"
                        else:
                            formatted_data[col] = str(val)
                    
                    # Create a formatted string
                    content_parts = [
                        f"Row {idx + 1}:",
                        "-" * 40  # Separator line
                    ]
                    
                    # Add each field
                    max_col_length = max(len(str(col)) for col in df.columns)
                    for col in df.columns:
                        # Align column names for better readability
                        content_parts.append(f"{str(col):<{max_col_length}}: {formatted_data[col]}")
                    
                    # Join all parts with newlines to create the page content
                    page_content = "\n".join(content_parts)
                    
                    # Create metadata dictionary
                    metadata = {
                        'source': os.path.basename(file_path),
                        'row_index': idx,
                        'has_headers': has_headers
                    }
                    
                    # Add original values to metadata
                    for col, val in row.items():
                        # Convert numpy/pandas types to Python native types
                        if pd.isna(val):
                            metadata[str(col)] = None
                        else:
                            metadata[str(col)] = val.item() if hasattr(val, 'item') else val
                    
                    # Create document
                    from langchain_core.documents import Document
                    doc = Document(
                        page_content=page_content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    logger.debug(f"Created document for row {idx + 1}")
                
                logger.info(f"Processed {len(documents)} rows from CSV")
                return documents
                
            elif content_type == 'application/pdf':
                logger.debug("Using PDF loader")
                loader = PyPDFLoader(file_path)
                return loader.load()
            elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                logger.debug("Using DOCX loader")
                loader = Docx2txtLoader(file_path)
                return loader.load()
            elif content_type == 'text/plain':
                logger.debug("Using text loader")
                loader = TextLoader(file_path)
                return loader.load()
            else:
                logger.error(f"Unsupported file type: {content_type}")
                raise DocumentProcessingError(f"Unsupported file type: {content_type}")
                
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}", exc_info=True)
            raise DocumentProcessingError(f"Error loading document: {str(e)}")