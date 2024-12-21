# app/services/document_processor.py
import logging
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.errors import DocumentProcessingError
import tempfile
import os
from flask import current_app

# Configure logger
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        """Initialize DocumentProcessor."""
        logger.info("Initializing DocumentProcessor")
        try:
            logger.debug("Setting up text splitter")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
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
        """Get all documents from ChromaDB."""
        try:
            collection = self.vectorstore._collection
            return collection.get()
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}", exc_info=True)
            raise DocumentProcessingError(str(e))

    def delete_document(self, doc_id):
        """Delete a document from ChromaDB."""
        try:
            collection = self.vectorstore._collection
            collection.delete(ids=[doc_id])
            self.vectorstore.persist()
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            raise DocumentProcessingError(str(e))

    def update_document(self, doc_id, metadata):
        """Update document metadata in ChromaDB."""
        try:
            collection = self.vectorstore._collection
            existing = collection.get(ids=[doc_id])
            if not existing['ids']:
                raise DocumentProcessingError("Document not found")
            
            # Merge existing metadata with updates
            current_metadata = existing['metadatas'][0] if existing['metadatas'] else {}
            updated_metadata = {**current_metadata, **metadata}
            
            collection.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )
            self.vectorstore.persist()
            return updated_metadata
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}", exc_info=True)
            raise DocumentProcessingError(str(e))

    def process_file(self, file):
        """Process an uploaded file and store its chunks in ChromaDB."""
        logger.info(f"Processing file: {file.filename} (type: {file.content_type})")
        temp_dir = None
        temp_path = None
        
        try:
            # Create temporary directory and save file
            logger.debug("Creating temporary directory")
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.filename)
            file.save(temp_path)
            logger.debug(f"File saved temporarily at: {temp_path}")
            
            # Load and process document
            logger.debug("Loading document")
            documents = self._load_document(temp_path, file.content_type)
            logger.debug(f"Loaded {len(documents)} document(s)")
            
            logger.debug("Splitting documents into chunks")
            chunks = self.text_splitter.split_documents(documents)
            logger.debug(f"Created {len(chunks)} chunks")
            
            # Add metadata to chunks
            for chunk in chunks:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata['fileName'] = file.filename
                chunk.metadata['contentType'] = file.content_type
            
            # Add documents to ChromaDB using the application's vectorstore
            logger.debug(f"Adding {len(chunks)} chunks to ChromaDB")
            ids = self.vectorstore.add_documents(chunks)
            
            # Persist the changes to disk
            logger.debug("Persisting changes to ChromaDB")
            self.vectorstore.persist()
            
            logger.info(f"Successfully processed and stored {len(chunks)} chunks")
            
            return {
                'fileName': file.filename,
                'chunks': len(chunks),
                'ids': ids
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
            raise DocumentProcessingError(str(e))
            
        finally:
            # Cleanup temporary files
            logger.debug("Cleaning up temporary files")
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)

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