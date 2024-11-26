from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from app.utils.errors import DocumentProcessingError
import tempfile
import os

class DocumentProcessor:
    def __init__(self, db):
        self.db = db
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def process_file(self, file):
        try:
            # Save file temporarily
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.filename)
            file.save(temp_path)

            # Load and process document
            documents = self._load_document(temp_path, file.content_type)
            chunks = self.text_splitter.split_documents(documents)
            
            processed_docs = []
            for i, chunk in enumerate(chunks):
                embedding = self.embeddings.embed_query(chunk.page_content)
                doc = {
                    'content': chunk.page_content,
                    'embedding': embedding,
                    'metadata': {
                        'fileName': file.filename,
                        'fileType': file.content_type,
                        'chunkIndex': i,
                        **chunk.metadata
                    }
                }
                processed_docs.append(doc)

            # Save to MongoDB
            result = self.db.documents.insert_many(processed_docs)
            
            return {
                'fileName': file.filename,
                'chunks': len(processed_docs),
                'ids': [str(id) for id in result.inserted_ids]
            }

        except Exception as e:
            raise DocumentProcessingError(str(e))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(temp_dir)

    def _load_document(self, file_path, content_type):
        if content_type == 'application/pdf':
            loader = PyPDFLoader(file_path)
        elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            loader = Docx2txtLoader(file_path)
        elif content_type == 'text/plain':
            loader = TextLoader(file_path)
        elif content_type == 'text/csv':
            loader = CSVLoader(file_path)
        else:
            raise DocumentProcessingError(f"Unsupported file type: {content_type}")
        
        return loader.load()