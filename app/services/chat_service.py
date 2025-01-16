import logging
import re
import os
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from app.utils.errors import ChatProcessingError
from flask import current_app
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure logger
logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def create_llm(config):
        """Create LLM instance based on configuration."""
        # Handle both Config object and dictionary
        if isinstance(config, dict):
            llm_type = config.get('llm_type', 'openai').lower()
            openai_config = config.get('openai', {})
            google_config = config.get('google', {})
        else:
            llm_type = getattr(config, 'llm_type', 'openai').lower()
            openai_config = config.openai if hasattr(config, 'openai') else {}
            google_config = config.google if hasattr(config, 'google') else {}
        
        if llm_type == 'openai':
            if not openai_config.get('api_key'):
                logger.warning("OpenAI API key not found, falling back to Google Gemini")
                return LLMFactory._create_gemini(config)
            return LLMFactory._create_openai(config)
        elif llm_type == 'gemini':
            if not google_config.get('api_key'):
                logger.warning("Google API key not found, falling back to OpenAI")
                return LLMFactory._create_openai(config)
            return LLMFactory._create_gemini(config)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    @staticmethod
    def _create_openai(config):
        if isinstance(config, dict):
            openai_config = config.get('openai', {})
            api_key = openai_config.get('api_key')
            model = openai_config.get('model', 'gpt-4o')
        else:
            api_key = config.openai.api_key
            model = getattr(config.openai, 'model', 'gpt-4o')
        
        return ChatOpenAI(
            api_key=api_key,
            model_name=model
        )

    @staticmethod
    def _create_gemini(config):
        if isinstance(config, dict):
            google_config = config.get('google', {})
            api_key = google_config.get('api_key')
            model = google_config.get('model', 'gemini-1.5-flash')
        else:
            api_key = config.google.api_key
            model = getattr(config.google, 'model', 'gemini-1.5-flash')
        
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model_name=model
        )

class EmbeddingsFactory:
    @staticmethod
    def create_embeddings(config):
        """Create embeddings instance based on configuration."""
        # Handle both Config object and dictionary
        if isinstance(config, dict):
            embedding_type = config.get('embedding_type', 'openai').lower()
            openai_config = config.get('openai', {})
            google_config = config.get('google', {})
        else:
            embedding_type = getattr(config, 'embedding_type', 'openai').lower()
            openai_config = config.openai if hasattr(config, 'openai') else {}
            google_config = config.google if hasattr(config, 'google') else {}
        
        if embedding_type == 'openai':
            if not openai_config.get('api_key'):
                logger.warning("OpenAI API key not found, falling back to Google embeddings")
                return EmbeddingsFactory._create_google_embeddings(config)
            return EmbeddingsFactory._create_openai_embeddings(config)
        elif embedding_type == 'google':
            if not google_config.get('api_key'):
                logger.warning("Google API key not found, falling back to OpenAI embeddings")
                return EmbeddingsFactory._create_openai_embeddings(config)
            return EmbeddingsFactory._create_google_embeddings(config)
        else:
            raise ValueError(f"Unsupported embeddings type: {embedding_type}")

    @staticmethod
    def _create_openai_embeddings(config):
        if isinstance(config, dict):
            api_key = config.get('openai', {}).get('api_key')
        else:
            api_key = config.openai.api_key
        
        return OpenAIEmbeddings(
            api_key=api_key
        )

    @staticmethod
    def _create_google_embeddings(config):
        if isinstance(config, dict):
            api_key = config.get('google', {}).get('api_key')
        else:
            api_key = config.google.api_key
        
        return GoogleGenerativeAIEmbeddings(
            google_api_key=api_key
        )

class ChatService:
    def __init__(self, config):
        """Initialize ChatService with configuration."""
        logger.info("Initializing ChatService")
        try:
            # Ensure config is an instance of Config, not a dictionary
            if isinstance(config, dict):
                raise ValueError("Config must be an instance of the Config class, not a dictionary.")
            
            # Use the config object directly
            self.config = config
            
            # Initialize LLM
            logger.debug("Initializing LLM")
            self.llm = LLMFactory.create_llm(self.config)
            
            # Initialize Pinecone
            logger.debug("Initializing Pinecone")
            pinecone.init(
                api_key=self.config.pinecone.api_key,  # Use dot notation
                environment=self.config.pinecone.environment  # Use dot notation
            )
            
            # Initialize embeddings
            self.embeddings = EmbeddingsFactory.create_embeddings(self.config)
            
            # Initialize vector store
            self.index_name = self.config.pinecone.index_name  # Use dot notation
            self.vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings
            )
            
            
            
            logger.debug("Setting up chat prompt template")
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are Saanvi, an AI assistant specializing in cybersecurity. Your purpose is to provide clear and accurate answers based on the cyber security context provided through the book 'Cyber Safe Girl' by Dr. Ananth Prabhu G. If the context doesn't match the query, use your knowledge about cybersecurity, but maintain the focus on digital safety and security.

Instructions for formatting your responses:
- Use only plain text without any special formatting
- No markdown, no bullet points, no numbered lists
- No special characters or symbols for formatting
- Present information in a natural, narrative style
- If listing multiple items, separate them with commas in a sentence
- Keep answers focused and relevant to the question

Context: {context}

Remember to focus on cybersecurity aspects and digital safety while maintaining a helpful and informative tone."""),
                ("human", "{question}")
            ])
            
            logger.debug("Creating LLMChain")
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                verbose=True
            )
            
            logger.info("ChatService initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatService: {str(e)}", exc_info=True)
            raise


    def extract_searchable_content(self, doc) -> List[str]:
        """Extract all searchable content from a document, including metadata."""
        searchable_content = []
        
        # Add page content
        if doc.page_content:
            searchable_content.append(doc.page_content)
        
        # Add relevant metadata content
        for key, value in doc.metadata.items():
            # Skip technical metadata
            if key not in ['source', 'row_index', 'contentType', 'has_headers']:
                if isinstance(value, (str, int, float)):
                    searchable_content.append(str(value))
        
        return searchable_content

    def calculate_match_score(self, query: str, text: str) -> float:
        """Calculate how well a piece of text matches the query."""
        score = 0.0
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Exact phrase match
        if query_lower in text_lower:
            score += 10.0
        
        # Word matches
        query_words = set(re.findall(r'\w+', query_lower))
        text_words = set(re.findall(r'\w+', text_lower))
        word_overlap = len(query_words & text_words)
        score += word_overlap * 2.0
        
        # Partial matches
        for word in query_words:
            if len(word) > 3:  # Only check substantial words
                if any(word in text_word for text_word in text_words):
                    score += 0.5
        
        # Positional scoring (higher score if matches appear early in text)
        first_occurrence = text_lower.find(query_words.pop() if query_words else query_lower)
        if first_occurrence >= 0:
            position_score = 1.0 - (first_occurrence / len(text_lower))
            score += position_score * 2.0
        
        return score

    def find_best_matches(self, query: str, docs: List, num_matches: int = 5) -> List:
        """Find the most relevant documents for the query."""
        logger.debug(f"Finding best matches for query: {query[:50]}...")
        
        scored_docs = []
        for doc in docs:
            max_score = 0.0
            matched_content = None
            
            # First check metadata for matches
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float)):
                    score = self.calculate_match_score(query, str(value))
                    if score > max_score:
                        max_score = score
                        matched_content = str(value)
            
            # Then check page content
            score = self.calculate_match_score(query, doc.page_content)
            if score > max_score:
                max_score = score
                matched_content = doc.page_content
            
            if max_score > 0:
                # Store both the matched content and original document
                scored_docs.append((doc, matched_content, max_score))
        
        # Sort by score and remove duplicates while preserving original content
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        unique_docs = []
        seen_content = set()
        
        for doc, matched_content, _ in scored_docs:
            content_hash = hash(matched_content)
            if content_hash not in seen_content and len(unique_docs) < num_matches:
                seen_content.add(content_hash)
                # Create a new document with the matched content and original metadata
                from langchain_core.documents import Document
                new_doc = Document(
                    page_content=matched_content,
                    metadata={
                        **doc.metadata,
                        'original_content': doc.page_content
                    }
                )
                unique_docs.append(new_doc)
        
        logger.debug(f"Found {len(unique_docs)} unique matching documents")
        return unique_docs

    def format_document_content(self, doc) -> str:
        """Format document content while preserving structure."""
        logger.debug("Formatting document content")
        
        formatted_parts = []
        
        # Add the matched content first
        if doc.page_content:
            cleaned_content = doc.page_content.strip()
            if cleaned_content:
                formatted_parts.append(cleaned_content)
        
        # Add source information
        source_info = f"[Source: {doc.metadata.get('source', 'Unknown')}]" if doc.metadata.get('source') else ""
        
        # For CSV files or other structured data, include any additional relevant metadata
        relevant_metadata = []
        for key, value in doc.metadata.items():
            # Skip technical or duplicate metadata
            if (key not in ['source', 'row_index', 'contentType', 'has_headers', 'original_content'] 
                and isinstance(value, (str, int, float)) 
                and str(value) not in formatted_parts):
                relevant_metadata.append(f"{key}: {value}")
        
        if relevant_metadata:
            metadata_str = " | ".join(relevant_metadata)
            if metadata_str:
                formatted_parts.append(metadata_str)
        
        # Join all parts
        final_content = " | ".join(formatted_parts)
        if source_info:
            final_content = f"{final_content} {source_info}"
        
        return final_content

# app/services/chat_service.py

    def chat(self, message: str) -> str:
        """Process a chat message and return the response."""
        logger.info(f"Processing chat message: {message[:50]}...")
        try:
            # Get initial documents
            vectorstore = current_app.vectorstore
            docs = vectorstore.similarity_search(message, k=10)
            
            # Format and combine relevant documents
            formatted_docs = []
            for doc in docs:
                # Get content and metadata
                content = doc.page_content
                metadata = doc.metadata
                
                # Add source if available
                source_info = f" [Source: {metadata.get('source', 'Unknown')}]"
                
                # Add any additional metadata that matches the query
                query_terms = message.lower().split()
                relevant_metadata = []
                
                for key, value in metadata.items():
                    if key not in ['source', 'row_index', 'content_type']:
                        value_str = str(value).lower()
                        if any(term in value_str or term in key.lower() for term in query_terms):
                            relevant_metadata.append(f"{key}: {value}")
                
                # Combine content and relevant metadata
                if relevant_metadata:
                    formatted_content = f"{content}\nAdditional information:\n{', '.join(relevant_metadata)}"
                else:
                    formatted_content = content
                    
                formatted_docs.append(f"{formatted_content}{source_info}")
            
            # Join documents with clear separator
            context = " || ".join(formatted_docs)
            
            # Get response from LLM
            response = self.chain.invoke({
                "context": context,
                "question": message
            })
            
            # Clean response
            clean_response = ' '.join(response['text']
                .replace('*', '')
                .replace('_', '')
                .replace('#', '')
                .replace('`', '')
                .replace('\n', ' ')
                .strip()
                .split())
            
            return clean_response
            
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}", exc_info=True)
            raise ChatProcessingError(str(e))