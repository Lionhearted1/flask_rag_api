import logging
import os
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
        elif llm_type == 'google':
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
        
        logger.info("Using OpenAI LLM with model: %s", model)
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
            # api_key = config.google.api_key
            api_key=os.getenv('GOOGLE_API_KEY')
            model = getattr(config.google, 'model', 'gemini-1.5-flash')
        
        logger.info("Using Google Gemini LLM with model: %s", model)
        return ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model
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
        
        logger.info("Using OpenAI embeddings")
        return OpenAIEmbeddings(
            api_key=api_key
        )

    @staticmethod
    def _create_google_embeddings(config):
        if isinstance(config, dict):
            api_key = config.get('google', {}).get('api_key')
        else:
            # api_key = config.google.api_key
            api_key=os.getenv('GOOGLE_API_KEY')
        
        logger.info("Using Google embeddings")
        # return GoogleGenerativeAIEmbeddings(
        #     google_api_key=api_key,
        #     model=os.getenv('')
        # )
        return current_app.embeddings


class ChatService:
    def __init__(self, config):
        """Initialize ChatService with configuration."""
        logger.info("Initializing ChatService")
        try:
            # Store the config object
            self.config = config
            
            # Initialize LLM
            logger.debug("Initializing LLM")
            self.llm = LLMFactory.create_llm(self.config)
            
            # Initialize Pinecone
            logger.debug("Initializing Pinecone")
            pinecone_config = self.config.pinecone if hasattr(self.config, 'pinecone') else self.config.get('pinecone', {})
            
            # Create an instance of the Pinecone class
            self.pinecone_client = pinecone.Pinecone(
                api_key=pinecone_config.get('api_key')
            )
            
            # Initialize embeddings
            self.embeddings = EmbeddingsFactory.create_embeddings(self.config)
            
            # Initialize vector store
            self.index_name = pinecone_config.get('index_name', 'saanvi-production')
            self.vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings
            )
            
            logger.debug("Setting up chat prompt template")
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are Saanvi, an AI assistant specializing in cybersecurity. Your purpose is to provide clear and accurate answers based on the cyber security context provided through the book 'Cyber Safe Girl' by Dr. Ananth Prabhu G. If the context doesn't match the query, use your knowledge about cybersecurity, but maintain the focus on digital safety and security.
                 Do not explictly mention about the source of the knowledge in your responses such as "the provided text ..."

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