# app/services/chat_service.py
import logging
import re
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from app.utils.errors import ChatProcessingError
from flask import current_app

# Configure logger
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, config):
        """Initialize ChatService with configuration."""
        logger.info("Initializing ChatService")
        try:
            self.config = config
            
            logger.debug("Initializing OpenAI Chat model")
            self.llm = ChatOpenAI(
                api_key=self.config.openai['api_key'],
                base_url=self.config.openai['base_url'],
                model_name=self.config.openai['model']
            )
            
            logger.debug("Setting up chat prompt template")
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a legal assistant that provides clear and accurate answers based on the given context or your knowledge base about Indian Constitution. if context doesnt match the query use your knaowledge, dont complain that the context provided does not include any information about blah blah blah.

Instructions for formatting your responses:
- Use only plain text without any special formatting
- No markdown, no bullet points, no numbered lists
- No special characters or symbols for formatting
- Present information in a natural, narrative style
- If listing multiple items, separate them with commas in a sentence
- Keep answers focused and relevant to the question

Context: {context}

Remember to only use information present in the context. If information isn't available in the context, use your knowledge thats it,."""),
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