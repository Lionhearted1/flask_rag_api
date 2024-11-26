from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from app.utils.errors import ChatProcessingError

class ChatService:
    def __init__(self, db, config):
        self.db = db
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vectorstore = MongoDBAtlasVectorSearch(
            collection=self.db.documents,
            embedding=self.embeddings,
            index_name="default"
        )
        
        self.llm = ChatOpenAI(
            api_key=config.openai['api_key'],
            base_url=config.openai['base_url'],
            model_name=config.openai['model']
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context.\n\nContext: {context}"),
            ("human", "{question}")
        ])
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True
        )

    def chat(self, message):
        try:
            docs = self.vectorstore.similarity_search(message, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            response = self.chain.invoke({
                "context": context,
                "question": message
            })
            return response["text"]
        except Exception as e:
            raise ChatProcessingError(str(e))