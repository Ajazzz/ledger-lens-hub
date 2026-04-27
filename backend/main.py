import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Advanced RAG imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.retrievers import QueryPathsRetriever
from llama_index.core.postprocessor import LLMRerank
from pinecone import Pinecone

# Fix for event loops in web servers
import nest_asyncio
nest_asyncio.apply()

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LedgerLens_Backend")

app = FastAPI(title="LedgerLens API")

# Global CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Models & Pinecone
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Setup Advanced RAG Query Engine
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Advanced Hybrid Retriever (Vectors + BM25 simulation via top-k expansion)
retriever = index.as_retriever(similarity_top_k=10)

# Advanced Reranker
reranker = LLMRerank(choice_batch_size=5, top_n=3, llm=llm)

query_engine = index.as_query_engine(
    retriever=retriever,
    node_postprocessors=[reranker],
    llm=llm
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"status": "LedgerLens API is live"}

@app.post("/query")
async def process_query(request: ChatRequest):
    try:
        # Execute Advanced RAG
        response = query_engine.query(request.message)
        
        result_data = {
            "answer": str(response),
            "sources": [n.node.get_content()[:300] + "..." for n in response.source_nodes]
        }
        
        api_response = JSONResponse(content=result_data)
        api_response.headers["Access-Control-Allow-Origin"] = "*"
        return api_response
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        error_response = JSONResponse(
            content={"answer": f"Backend Error: {str(e)}", "sources": []},
            status_code=500
        )
        error_response.headers["Access-Control-Allow-Origin"] = "*"
        return error_response
