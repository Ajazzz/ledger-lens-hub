import os
import logging
#from fastapi import FastAPI, HTTPException
from fastapi import Request, Response
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from upstash_redis import Redis
import nest_asyncio
nest_asyncio.apply()

# Import your previous logic
from query_engine import LedgerLensQuery

# 1. Setup & Environment
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LedgerLens_Backend")

app = FastAPI(title="LedgerLens API")

# Enable CORS for your Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Your Vite Frontend
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 2. Connect to Upstash Redis for Memory
redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"), 
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
)

# Initialize the Query Engine
engine = LedgerLensQuery()

# 3. Data Models
class ChatRequest(BaseModel):
    user_id: str = "default_user"
    message: str

# 4. API Endpoints
@app.get("/")
async def root():
    return {"status": "LedgerLens API is live"}




# 🚨 THE CORS PREFLIGHT BYPASS
# This intercepts the browser's OPTIONS probe and answers it directly
@app.options("/query")
async def preflight_query(request: Request):
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


@app.post("/query")
async def process_query(request: ChatRequest):
    try:
        # A. Fetch previous context from Upstash (Last 3 messages)
        history_key = f"chat_history:{request.user_id}"
        previous_context = redis.lrange(history_key, 0, 2)
        
        # B. Construct Enhanced Query with Memory
        full_query = f"Context from previous conversation: {previous_context}\nCurrent Question: {request.message}"
        
        # C. Execute RAG
        response = engine.ask(full_query)
        
        # D. Save to Memory (Push to Redis) - STR() APPLIED HERE TO PREVENT CRASH
        redis.lpush(history_key, f"User: {request.message}", f"AI: {str(response)}")
        redis.ltrim(history_key, 0, 10) 
        
        # Extract sources safely
        source_texts = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            source_texts = [n.node.get_content()[:200] + "..." for n in response.source_nodes]
            
        result_data = {
            "answer": str(response),
            "sources": source_texts
        }
        
        # Manually attach CORS headers to the JSON response to prevent blocking
        api_response = JSONResponse(content=result_data)
        api_response.headers["Access-Control-Allow-Origin"] = "*"
        return api_response
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        
        # Manually attach CORS headers to error response so the browser reads the error
        error_response = JSONResponse(
            content={"answer": f"Internal Financial Analysis Error: {str(e)}", "sources": []},
            status_code=500
        )
        error_response.headers["Access-Control-Allow-Origin"] = "*"
        return error_response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
