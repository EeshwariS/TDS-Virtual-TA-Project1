# app.py
import os
import json
import re
import logging
import asyncio
import traceback
from typing import Optional, List, Dict, Any

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pinecone import Pinecone

# Load environment variables
load_dotenv()
API_KEY = os.getenv("AIPIPE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index("discourse-embeddings")

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"message": "Welcome to the TDS Virtual TA API", "status": "ok"}

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Embedding helper
async def get_embedding(text: str) -> List[float]:
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": text}
    async with aiohttp.ClientSession() as session:
        async with session.post("https://aipipe.org/openai/v1/embeddings", headers=headers, json=payload) as r:
            r.raise_for_status()
            return (await r.json())["data"][0]["embedding"]

# Query Pinecone index
async def query_pinecone(embedding: List[float], top_k=5) -> List[Dict[str, Any]]:
    response = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return [
        {
            "url": m.metadata.get("url", ""),
            "text": m.metadata.get("combined_text", "")[:300],
            "score": m.score,
        }
        for m in response.matches
    ]

# Call LLM for answer generation
async def call_llm(question: str, contexts: List[str]) -> str:
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    content = "\n\n---\n\n".join(contexts)
    messages = [
        {"role": "system", "content": "Answer based on forum context only."},
        {"role": "user", "content": f"{content}\n\nQuestion: {question}"},
    ]
    payload = {"model": "gpt-4o-mini", "messages": messages, "temperature": 0.3}
    async with aiohttp.ClientSession() as session:
        async with session.post("https://aipipe.org/openai/v1/chat/completions", headers=headers, json=payload) as r:
            r.raise_for_status()
            return (await r.json())["choices"][0]["message"]["content"]

# Main endpoint
@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="AIPIPE_API_KEY not set")
    try:
        embedding = await get_embedding(request.question)
        results = await query_pinecone(embedding)
        if not results:
            return QueryResponse(answer="No relevant content found.", links=[])
        context_texts = [r["text"] for r in results]
        answer = await call_llm(request.question, context_texts)
        links = [LinkInfo(url=r["url"], text=r["text"]) for r in results]
        return QueryResponse(answer=answer, links=links)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok", "api_key_set": bool(API_KEY)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
 
