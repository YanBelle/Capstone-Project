from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import json

from app.api.endpoints import router, initialize_model

app = FastAPI(title="BERT Debug Dashboard API")

# Configure CORS - Updated to allow port 3333
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3333", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    print("FastAPI starting up...")
    success = initialize_model()
    if success:
        print("Application startup complete with model loaded")
    else:
        print("Application startup complete but model failed to load")

@app.get("/")
async def root():
    return {"message": "BERT Debug Dashboard API", "status": "running"}
