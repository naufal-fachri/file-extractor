import os
import json
import time
import uuid
import shutil
import asyncio
import uvicorn
import tempfile
import subprocess
from loguru import logger
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from file_extractor.tools.pdf_extractor import PDFExtractor
from file_extractor.tools.word_extractor import WordDocumentExtractor
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional, List, Tuple

# Configuration
TEMP_STORAGE_PATH = tempfile.mkdtemp(prefix="pdf_files_")
MAX_FILE_SIZE_MB = 50 * 1024 * 1024 # Maximum file size in MB
ALLOWED_FILE_TYPES = [".pdf", ".docx"] # Allowed file types
OCR_TIMEOUT = 20 # Timeout for OCR in seconds
SESSION_TIMEOUT_HOURS = 24 # Session timeout in hours

# Request Configuration
MAX_CONCURRENT_PROCESS = 4
MAX_PROCESS_WORKERS = (os.cpu_count() // MAX_CONCURRENT_PROCESS) or 1

# Semaphore & Workers Init
semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESS)
process_executor = None

# Request statistics
req_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "request_wait_times": []
}

app = FastAPI("File Extraction API")

app.add_middleware(
    CORSMiddleware,
    all_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global process_executor
    process_executor = ProcessPoolExecutor(max_workers=MAX_PROCESS_WORKERS)
    print("Thread executor started")
    
    yield
    
    # Shutdown
    if process_executor:
        print("Shutting down thread executor...")
        process_executor.shutdown(wait=True)
        print("Thread executor shut down complete")

@app.post("/extract")
async def extract_file(file: UploadFile=File(...)):

    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOC files are supported"
        )
    
    try:
        file_content = await file.read()
        filename = file.filename

        if len(file_content) > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=423,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB // (1024 * 1024)}MB"
            )
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empyt file uploaded"
            )
        
        async with semaphore:
            logger.info(f"Request ({file.filename}): ACQUIRED a spot. Processing file.")

            if file.filename.lower().endswith(".pdf"):
                extractor = PDFExtractor(max_workers=MAX_PROCESS_WORKERS)
                result = await asyncio.create_task(extractor.extract_async(file, filename, extract_tables=False, executor=process_executor))
                return JSONResponse(content=result)
            
            else:
                extractor = WordDocumentExtractor(infer_table_structure=True)
                result = await asyncio.create_task(extractor.extract_async(file, filename, executor=process_executor))
                return JSONResponse(content=result)
    pass




