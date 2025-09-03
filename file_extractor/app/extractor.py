import os
import asyncio
import uvicorn
import tempfile
import subprocess
from loguru import logger
from io import BytesIO
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from file_extractor.tools.pdf_extractor import PDFExtractor
from file_extractor.tools.word_extractor import WordDocumentExtractor
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

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

@app.post("/extract/{useriId}/{chatId}")
async def extract_file(file: UploadFile, userId: str, chatId: str):

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
                detail="Empyty file uploaded"
            )

        async with semaphore:
            logger.info(f"Request ({file.filename}): ACQUIRED a spot. Processing file.")

            if file.filename.lower().endswith(".pdf"):
                with tempfile.TemporaryDirectory(prefix="file_") as temp_dir:

                        file_dir = os.path.join(temp_dir, userId, chatId)
                        os.makedirs(file_dir, exist_ok=True)

                        file_path = os.path.join(file_dir, filename)
                        with open(file_path, "wb") as file:
                            file.write(binary_file)

                        # start pdf extraction
                        extractor = PDFExtractor(max_workers=MAX_PROCESS_WORKERS)
                        result = await extractor.extract_async(file=binary_file, filename=filename, extract_tables=False, executor=process_executor)

                        if result["status"]:
                            logger.info("Extraction successful.")
                            return JSONResponse(jsonable_encoder(result))
                        
                        # Fall back to OCR
                        logger.info("Fall back to OCR since no text extracted.")
                        ocr_output_path = os.path.join(file_dir, f"ocr_{filename}")
                        ocr_command = [
                                    "ocrmypdf", "--output-type", "pdf", "--jobs", str(MAX_PROCESS_WORKERS), "--language", "eng+ind", "-q", "-f", file_path, ocr_output_path
                                ]
                        try:
                            process = await asyncio.create_subprocess_exec(*ocr_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            stdout, stderr = await process.communicate()

                            if process.returncode != 0:
                                logger.error(f"OCR extraction failed: {stderr.decode()}")
                                return JSONResponse(jsonable_encoder({"error": "OCR extraction failed"}))
                            
                            logger.info(f"OCR extraction completed successfully. Output saved to {ocr_output_path}")
                            logger.info("Retry extraction again.")

                            with open(ocr_output_path, 'rb') as file:
                                binary_file = file.read()

                            result = await extractor.extract_async(file=binary_file, filename=filename, extract_tables=False, executor=process_executor)

                            if result["status"]:
                                logger.info("Extraction successful.")
                                return JSONResponse(jsonable_encoder(result))
                            
                            else:
                                logger.error("No content found after being ocrd.")
                                return JSONResponse(jsonable_encoder({"error": "No content found"}))
                            
                        except Exception as e:
                            raise HTTPException(
                                status_code=400,
                                detail=str(e)
                            )
                
            else:
                try:
                    extractor = WordDocumentExtractor(infer_table_structure=True)
                    result = await extractor.extract_async(file=BytesIO(binary_file), filename=filename, executor=process_executor)

                    if result["status"]:
                        logger.info("Extraction successful.")
                        return JSONResponse(jsonable_encoder(result))
                    
                    else:
                        logger.error("No content found")
                        return JSONResponse(jsonable_encoder({"error": "NO content found"}))
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=200,
                        detail=str(e)
                    )
                
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)




