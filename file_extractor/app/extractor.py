import sys
sys.path.append("~/file_extractor")
import os
import asyncio
import tempfile
import subprocess
import uvicorn
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
from io import BytesIO
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from loguru import logger
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

# Import extractors
try:
    from file_extractor.tools.pdf_extractor import PDFExtractor
    from file_extractor.tools.word_extractor import WordDocumentExtractor
except ImportError as e:
    logger.error(f"Failed to import extractors: {e}")
    raise

@dataclass
class Config:
    """Application configuration."""
    MAX_FILE_SIZE_MB: int = 50 * 1024 * 1024  # 50MB in bytes
    ALLOWED_EXTENSIONS: tuple = ('.pdf', '.docx')
    OCR_TIMEOUT: int = 20
    SESSION_TIMEOUT_HOURS: int = 24
    MAX_CONCURRENT_PROCESSES: int = 4
    MAX_PROCESS_WORKERS: int = max(1, os.cpu_count() // 4)
    OCR_LANGUAGES: str = "eng+ind"


class FileValidationError(Exception):
    """Custom exception for file validation errors."""
    pass


class FileValidator:
    """Handles file validation logic."""
    
    @staticmethod
    def validate_file(file: UploadFile, content: bytes) -> None:
        """Validate uploaded file."""
        if not file.filename:
            raise FileValidationError("No filename provided")
        
        # Check file extension
        if not file.filename.lower().endswith(Config.ALLOWED_EXTENSIONS):
            supported = ', '.join(Config.ALLOWED_EXTENSIONS)
            raise FileValidationError(f"Unsupported file type. Supported types: {supported}")
        
        # Check file size
        if len(content) > Config.MAX_FILE_SIZE_MB:
            max_size_mb = Config.MAX_FILE_SIZE_MB // (1024 * 1024)
            raise FileValidationError(f"File too large. Maximum size is {max_size_mb}MB")
        
        # Check if file is empty
        if len(content) == 0:
            raise FileValidationError("Empty file uploaded")


class OCRProcessor:
    """Handles OCR processing logic."""
    
    @staticmethod
    async def process_with_ocr(file_path: Path, output_path: Path) -> bool:
        """Process file with OCR and return success status."""
        ocr_command = [
            "ocrmypdf",
            "--output-type", "pdf",
            "--jobs", str(Config.MAX_PROCESS_WORKERS),
            "--language", Config.OCR_LANGUAGES,
            "-q", "-f",
            str(file_path),
            str(output_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *ocr_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=Config.OCR_TIMEOUT
            )
            
            if process.returncode != 0:
                logger.error(f"OCR failed: {stderr.decode()}")
                return False
            
            logger.info(f"OCR completed successfully: {output_path}")
            return True
            
        except asyncio.TimeoutError:
            logger.error("OCR process timed out")
            return False
        except Exception as e:
            logger.error(f"OCR process failed: {e}")
            return False


class FileExtractionService:
    """Main service for file extraction operations."""
    
    def __init__(self, semaphore: asyncio.Semaphore, executor: ProcessPoolExecutor):
        self.semaphore = semaphore
        self.executor = executor
    
    async def extract_pdf(self, content: bytes, filename: str, user_id: str, chat_id: str) -> Dict[str, Any]:
        """Extract content from PDF file."""
        extractor = PDFExtractor(max_workers=Config.MAX_PROCESS_WORKERS)
        
        # Try direct extraction first
        result = await extractor.extract_async(
            file=content,
            filename=filename,
            extract_tables=False,
            executor=self.executor
        )
        
        if result.get("status"):
            logger.info("Direct PDF extraction successful")
            return result
        
        # Fallback to OCR
        logger.info("Falling back to OCR extraction")
        return await self._extract_pdf_with_ocr(content, filename, user_id, chat_id, extractor)
    
    async def _extract_pdf_with_ocr(
        self,
        content: bytes,
        filename: str,
        user_id: str,
        chat_id: str,
        extractor: PDFExtractor
    ) -> Dict[str, Any]:
        """Extract PDF content using OCR as fallback."""
        with tempfile.TemporaryDirectory(prefix="file_") as temp_dir:
            file_dir = Path(temp_dir) / user_id / chat_id
            file_dir.mkdir(parents=True, exist_ok=True)
            
            # Save original file
            file_path = file_dir / filename
            file_path.write_bytes(content)
            
            # Process with OCR
            ocr_output_path = file_dir / f"ocr_{filename}"
            ocr_success = await OCRProcessor.process_with_ocr(file_path, ocr_output_path)
            
            if not ocr_success:
                return {"error": "OCR extraction failed", "status": False}
            
            # Try extraction again with OCR'd file
            ocr_content = ocr_output_path.read_bytes()
            result = await extractor.extract_async(
                file=ocr_content,
                filename=filename,
                extract_tables=False,
                executor=self.executor
            )
            
            if result.get("status"):
                logger.info("OCR-based extraction successful")
                return result
            else:
                logger.error("No content found after OCR processing")
                return {"error": "No content found after OCR", "status": False}
    
    async def extract_word(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Extract content from Word document."""
        try:
            extractor = WordDocumentExtractor(infer_table_structure=True)
            result = await extractor.extract_async(
                file=BytesIO(content),
                filename=filename,
                executor=self.executor
            )
            
            if result.get("status"):
                logger.info("Word document extraction successful")
                return result
            else:
                logger.error("No content found in Word document")
                return {"error": "No content found", "status": False}
                
        except Exception as e:
            logger.error(f"Word extraction failed: {e}")
            return {"error": f"Word extraction failed: {str(e)}", "status": False}
    
    async def process_file(self, file: UploadFile, user_id: str, chat_id: str) -> Dict[str, Any]:
        """Main file processing method."""
        try:
            # Read and validate file
            content = await file.read()
            FileValidator.validate_file(file, content)
            
            async with self.semaphore:
                logger.info(f"Processing file: {file.filename}")
                
                if file.filename.lower().endswith('.pdf'):
                    return await self.extract_pdf(content, file.filename, user_id, chat_id)
                else:  # .docx
                    return await self.extract_word(content, file.filename)
                    
        except FileValidationError as e:
            logger.error(f"File validation failed: {e}")
            return {"error": str(e), "status": False}
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return {"error": f"Processing failed: {str(e)}", "status": False}


# Global variables
semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_PROCESSES)
process_executor: Optional[ProcessPoolExecutor] = None
extraction_service: Optional[FileExtractionService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global process_executor, extraction_service
    
    # Startup
    try:
        process_executor = ProcessPoolExecutor(max_workers=Config.MAX_PROCESS_WORKERS)
        extraction_service = FileExtractionService(semaphore, process_executor)
        logger.info(f"Started process executor with {Config.MAX_PROCESS_WORKERS} workers")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    if process_executor:
        logger.info("Shutting down process executor...")
        process_executor.shutdown(wait=True)
        logger.info("Process executor shut down complete")


# FastAPI app initialization
app = FastAPI(
    title="File Extraction API",
    description="API for extracting text content from PDF and Word documents",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post("/extract/{user_id}/{chat_id}")
async def extract_file(file: UploadFile = File(...), user_id: str = "", chat_id: str = ""):
    """
    Extract text content from uploaded PDF or Word document.
    
    Args:
        file: The uploaded file (PDF or DOCX)
        user_id: User identifier
        chat_id: Chat session identifier
    
    Returns:
        JSON response with extraction results
    """
    if not extraction_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    
    try:
        result = await extraction_service.process_file(file, user_id, chat_id)
        
        if result.get("status"):
            return JSONResponse(
                content=jsonable_encoder(result),
                status_code=status.HTTP_200_OK
            )
        else:
            return JSONResponse(
                content=jsonable_encoder(result),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            
    except Exception as e:
        logger.error(f"Unexpected error in extract_file endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "file-extraction-api"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )