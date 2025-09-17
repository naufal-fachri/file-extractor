import sys
sys.path.append("~/file_extractor")
import os
import asyncio
import tempfile
import subprocess
import uvicorn
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
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

# Import Vector Store
try:
    from file_extractor.tools.vectore_store import QdrantVectorStore
except ImportError as e:
    logger.error(f"Failed to import vector store: {e}")
    raise

# Import OPENAI Embedding
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError as e:
    logger.error(f"Failed to import OpenAI Embeddings: {e}")
    raise

# Import Qdrant Client
try:
    from qdrant_client import AsyncQdrantClient
except ImportError as e:
    logger.error(f"Failed to import Qdrant Client: {e}")
    raise

# Import Document & Text Splitter
try:
    from langchain_core.documents import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    logger.error(f"Failed to import Text Splitter: {e}")
    raise

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")
qdrant_url = os.environ.get("QDRANT_LOCATION_URL")

assert openai_api_key is not None, "OpenAI API KEY doesn't exist."
assert qdrant_api_key is not None, "Qdrant API KEY doesn't exist."
assert qdrant_url is not None, "Qdrant URL doesn't exist."

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
    MODEL_NAME: str = "gpt-4"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 50
    VECTOR_STORE_BATCH_SIZE: int = 100


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
    
    async def chunk_file(self, parsed_file_result: dict, user_id: str, chat_id: str, chunker: RecursiveCharacterTextSplitter) -> list[Document]:
        """
        Chunk parsed file result into a list of documents.
        
        Args:
            parsed_file_result: Dictionary containing parsed file data with 'filename' and 'pages' keys
            chunker: Text splitter instance for chunking documents
            user_id: Unique identifier for the user
            chat_id: Unique identifier for the chat session
            
        Returns:
            List of chunked Document objects
            
        Raises:
            FileChunkingError: If chunking operation fails
            KeyError: If required keys are missing from parsed_file_result
            ValueError: If input parameters are invalid
        """
        # Validate input parameters
        if not parsed_file_result:
            raise ValueError("parsed_file_result cannot be empty")
    
        if not user_id or not chat_id:
            raise ValueError("user_id and chat_id must be non-empty strings")
        
        # Ensure required keys are present
        try:
            file_name = parsed_file_result['filename']
            pages = parsed_file_result['pages']
        except KeyError as e:
            raise KeyError(f"Missing required key in parsed_file_result: {e}")
        
        if not pages:
            logger.warning("No pages found in parsed_file_result")
            return []
        
        # create document
        documents = []
        for page in pages:
            try:
                document = Document(
                    page_content=page["text"],
                    metadata={
                        "full_content": page["text"],
                        "file_name": file_name,
                        "user_id": user_id,
                        "chat_id": chat_id,
                        "page": page["page_index"]
                    }
                )
                documents.append(document)
            except KeyError as e:
                logger.error(f"Missing expected key in page data: {e}")
                continue
        if not documents:
            logger.warning("No valid documents created from pages")
            return []
        
        # Chunk documents
        try:
            logger.info(f"Starting to chunk {len(documents)} documents from file: {file_name}")
            chunked_documents = await chunker.atransform_documents(documents)
            logger.info(
                f"Successfully chunked documents from {len(pages)} pages to "
                f"{len(chunked_documents)} chunks for file: {file_name}"
            )
            return chunked_documents
        except Exception as e:
            logger.error(f"Chunking operation failed: {e}")
            raise Exception(f"Chunking operation failed: {e}")
        
    async def upsert_chunks_to_vector_store(documents: list[Document], batch_size: int, vector_store: QdrantVectorStore) -> bool:
        """
        Upsert chunked documents into the vector store with retry logic.
        
        Args:
            documents: List of Document objects to upsert
            vector_store: Instance of QdrantVectorStore for upserting
            
        Returns:
            True if upsert is successful, False otherwise
            
        Raises:
            ValueError: If input parameters are invalid
        """
        if not documents:
            raise ValueError("documents list cannot be empty")
        
        if not vector_store:
            raise ValueError("vector_store instance is required")

        try:
            if batch_size:
                return await vector_store.aadd_documents(documents, batch_size=batch_size)
            
            return await vector_store.aadd_documents(documents)
        
        except Exception as e:
            logger.error(f"Upsert operation failed: {e}")
            return False

    async def process_file(self, file: UploadFile, user_id: str, chat_id: str) -> Dict[str, Any]:
        """Main file processing method."""
        try:
            # Read and validate file
            content = await file.read()
            FileValidator.validate_file(file, content)
            
            async with self.semaphore:
                logger.info(f"Processing file: {file.filename}")
                
                if file.filename.lower().endswith('.pdf'):
                    extraction_result = await self.extract_pdf(content, file.filename, user_id, chat_id)

                else:  # .docx
                    extraction_result = await self.extract_word(content, file.filename)

            # chunking and upserting document
            logger.info(f"Chunking and upserting document: {file.filename}")

            chunked_documents = await self.chunk_file(
                parsed_file_result=extraction_result,
                user_id=user_id,
                chat_id=chat_id,
                chunker=self.chunker)
            
            logger.info(f"Upserting {len(chunked_documents)} chunks to vector store")
            
            upsert_status = await self.upsert_chunks_to_vector_store(
                documents=chunked_documents,
                batch_size=Config.VECTOR_STORE_BATCH_SIZE,
                vector_store=self.vector_store)
            
            logger.info(f"Upsert status: {upsert_status}")
            logger.info(f"File processing completed: {file.filename}")
            
            return {"status": extraction_result.get("status", False) and upsert_status,
                    "filename": file.filename,
                    "num_pages": len(extraction_result.get("pages", [])),
                    "num_chunks": len(chunked_documents),
                    "error": extraction_result.get("error", None) if not extraction_result.get("status", False) else None}
                    
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
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-large", dimensions=1024)
qdrant_client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
vector_store = QdrantVectorStore(client=qdrant_client,
                                    embedding_function=embeddings,
                                    collection_name="file-chat-history",
                                    vector_name="dense",
                                    retrieval_mode="dense")
chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name=Config.MODEL_NAME,
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ",", "."]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global process_executor, extraction_service
    
    # Startup
    try:
        process_executor = ProcessPoolExecutor(max_workers=Config.MAX_PROCESS_WORKERS)
        extraction_service = FileExtractionService(semaphore=semaphore,
                                                   executor=process_executor,
                                                   chunker=chunker,
                                                   vector_store=vector_store)

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
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )