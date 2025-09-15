import os
import re
import asyncio
import pymupdf
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from loguru import logger

__version__ = "0.1.0"
__all__ = ["PDFExtractor", "PDFExtractionError"]


class PDFExtractionError(Exception):
    """Custom exception for PDF extraction errors."""
    pass


class TextProcessor:
    """Text processing utilities for cleaning extracted content."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from PDF extraction
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
            
        # Replace newlines with spaces
        text = text.replace("\n", " ")
        
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        
        # Add space between lowercase and uppercase letters
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove space before punctuation
        text = re.sub(r'\s+([.,!:?])', r'\1', text)
        
        # Add space after punctuation if not preceded by a digit
        text = re.sub(r'(?<!\d)([.,])([A-Za-z0-9])', r'\1 \2', text)
        
        return text.strip()
    
    @staticmethod
    def clean_table(table_markdown: str) -> str:
        """
        Clean table markdown formatting.
        
        Args:
            table_markdown: Raw table markdown from PDF extraction
            
        Returns:
            Cleaned table markdown
        """
        if not table_markdown:
            return ""
            
        # Remove <br> tags while preserving table structure
        table_markdown = re.sub(
            r'(<br>)(?=\|)|(?<=\|)(<br>)|<br>', 
            lambda m: '' if m.group(1) or m.group(2) else ' ', 
            table_markdown
        )
        
        return table_markdown.strip()


def _extract_page_content(file: bytes, page_index: int, extract_tables: bool = False) -> Dict[str, Any]:
    """
    Worker function for extracting content from a single page.
    
    This function is designed to work with multiprocessing.
    
    Args:
        pdf_path: Path to the PDF file
        page_index: Index of the page to extract
        extract_tables: Whether to extract tables
        
    Returns:
        PDFPageResult containing the extracted content
    """
    try:
        with pymupdf.open(stream=file, filetype="pdf") as pdf_document:
            page = pdf_document.load_page(page_index)
            
            # Extract text
            raw_text = page.get_text(sort=True)
            cleaned_text = TextProcessor.clean_text(raw_text)
            
            # Extract tables if requested
            tables = None
            if extract_tables:
                raw_tables = page.find_tables()
                tables = [
                    TextProcessor.clean_table(table.to_markdown()) 
                    for table in raw_tables
                ]
            
            # Determine status
            has_content = bool(cleaned_text or (tables and any(tables)))

            return {
                "page_index": page_index,
                "text": cleaned_text,
                "tables": tables,
                "status": has_content
            }

            
    except Exception as e:
        error_msg = f"Error extracting content from page {page_index}: {str(e)}"
        logger.error(error_msg)

        return {
            "page_index": page_index,
            "text": "",
            "tables": [] if extract_tables else None,
            "status": False
        }
        


class PDFExtractor:
    """
    High-performance PDF text and table extractor using concurrent processing.
    
    This class provides methods to extract text and tables from PDF files
    using multiprocessing for improved performance on multi-page documents.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the PDF extractor.
        
        Args:
            max_workers: Maximum number of worker processes. If None, uses
                        min(32, (os.cpu_count() or 1) + 4)
        """
        if max_workers is None:
            max_workers = (os.cpu_count() - 2) or 1
        
        self.max_workers = max_workers
        logger.info(f"Initialized PDFExtractor with {max_workers} max workers")
    
    def validate_pdf_path(self, file_path: Union[str, Path]) -> Path:
        """
        Validate and normalize the PDF file path.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Normalized Path object
            
        Raises:
            PDFExtractionError: If path is invalid or file doesn't exist
        """
        if not file_path:
            raise PDFExtractionError("File path cannot be empty")
        
        path = Path(file_path)
        
        if not path.exists():
            raise PDFExtractionError(f"PDF file not found: {path}")
        
        if not path.suffix.lower() == '.pdf':
            raise PDFExtractionError(f"File must have .pdf extension: {path}")
        
        return path
    
    def get_pdf_info(self, file: bytes, filename: str) -> Dict[str, Any]:
        """
        Get basic information about the PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
            
        Raises:
            PDFExtractionError: If unable to read PDF file
        """
        
        try:
            with pymupdf.open(stream=file, filetype="pdf") as doc:
                return {
                    "filename": filename,
                    "total_pages": doc.page_count,
                    "metadata": doc.metadata
                }
        except Exception as e:
            raise PDFExtractionError(f"Unable to read PDF metadata: {str(e)}") from e
    
    async def extract_async(
        self,
        file: bytes,
        filename: str,
        extract_tables: bool = False,
        executor: ProcessPoolExecutor = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously extract text and optionally tables from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            extract_tables: Whether to extract tables in addition to text
            max_workers: Override the default max_workers for this extraction
            
        Returns:
            PDFExtractionResult containing all extracted content
            
        Raises:
            PDFExtractionError: If extraction fails
        """
        pdf_info = self.get_pdf_info(file=file, filename=filename)
                
        logger.info(
            f"Starting {'text and table' if extract_tables else 'text'} extraction "
            f"for {pdf_info['filename']} ({pdf_info['total_pages']} pages) "
            f"with {self.max_workers} workers"
        )
        
        try:
            loop = asyncio.get_event_loop()
            
            # Create tasks for all pages
            tasks = [
                loop.run_in_executor(
                    executor,
                    _extract_page_content,
                    file,
                    page_index,
                    extract_tables
                )
                for page_index in range(pdf_info["total_pages"])
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            page_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                    page_results.append({
                        "page_index": result["page_index"],
                        "text": "",
                        "tables": [] if extract_tables else None,
                        "status": False
                    })
                else:
                    page_results.append(result)
            
            # Sort by page index
            page_results.sort(key=lambda x: x["page_index"])
            
            # Calculate statistics
            success_count = sum(1 for page in page_results if page["status"])
            failed_count = len(page_results) - success_count
            
            logger.info(
                f"Extraction completed for {pdf_info['filename']}: "
                f"{success_count}/{pdf_info['total_pages']} pages successful"
            )
            
            return {"filename": pdf_info["filename"],
                    "total_pages": pdf_info["total_pages"],
                    "pages": page_results,
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "status": True if (success_count/len(page_results)) == 1 else False,
                    "file_type": "pdf" 
                    }

        except Exception as e:
            raise PDFExtractionError(f"Failed to extract PDF content: {str(e)}") from e
    
    def extract(
        self,
        file: bytes,
        filename: str,
        extract_tables: bool = False,
        max_workers: Optional[int] = None,
        executor: ProcessPoolExecutor = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Synchronously extract text and optionally tables from a PDF file.
        
        This is a convenience wrapper around the async method.
        
        Args:
            pdf_path: Path to the PDF file
            extract_tables: Whether to extract tables in addition to text
            max_workers: Override the default max_workers for this extraction
            timeout: Maximum time to wait for extraction (in seconds)
            
        Returns:
            PDFExtractionResult containing all extracted content
            
        Raises:
            PDFExtractionError: If extraction fails or times out
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                coro = self.extract_async(file, filename, extract_tables, max_workers, executor)
                
                if timeout:
                    return loop.run_until_complete(
                        asyncio.wait_for(coro, timeout=timeout)
                    )
                else:
                    return loop.run_until_complete(coro)
                    
            except asyncio.TimeoutError:
                raise PDFExtractionError(f"Extraction timed out after {timeout} seconds")
                
        finally:
            if not loop.is_closed():
                loop.close()