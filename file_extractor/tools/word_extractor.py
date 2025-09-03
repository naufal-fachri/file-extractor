import re
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor

try:
    from unstructured.partition.doc import partition_doc
    from unstructured.partition.docx import partition_docx
except ImportError as e:
    raise ImportError(
        "Required dependencies not found. Install with: pip install unstructured beautifulsoup4"
    ) from e

__version__ = "0.1.0"
__all__ = [
    "WordDocumentExtractor", 
    "WordExtractionError",
]


class WordExtractionError(Exception):
    """Custom exception for Word document extraction errors."""
    pass

class MarkdownProcessor:
    """Utilities for processing and formatting extracted content as Markdown."""
    
    @staticmethod
    def process_title(text: str) -> str:
        """
        Process title elements with proper Markdown formatting.
        
        Args:
            text: Raw title text
            
        Returns:
            Markdown-formatted title
        """
        clean_text = text.strip()
        if not clean_text:
            return ""
        return f"### {clean_text}"
    
    @staticmethod
    def process_text(text: str) -> str:
        """
        Process general text elements.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        return text.strip()
    
    @staticmethod
    def process_list_item(text: str) -> str:
        """
        Process list items with proper Markdown formatting.
        
        Args:
            text: Raw list item text
            
        Returns:
            Markdown-formatted list item
        """
        clean_text = text.strip()
        if not clean_text:
            return ""
        return f"- {clean_text}"
    
    @staticmethod
    def process_table(element) -> str:
        """
        Process table elements with HTML to Markdown conversion.
        
        Args:
            element: Unstructured table element
            
        Returns:
            Markdown-formatted table
        """
        text = element.text.strip() if element.text else ""
        
        # Try to extract HTML table from metadata
        html_table = MarkdownProcessor._extract_html_table(element)
        if html_table:
            return MarkdownProcessor._convert_html_table_to_markdown(html_table)
        
        # Fallback: process as text table
        return MarkdownProcessor._process_text_table(text)
    
    @staticmethod
    def _extract_html_table(element) -> Optional[str]:
        """
        Extract HTML table from element metadata.
        
        Args:
            element: Unstructured element
            
        Returns:
            HTML table string if found, None otherwise
        """
        if not hasattr(element, 'metadata') or not element.metadata:
            return None
            
        metadata = element.metadata
        
        # Check various possible metadata attributes for HTML table
        for attr in ['text_as_html', 'table_html', 'html']:
            if hasattr(metadata, attr):
                html_content = getattr(metadata, attr)
                if html_content:
                    return html_content
        
        return None
    
    @staticmethod
    def _convert_html_table_to_markdown(html_table: str) -> str:
        """
        Convert HTML table to Markdown table format.
        
        Args:
            html_table: HTML table string
            
        Returns:
            Markdown-formatted table
        """
        try:
            soup = BeautifulSoup(html_table, 'html.parser')
            table = soup.find('table')
            
            if not table:
                logger.warning("No table found in HTML, returning as code block")
                return f"```html\n{html_table}\n```"
            
            rows = table.find_all('tr')
            if not rows:
                logger.warning("HTML table has no rows, returning as code block")
                return f"```html\n{html_table}\n```"
            
            markdown_rows = []
            
            for i, row in enumerate(rows):
                cells = row.find_all(['td', 'th'])
                if not cells:
                    continue
                    
                # Clean cell content and handle multiline text
                clean_cells = []
                for cell in cells:
                    cell_text = cell.get_text(strip=True)
                    # Replace newlines in cells with spaces
                    cell_text = cell_text.replace('\n', ' ').replace('\r', ' ')
                    # Escape pipe characters that would break markdown
                    cell_text = cell_text.replace('|', '\\|')
                    clean_cells.append(cell_text)
                
                markdown_row = "| " + " | ".join(clean_cells) + " |"
                markdown_rows.append(markdown_row)
                
                # Add header separator after first row
                if i == 0:
                    separator = "| " + " | ".join(["---"] * len(clean_cells)) + " |"
                    markdown_rows.append(separator)
            
            return "\n".join(markdown_rows)
            
        except Exception as e:
            logger.warning(f"Failed to parse HTML table: {e}")
            return f"```html\n{html_table}\n```"
    
    @staticmethod
    def _process_text_table(text: str) -> str:
        """
        Process table as plain text format.
        
        Args:
            text: Raw table text
            
        Returns:
            Formatted table text
        """
        if not text:
            return ""
        return f"**Table:**\n```\n{text}\n```"


class ElementProcessor:
    """Handles processing of different document element types."""
    
    # Element type mappings for consistent handling
    TITLE_TYPES = {"Title", "Header"}
    TEXT_TYPES = {"Text", "NarrativeText", "UncategorizedText"}
    LIST_TYPES = {"ListItem", "BulletPoint"}
    TABLE_TYPES = {"Table"}
    PAGE_BREAK_TYPES = {"PageBreak"}
    
    def __init__(self):
        self.processor = MarkdownProcessor()
    
    def get_element_type(self, element) -> str:
        """
        Determine the element type from the element object.
        
        Args:
            element: Unstructured element
            
        Returns:
            Normalized element type string
        """
        # Try to get type from class name first
        element_type = type(element).__name__
        
        # Fall back to category attribute if available
        if hasattr(element, 'category') and element.category:
            element_type = element.category
        
        return element_type
    
    def process_element(self, element) -> Tuple[str, str, bool]:
        """
        Process a single document element.
        
        Args:
            element: Unstructured element to process
            
        Returns:
            Tuple of (element_type, processed_content, is_page_break)
        """
        element_type = self.get_element_type(element)
        text = element.text.strip() if element.text else ""
        
        # Handle page breaks
        if element_type in self.PAGE_BREAK_TYPES:
            return element_type, "", True
        
        # Skip empty elements
        if not text:
            return element_type, "", False
        
        # Process based on element type
        try:
            if element_type in self.TITLE_TYPES:
                content = self.processor.process_title(text)
            elif element_type in self.TEXT_TYPES:
                content = self.processor.process_text(text)
            elif element_type in self.LIST_TYPES:
                content = self.processor.process_list_item(text)
            elif element_type in self.TABLE_TYPES:
                content = self.processor.process_table(element)
            else:
                # Handle unknown element types as plain text
                content = self.processor.process_text(text)
                logger.debug(f"Unknown element type '{element_type}' processed as text")
            
            return element_type, content, False
            
        except Exception as e:
            logger.error(f"Error processing element of type '{element_type}': {e}")
            return element_type, text, False  # Return raw text as fallback


class PageFormatter:
    """Handles formatting of page content with proper spacing and structure."""
    
    def __init__(self):
        self.spacing_rules = {
            "Title": {"after": True},
            "Header": {"after": True},
            "Table": {"before": True, "after": True},
            "ListItem": {"group_spacing": True},  # Special handling for list groups
            "Text": {"after": True},
            "NarrativeText": {"after": True}
        }
    
    def format_page_content(self, content: List[Tuple[str, str]]) -> str:
        """
        Format page content with proper spacing and structure.
        
        Args:
            content: List of (element_type, content) tuples
            
        Returns:
            Formatted Markdown content for the page
        """
        if not content:
            return ""
        
        formatted_lines = []
        
        for i, (element_type, element_content) in enumerate(content):
            # Add the content
            formatted_lines.append(element_content)
            
            # Determine if we need spacing after this element
            needs_spacing = self._needs_spacing_after(
                element_type, content, i
            )
            
            if needs_spacing:
                formatted_lines.append("")
        
        # Clean up excessive blank lines
        result = "\n".join(formatted_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 consecutive newlines
        
        return result.strip()
    
    def _needs_spacing_after(
        self, 
        current_type: str, 
        content: List[Tuple[str, str]], 
        current_index: int
    ) -> bool:
        """
        Determine if spacing is needed after the current element.
        
        Args:
            current_type: Type of current element
            content: Full content list
            current_index: Index of current element
            
        Returns:
            True if spacing should be added
        """
        # Don't add spacing after the last element
        if current_index >= len(content) - 1:
            return False
        
        next_type = content[current_index + 1][0]
        
        # Check spacing rules
        rules = self.spacing_rules.get(current_type, {})
        
        if rules.get("after"):
            return True
        
        # Special handling for list groups
        if rules.get("group_spacing"):
            return next_type not in self.spacing_rules.get("ListItem", {}).get("group_types", ["ListItem"])
        
        # Add spacing before tables
        if next_type in ["Table"]:
            return True
        
        return False


class WordDocumentExtractor:
    """
    Production-ready extractor for converting Word documents to Markdown format.
    
    Supports both .doc and .docx files with intelligent element processing,
    proper Markdown formatting, and comprehensive error handling.
    """
    
    SUPPORTED_EXTENSIONS = {'.doc', '.docx'}
    
    def __init__(self, infer_table_structure: bool = True):
        """
        Initialize the Word document extractor.
        
        Args:
            infer_table_structure: Whether to infer table structure during parsing
        """
        self.infer_table_structure = infer_table_structure
        self.element_processor = ElementProcessor()
        self.page_formatter = PageFormatter()
        
        logger.info(f"Initialized WordDocumentExtractor with table structure inference: {infer_table_structure}")
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """
        Validate and normalize the Word document file path.
        
        Args:
            file_path: Path to the Word document
            
        Returns:
            Normalized Path object
            
        Raises:
            WordExtractionError: If path is invalid or file doesn't exist
        """
        if not file_path:
            raise WordExtractionError("File path cannot be empty")
        
        path = Path(file_path)
        
        if not path.exists():
            raise WordExtractionError(f"Word document not found: {path}")
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise WordExtractionError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        return path
    
    def _partition_document(self, file: bytes) -> List[Any]:
        """
        Partition the document into elements using the appropriate unstructured method.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document elements
            
        Raises:
            WordExtractionError: If partitioning fails
        """
        try:
            return partition_docx(
                    file=file,
                    infer_table_structure=self.infer_table_structure
                )
                
        except Exception as e:
            raise WordExtractionError(f"Failed to partition document: {str(e)}") from e
    
    def _process_elements_to_pages(self, elements: List[Any]) -> List[Dict[str, Any]]:
        """
        Process document elements and organize them into pages.
        
        Args:
            elements: List of unstructured elements
            
        Returns:
            List of WordPageResult objects
        """
        pages = []
        current_page_content = []
        current_page_index = 0
        element_count = 0
        
        try:
            for element in elements:
                element_type, content, is_page_break = self.element_processor.process_element(element)
                element_count += 1
                
                # Handle page breaks
                if is_page_break:
                    # Finalize current page
                    if current_page_content:
                        page_text = self.page_formatter.format_page_content(current_page_content)
                        pages.append({
                            "page_index": current_page_index,
                            "text": page_text,
                            "status": bool(page_text.strip())
                            }
                        )

                    # Start new page
                    current_page_index += 1
                    current_page_content = []
                    continue
                
                # Add content to current page
                if content:
                    current_page_content.append((element_type, content))
            
            # Finalize the last page
            if current_page_content:
                page_text = self.page_formatter.format_page_content(current_page_content)
                pages.append({
                    "page_index": current_page_index,
                    "text": page_text,
                    "status": bool(page_text.strip())
                    }
                )
    
            # If no pages were created, create a single page with all content
            if not pages and element_count > 0:
                combined_content = []
                for element in elements:
                    element_type, content, _ = self.element_processor.process_element(element)
                    if content:
                        combined_content.append((element_type, content))
                
                if combined_content:
                    page_text = self.page_formatter.format_page_content(combined_content)
                    pages.append({
                        "page_index": 0,
                        "text": page_text,
                        "status": True
                        }
                    )
            
            return pages
            
        except Exception as e:
            logger.error(f"Error processing elements to pages: {e}")
            # Return a single failed page
            return [{
                    "page_index": 0,
                    "text": "",
                    "status": False
                    }
                ]
    
    async def extract_async(self, file: bytes, filename: str, executor: ProcessPoolExecutor) -> Dict[str, Any]:
        """
        Asynchronously extract content from a Word document.
        
        Args:
            file_path: Path to the Word document
            
        Returns:
            WordExtractionResult containing all extracted content
            
        Raises:
            WordExtractionError: If extraction fails
        """
        
        logger.info(f"Starting extraction for {filename}")
        
        try:
            # Partition the document (run in thread to avoid blocking)
            loop = asyncio.get_event_loop()
            elements = await loop.run_in_executor(
                executor, self._partition_document, file
            )
            
            logger.info(f"Document partitioned into {len(elements)} elements")
            
            # Process elements into pages
            pages = await loop.run_in_executor(
                executor, self._process_elements_to_pages, elements
            )
            
            # Calculate statistics
            success_count = sum(1 for page in pages if page["status"])
            failed_count = len(pages) - success_count
            
            logger.info(
                f"Extraction completed for {filename}: "
                f"{success_count}/{len(pages)} pages successful, "
            )
            
            return {"filename": filename,
                    "total_pages": len(pages),
                    "pages": pages,
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "file_type": "docx"
                    }
            
        except Exception as e:
            if isinstance(e, WordExtractionError):
                raise
            raise WordExtractionError(f"Failed to extract Word document content: {str(e)}") from e