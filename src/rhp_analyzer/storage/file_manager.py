"""File Manager Module for RHP Analyzer.

This module provides organized file structure management for documents and outputs.
It handles storage path conventions, directory creation, and document ID generation.

Path Structure:
    data/
    ├── input/{doc_id}/original.pdf
    ├── processed/{doc_id}/
    │   ├── pages/
    │   ├── tables/
    │   ├── sections/
    │   └── entities/
    ├── embeddings/{doc_id}/
    ├── qdrant/
    ├── checkpoints/
    └── output/{doc_id}/
        ├── report.md
        ├── report.pdf
        └── metadata.json
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


class FileManager:
    """
    Manages file storage paths and directory structure for RHP documents.
    
    This class provides:
    - Storage path conventions following the project structure
    - Helper functions for path resolution
    - Automatic directory creation
    - Document ID generation with sanitization
    
    Attributes:
        data_dir: Base data directory path
        input_dir: Directory for input RHP PDFs
        output_dir: Directory for generated reports
        processed_dir: Directory for processed document data
        embeddings_dir: Directory for embedding files
        qdrant_dir: Directory for Qdrant vector storage
        checkpoints_dir: Directory for pipeline checkpoints
    
    Example:
        >>> file_mgr = FileManager(data_dir=Path("./data"))
        >>> doc_id = file_mgr.generate_document_id("company_rhp.pdf")
        >>> input_path = file_mgr.get_input_path(doc_id)
        >>> file_mgr.ensure_document_directories(doc_id)
    """

    # Subdirectory names within processed/{doc_id}/
    PROCESSED_SUBDIRS = ["pages", "tables", "sections", "entities"]
    
    # Standard filenames
    ORIGINAL_PDF = "original.pdf"
    REPORT_MD = "report.md"
    REPORT_PDF = "report.pdf"
    METADATA_JSON = "metadata.json"
    CHUNKS_JSONL = "chunks.jsonl"
    EMBEDDINGS_NPY = "embeddings.npy"
    EMBEDDINGS_META_JSON = "metadata.json"
    CHECKPOINT_JSON = "checkpoint.json"

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the FileManager with directory paths.
        
        Args:
            data_dir: Base data directory. Defaults to "./data".
            input_dir: Directory for input PDFs. Defaults to "{data_dir}/input".
            output_dir: Directory for output reports. Defaults to "{data_dir}/output".
        """
        # Resolve base data directory
        self.data_dir = Path(data_dir).resolve() if data_dir else Path("./data").resolve()
        
        # Set primary directories (can be overridden)
        self.input_dir = Path(input_dir).resolve() if input_dir else self.data_dir / "input"
        self.output_dir = Path(output_dir).resolve() if output_dir else self.data_dir / "output"
        
        # Set derived directories
        self.processed_dir = self.data_dir / "processed"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.qdrant_dir = self.data_dir / "qdrant"
        self.checkpoints_dir = self.data_dir / "checkpoints"
        
        logger.debug(f"FileManager initialized with data_dir={self.data_dir}")

    @classmethod
    def from_config(cls, config) -> "FileManager":
        """
        Create a FileManager from application configuration.
        
        Args:
            config: RHPAnalyzerConfig or similar configuration object.
            
        Returns:
            FileManager instance configured from the application config.
        """
        return cls(
            data_dir=getattr(config.paths, "data_dir", None),
            input_dir=getattr(config.paths, "input_dir", None),
            output_dir=getattr(config.paths, "output_dir", None),
        )

    # =========================================================================
    # Document ID Generation
    # =========================================================================

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a filename to be safe for all filesystems.
        
        Removes or replaces characters that are invalid on Windows, Linux, or macOS.
        
        Args:
            filename: The original filename (may include extension).
            
        Returns:
            Sanitized filename safe for filesystem use.
            
        Example:
            >>> FileManager.sanitize_filename("Company: IPO (2024).pdf")
            'Company_IPO_2024'
        """
        # Remove extension if present
        name = Path(filename).stem
        
        # Replace invalid characters with underscore
        # Invalid on Windows: < > : " / \ | ? *
        # Also replace spaces and other problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*\s\-]+', '_', name)
        
        # Remove any non-alphanumeric characters except underscore
        sanitized = re.sub(r'[^\w]', '_', sanitized)
        
        # Collapse multiple underscores into one
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure non-empty result
        if not sanitized:
            sanitized = "document"
        
        # Limit length to reasonable filesystem limit (100 chars for name part)
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        return sanitized

    @staticmethod
    def generate_timestamp() -> str:
        """
        Generate a timestamp string for document ID.
        
        Returns:
            Timestamp in format YYYYMMDD_HHMMSS.
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_document_id(
        self,
        filename: str,
        custom_id: Optional[str] = None,
    ) -> str:
        """
        Generate a unique document ID from a filename.
        
        Format: {sanitized_filename}_{YYYYMMDD_HHMMSS}
        
        Args:
            filename: Original PDF filename.
            custom_id: Optional custom ID to use instead of generated one.
            
        Returns:
            Unique document ID safe for filesystem use.
            
        Example:
            >>> file_mgr = FileManager()
            >>> doc_id = file_mgr.generate_document_id("Company RHP.pdf")
            >>> # Returns something like: "Company_RHP_20260111_143022"
        """
        if custom_id:
            # Sanitize custom ID too
            return self.sanitize_filename(custom_id)
        
        sanitized_name = self.sanitize_filename(filename)
        timestamp = self.generate_timestamp()
        
        document_id = f"{sanitized_name}_{timestamp}"
        logger.debug(f"Generated document ID: {document_id} from filename: {filename}")
        
        return document_id

    def is_valid_document_id(self, document_id: str) -> bool:
        """
        Check if a document ID is valid.
        
        Args:
            document_id: The document ID to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        if not document_id:
            return False
        
        # Check for invalid filesystem characters
        if re.search(r'[<>:"/\\|?*]', document_id):
            return False
        
        # Check reasonable length
        if len(document_id) > 200:
            return False
        
        return True

    # =========================================================================
    # Path Resolution
    # =========================================================================

    def get_input_path(self, document_id: str) -> Path:
        """
        Get the input directory path for a document.
        
        Args:
            document_id: The document's unique identifier.
            
        Returns:
            Path to the document's input directory.
        """
        return self.input_dir / document_id

    def get_input_pdf_path(self, document_id: str) -> Path:
        """
        Get the path to the original PDF file.
        
        Args:
            document_id: The document's unique identifier.
            
        Returns:
            Path to the original PDF file.
        """
        return self.get_input_path(document_id) / self.ORIGINAL_PDF

    def get_processed_path(self, document_id: str) -> Path:
        """
        Get the processed data directory path for a document.
        
        Args:
            document_id: The document's unique identifier.
            
        Returns:
            Path to the document's processed data directory.
        """
        return self.processed_dir / document_id

    def get_pages_path(self, document_id: str) -> Path:
        """Get the pages subdirectory path."""
        return self.get_processed_path(document_id) / "pages"

    def get_tables_path(self, document_id: str) -> Path:
        """Get the tables subdirectory path."""
        return self.get_processed_path(document_id) / "tables"

    def get_sections_path(self, document_id: str) -> Path:
        """Get the sections subdirectory path."""
        return self.get_processed_path(document_id) / "sections"

    def get_entities_path(self, document_id: str) -> Path:
        """Get the entities subdirectory path."""
        return self.get_processed_path(document_id) / "entities"

    def get_embeddings_path(self, document_id: str) -> Path:
        """
        Get the embeddings directory path for a document.
        
        Args:
            document_id: The document's unique identifier.
            
        Returns:
            Path to the document's embeddings directory.
        """
        return self.embeddings_dir / document_id

    def get_chunks_file_path(self, document_id: str) -> Path:
        """Get the path to the chunks JSONL file."""
        return self.get_embeddings_path(document_id) / self.CHUNKS_JSONL

    def get_embeddings_file_path(self, document_id: str) -> Path:
        """Get the path to the embeddings numpy file."""
        return self.get_embeddings_path(document_id) / self.EMBEDDINGS_NPY

    def get_embeddings_metadata_path(self, document_id: str) -> Path:
        """Get the path to the embeddings metadata JSON file."""
        return self.get_embeddings_path(document_id) / self.EMBEDDINGS_META_JSON

    def get_output_path(self, document_id: str) -> Path:
        """
        Get the output directory path for a document.
        
        Args:
            document_id: The document's unique identifier.
            
        Returns:
            Path to the document's output directory.
        """
        return self.output_dir / document_id

    def get_report_md_path(self, document_id: str) -> Path:
        """Get the path to the markdown report."""
        return self.get_output_path(document_id) / self.REPORT_MD

    def get_report_pdf_path(self, document_id: str) -> Path:
        """Get the path to the PDF report."""
        return self.get_output_path(document_id) / self.REPORT_PDF

    def get_output_metadata_path(self, document_id: str) -> Path:
        """Get the path to the output metadata JSON file."""
        return self.get_output_path(document_id) / self.METADATA_JSON

    def get_checkpoint_path(self, document_id: str) -> Path:
        """
        Get the checkpoint file path for a document.
        
        Args:
            document_id: The document's unique identifier.
            
        Returns:
            Path to the document's checkpoint file.
        """
        return self.checkpoints_dir / f"{document_id}_{self.CHECKPOINT_JSON}"

    def get_qdrant_path(self) -> Path:
        """Get the Qdrant storage directory path."""
        return self.qdrant_dir

    # =========================================================================
    # Directory Management
    # =========================================================================

    def ensure_directory(self, path: Path) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: The directory path to ensure exists.
            
        Returns:
            The path that was created/verified.
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_base_directories(self) -> None:
        """
        Ensure all base directories exist.
        
        Creates the main directory structure:
        - data/input/
        - data/output/
        - data/processed/
        - data/embeddings/
        - data/qdrant/
        - data/checkpoints/
        """
        directories = [
            self.input_dir,
            self.output_dir,
            self.processed_dir,
            self.embeddings_dir,
            self.qdrant_dir,
            self.checkpoints_dir,
        ]
        
        for directory in directories:
            self.ensure_directory(directory)
            logger.debug(f"Ensured directory exists: {directory}")

    def ensure_document_directories(self, document_id: str) -> dict[str, Path]:
        """
        Ensure all directories exist for a specific document.
        
        Creates the complete directory structure for document processing:
        - input/{doc_id}/
        - processed/{doc_id}/pages/
        - processed/{doc_id}/tables/
        - processed/{doc_id}/sections/
        - processed/{doc_id}/entities/
        - embeddings/{doc_id}/
        - output/{doc_id}/
        
        Args:
            document_id: The document's unique identifier.
            
        Returns:
            Dictionary mapping directory type to path.
        """
        paths = {
            "input": self.ensure_directory(self.get_input_path(document_id)),
            "processed": self.ensure_directory(self.get_processed_path(document_id)),
            "pages": self.ensure_directory(self.get_pages_path(document_id)),
            "tables": self.ensure_directory(self.get_tables_path(document_id)),
            "sections": self.ensure_directory(self.get_sections_path(document_id)),
            "entities": self.ensure_directory(self.get_entities_path(document_id)),
            "embeddings": self.ensure_directory(self.get_embeddings_path(document_id)),
            "output": self.ensure_directory(self.get_output_path(document_id)),
        }
        
        logger.info(f"Created directory structure for document: {document_id}")
        return paths

    # =========================================================================
    # Document Operations
    # =========================================================================

    def copy_pdf_to_input(self, source_pdf: Path, document_id: str) -> Path:
        """
        Copy a PDF file to the input directory structure.
        
        Args:
            source_pdf: Path to the source PDF file.
            document_id: The document's unique identifier.
            
        Returns:
            Path to the copied PDF file.
            
        Raises:
            FileNotFoundError: If source PDF doesn't exist.
            ValueError: If source is not a PDF file.
        """
        source_pdf = Path(source_pdf)
        
        if not source_pdf.exists():
            raise FileNotFoundError(f"Source PDF not found: {source_pdf}")
        
        if source_pdf.suffix.lower() != ".pdf":
            raise ValueError(f"Source file is not a PDF: {source_pdf}")
        
        # Ensure input directory exists
        self.ensure_directory(self.get_input_path(document_id))
        
        # Copy to standardized location
        dest_path = self.get_input_pdf_path(document_id)
        shutil.copy2(source_pdf, dest_path)
        
        logger.info(f"Copied PDF to: {dest_path}")
        return dest_path

    def document_exists(self, document_id: str) -> bool:
        """
        Check if a document has been processed.
        
        Args:
            document_id: The document's unique identifier.
            
        Returns:
            True if the document's input directory exists.
        """
        return self.get_input_path(document_id).exists()

    def get_processed_documents(self) -> list[str]:
        """
        Get a list of all processed document IDs.
        
        Returns:
            List of document IDs that have been processed.
        """
        documents = []
        
        if self.processed_dir.exists():
            for item in self.processed_dir.iterdir():
                if item.is_dir():
                    documents.append(item.name)
        
        return sorted(documents)

    def get_output_documents(self) -> list[str]:
        """
        Get a list of all documents with generated outputs.
        
        Returns:
            List of document IDs that have output reports.
        """
        documents = []
        
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir() and (item / self.REPORT_MD).exists():
                    documents.append(item.name)
        
        return sorted(documents)

    def cleanup_document(self, document_id: str, keep_output: bool = True) -> None:
        """
        Clean up intermediate files for a document.
        
        Args:
            document_id: The document's unique identifier.
            keep_output: If True, preserve the output directory.
        """
        directories_to_remove = [
            self.get_input_path(document_id),
            self.get_processed_path(document_id),
            self.get_embeddings_path(document_id),
        ]
        
        if not keep_output:
            directories_to_remove.append(self.get_output_path(document_id))
        
        # Also clean up checkpoint
        checkpoint_path = self.get_checkpoint_path(document_id)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.debug(f"Removed checkpoint: {checkpoint_path}")
        
        for directory in directories_to_remove:
            if directory.exists():
                shutil.rmtree(directory)
                logger.debug(f"Removed directory: {directory}")
        
        logger.info(f"Cleaned up document: {document_id} (keep_output={keep_output})")

    def get_storage_stats(self) -> dict:
        """
        Get storage statistics for the data directory.
        
        Returns:
            Dictionary with storage statistics.
        """
        def get_dir_size(path: Path) -> int:
            """Calculate total size of directory in bytes."""
            total = 0
            if path.exists():
                for item in path.rglob("*"):
                    if item.is_file():
                        total += item.stat().st_size
            return total
        
        stats = {
            "input_size_bytes": get_dir_size(self.input_dir),
            "processed_size_bytes": get_dir_size(self.processed_dir),
            "embeddings_size_bytes": get_dir_size(self.embeddings_dir),
            "qdrant_size_bytes": get_dir_size(self.qdrant_dir),
            "output_size_bytes": get_dir_size(self.output_dir),
            "total_documents": len(self.get_processed_documents()),
            "documents_with_output": len(self.get_output_documents()),
        }
        
        stats["total_size_bytes"] = sum([
            stats["input_size_bytes"],
            stats["processed_size_bytes"],
            stats["embeddings_size_bytes"],
            stats["qdrant_size_bytes"],
            stats["output_size_bytes"],
        ])
        
        # Convert to human-readable format
        def format_size(size_bytes: int) -> str:
            for unit in ["B", "KB", "MB", "GB"]:
                if size_bytes < 1024:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024
            return f"{size_bytes:.2f} TB"
        
        stats["total_size_human"] = format_size(stats["total_size_bytes"])
        
        return stats


# Convenience function for quick access
def create_file_manager(
    data_dir: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> FileManager:
    """
    Create a FileManager instance with optional custom paths.
    
    This is a convenience function for creating a FileManager with
    custom directory paths.
    
    Args:
        data_dir: Base data directory path.
        input_dir: Input directory path (overrides data_dir/input).
        output_dir: Output directory path (overrides data_dir/output).
        
    Returns:
        Configured FileManager instance.
    """
    return FileManager(
        data_dir=data_dir,
        input_dir=input_dir,
        output_dir=output_dir,
    )
