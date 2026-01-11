"""
Script to extract all tables from an RHP PDF and save them for manual review.

Usage:
    python scripts/extract_tables_to_file.py <pdf_path> [output_dir]

Example:
    python scripts/extract_tables_to_file.py input/RHP_test.pdf data/output/tables
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rhp_analyzer.ingestion.table_extractor import TableExtractor


def extract_tables_to_files(pdf_path: str, output_dir: str = None):
    """
    Extract all tables from a PDF and save them in multiple formats.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted tables (default: data/output/tables)
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Default output directory
    if output_dir is None:
        output_dir = Path("data/output/tables")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting tables from: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Initialize extractor
    extractor = TableExtractor()
    
    # Extract tables
    tables = extractor.extract_tables(str(pdf_path))
    
    if not tables:
        print("No tables found in the PDF.")
        return
    
    print(f"Found {len(tables)} tables")
    print("-" * 60)
    
    # Save all tables to a single markdown file for easy viewing
    markdown_path = output_dir / "all_tables.md"
    with open(markdown_path, "w", encoding="utf-8") as md_file:
        md_file.write(f"# Extracted Tables from {pdf_path.name}\n\n")
        md_file.write(f"Total tables found: {len(tables)}\n\n")
        md_file.write("---\n\n")
        
        for i, table in enumerate(tables, 1):
            print(f"Processing Table {i}: Page {table.page_num}, Type: {table.table_type or 'Unknown'}")
            
            # Write to markdown
            md_file.write(f"## Table {i}\n\n")
            md_file.write(f"- **Page**: {table.page_num}\n")
            md_file.write(f"- **Type**: {table.table_type or 'Unknown'}\n")
            md_file.write(f"- **Confidence**: {table.confidence:.2%}\n")
            md_file.write(f"- **Headers**: {table.headers}\n")
            md_file.write(f"- **Rows**: {len(table.rows)}\n\n")
            
            # Create markdown table
            if table.headers:
                md_file.write("| " + " | ".join(str(h) for h in table.headers) + " |\n")
                md_file.write("| " + " | ".join("---" for _ in table.headers) + " |\n")
            
            for row in table.rows:
                # Clean up cell values
                cleaned_row = []
                for cell in row:
                    cell_str = str(cell) if cell else ""
                    # Replace newlines and pipes for markdown compatibility
                    cell_str = cell_str.replace("\n", " ").replace("|", "\\|")
                    cleaned_row.append(cell_str)
                md_file.write("| " + " | ".join(cleaned_row) + " |\n")
            
            md_file.write("\n---\n\n")
            
            # Also save each table as a separate CSV file
            csv_path = output_dir / f"table_{i:03d}_page_{table.page_num}.csv"
            save_table_as_csv(table, csv_path)
    
    print("-" * 60)
    print(f"Saved all tables to: {markdown_path}")
    print(f"Individual CSV files saved in: {output_dir}")
    
    # Also create a summary text file
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Table Extraction Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Source PDF: {pdf_path}\n")
        f.write(f"Total Tables: {len(tables)}\n\n")
        f.write(f"Tables by Page:\n")
        
        page_tables = {}
        for table in tables:
            if table.page_num not in page_tables:
                page_tables[table.page_num] = []
            page_tables[table.page_num].append(table)
        
        for page_num in sorted(page_tables.keys()):
            f.write(f"\n  Page {page_num}: {len(page_tables[page_num])} table(s)\n")
            for t in page_tables[page_num]:
                f.write(f"    - Type: {t.table_type or 'Unknown'}, Rows: {len(t.rows)}, Confidence: {t.confidence:.2%}\n")
    
    print(f"Summary saved to: {summary_path}")


def save_table_as_csv(table, csv_path: Path):
    """Save a single table as a CSV file."""
    import csv
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write headers if present
        if table.headers:
            writer.writerow(table.headers)
        
        # Write data rows
        for row in table.rows:
            writer.writerow(row)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide a PDF path")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_tables_to_files(pdf_path, output_dir)


if __name__ == "__main__":
    main()
