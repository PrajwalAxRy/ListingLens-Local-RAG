# Sample Outputs

This directory contains sample outputs from the RHP Analyzer ingestion pipeline.

## Directory Structure

After processing an RHP document, outputs are organized as follows:

```
data/output/{document_id}/
├── metadata.json           # Document metadata and processing info
├── pages/
│   ├── page_001.txt        # Extracted text per page
│   ├── page_002.txt
│   └── ...
├── tables/
│   ├── table_001.json      # Extracted tables with classification
│   ├── table_002.json
│   └── ...
├── sections/
│   └── section_tree.json   # Hierarchical section structure
├── entities/
│   └── entities.json       # Extracted named entities
├── financials/
│   └── financial_data.json # Parsed financial statements
└── checkpoint.json         # Pipeline checkpoint (for resume)
```

## Sample Metadata Structure

```json
{
  "document_id": "company_rhp_20250103_143022",
  "original_filename": "company_rhp.pdf",
  "total_pages": 425,
  "processing_start": "2025-01-03T14:30:22",
  "processing_end": "2025-01-03T14:42:15",
  "processing_duration_seconds": 713,
  "stages_completed": [
    "PDF_VALIDATED",
    "PAGES_EXTRACTED",
    "TABLES_EXTRACTED",
    "SECTIONS_MAPPED",
    "ENTITIES_EXTRACTED",
    "COMPLETED"
  ],
  "statistics": {
    "total_pages": 425,
    "text_pages": 398,
    "scanned_pages": 27,
    "tables_extracted": 156,
    "sections_identified": 42,
    "entities_extracted": 1245
  }
}
```

## Sample Section Tree Structure

```json
{
  "sections": [
    {
      "section_id": "sec_001",
      "title": "RISK FACTORS",
      "level": 1,
      "start_page": 15,
      "end_page": 45,
      "word_count": 18500,
      "subsections": [
        {
          "section_id": "sec_001_001",
          "title": "Risks Related to Our Business",
          "level": 2,
          "start_page": 16,
          "end_page": 28
        }
      ]
    },
    {
      "section_id": "sec_002",
      "title": "OUR BUSINESS",
      "level": 1,
      "start_page": 46,
      "end_page": 95
    }
  ]
}
```

## Sample Financial Data Structure

```json
{
  "fiscal_years": ["FY22", "FY23", "FY24"],
  "income_statement": {
    "revenue": [1250.5, 1580.2, 1945.8],
    "ebitda": [185.2, 245.6, 312.4],
    "pat": [98.5, 142.3, 195.7]
  },
  "balance_sheet": {
    "total_assets": [850.2, 1120.5, 1450.8],
    "total_equity": [425.1, 567.3, 762.0],
    "total_debt": [225.4, 298.7, 345.2]
  },
  "ratios": {
    "roe": [23.2, 25.1, 25.7],
    "roce": [28.5, 30.2, 31.8],
    "debt_equity": [0.53, 0.53, 0.45]
  },
  "forensic_flags": [
    {
      "type": "RECEIVABLES_GROWTH_VS_REVENUE",
      "severity": "MINOR",
      "description": "Receivables grew 22% vs Revenue 23%",
      "fiscal_year": "FY24"
    }
  ]
}
```

## Sample Entities Structure

```json
{
  "companies": [
    {
      "name": "ABC Technologies Limited",
      "type": "ISSUER",
      "mentions": 342
    },
    {
      "name": "XYZ Holdings Private Limited",
      "type": "SUBSIDIARY",
      "mentions": 28
    }
  ],
  "people": [
    {
      "name": "John Doe",
      "role": "PROMOTER",
      "mentions": 45
    },
    {
      "name": "Jane Smith",
      "role": "MD_CEO",
      "mentions": 38
    }
  ],
  "monetary_amounts": [
    {
      "value": 50000000000,
      "display": "₹5,000 Cr",
      "context": "issue_size",
      "page": 12
    }
  ]
}
```

## Performance Benchmarks

Expected processing times for different RHP sizes:

| Document Size | Estimated Time | Memory Usage |
|---------------|----------------|--------------|
| 100 pages | 5-8 minutes | ~500 MB |
| 300 pages | 15-25 minutes | ~1.2 GB |
| 500 pages | 30-45 minutes | ~2.0 GB |

**Note**: Times assume digital PDFs. Scanned PDFs with OCR will take 2-3x longer.

## Running the Pipeline

```bash
# Basic ingestion
rhp-analyzer analyze path/to/rhp.pdf

# With custom output directory
rhp-analyzer analyze rhp.pdf --output-dir ./my_output

# Dry run (validation only)
rhp-analyzer analyze rhp.pdf --dry-run

# Verbose logging
rhp-analyzer --verbose analyze rhp.pdf
```

## Checkpoint and Resume

If processing is interrupted, the pipeline can resume from the last checkpoint:

```bash
# Resume will automatically detect and continue from checkpoint
rhp-analyzer analyze path/to/rhp.pdf
```

Checkpoints are saved in `data/output/{document_id}/checkpoint.json`.
