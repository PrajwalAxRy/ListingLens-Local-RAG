# RHP Analyzer - Implementation Plan

## Stack Recommendations & Complete Roadmap

---

## Table of Contents

1. Stack Recommendations
2. Executive Summary
3. Requirements
4. High-Level Architecture
5. Development Roadmap
   - Phase 1: Foundation & Infrastructure
   - Phase 2: Document Ingestion Pipeline
   - Phase 2.5: Financial Modeling & Valuation Prep
   - Phase 3: Storage & Indexing
   - Phase 4: Intelligence Tier - Agents
   - Phase 5: Report Generation
   - Phase 6: Production Polish
6. Project Checkpoints
7. Appendices

---

## 1. Stack Recommendations

### 1.1 Primary Stack (Recommended)

| Category | Technology | Rationale |
|----------|------------|-----------|
| **Runtime** | Python 3.10 | Stable, wide library support, matches your preference |
| **Package Management** | pip + venv | Simple, no extra tools, Windows-friendly |
| **CLI Framework** | Typer | Modern, auto-generates help, type hints, built on Click |
| **Configuration** | PyYAML + Pydantic | YAML parsing with validation and type safety |
| **Logging** | Loguru | Zero-config, daily rotation, beautiful output |
| **PDF Processing** | PyMuPDF (fitz) | Fast, reliable, handles digital PDFs excellently |
| **Table Extraction** | pdfplumber + camelot-py | Complementary strengths for financial tables |
| **NLP/NER** | spaCy + transformers | Industry standard, good accuracy |
| **Embeddings** | sentence-transformers | Local embedding generation, no API needed |
| **Vector Database** | Qdrant (embedded mode) | No Docker needed, file-based, excellent filtering |
| **SQL Database** | SQLite + SQLAlchemy | Zero setup, file-based, perfect for single-user |
| **LLM Access** | huggingface_hub InferenceClient | Direct API access, Pro subscription benefits |
| **Orchestration** | LangGraph | State machine for agent workflows |
| **Report Generation** | Jinja2 + markdown + WeasyPrint | Template-based, professional PDF output |
| **Testing** | pytest + pytest-cov | Comprehensive testing with coverage |
| **Code Quality** | Ruff + pre-commit | Fast linting, formatting, git hooks |

**Pros:**
- All components work well on Windows
- No Docker dependency
- Embedded databases (no services to manage)
- Well-documented, mature libraries

**Cons:**
- WeasyPrint has complex Windows installation (GTK dependency)
- Some libraries have large dependencies

**Complexity:** Medium

---

### 1.2 Alternative Stack (Backup)

| Category | Technology | When to Use |
|----------|------------|-------------|
| **Vector Database** | ChromaDB | If Qdrant causes issues |
| **PDF Processing** | pypdf + pdfplumber | If PyMuPDF installation fails |
| **PDF Generation** | markdown + md2pdf | If WeasyPrint GTK issues persist |
| **CLI Framework** | Click | If Typer causes issues |
| **Logging** | stdlib logging | If Loguru conflicts arise |

---

### 1.3 Lightweight Stack (Fastest Prototype)

| Category | Technology | Trade-off |
|----------|------------|-----------|
| **Vector Storage** | NumPy + pickle | No filtering, basic similarity |
| **Database** | JSON files | No queries, simple structure |
| **PDF Generation** | Markdown only | No PDF initially |
| **Agents** | Single-file scripts | Less modularity |

**Use when:** You need a working demo in < 1 week

---

## 2. Executive Summary

### 2.1 System Purpose

RHP Analyzer is a local, AI-powered CLI application that processes Indian IPO Red Herring Prospectus (RHP) documents and generates comprehensive investment analysis reports. It uses a multi-agent architecture with state-of-the-art LLMs via Hugging Face Inference API.

### 2.2 Key Goals

1. **Automated Ingestion**: Parse 300-450 page RHP PDFs, extract text, tables, and structure
2. **Intelligent Analysis**: Multi-agent system providing forensic, governance, and risk analysis
3. **Quality Assurance**: Self-critic agent validates findings, reduces hallucinations
4. **Professional Output**: Markdown and PDF reports suitable for investment decisions
5. **Robust Operations**: Comprehensive logging, error handling, and recovery

### 2.3 Success Metrics

| Metric | Target |
|--------|--------|
| Processing time (400-page RHP) | < 90 minutes |
| Text extraction accuracy | > 95% |
| Table extraction accuracy | > 85% |
| Agent analysis coverage | All 14 core agents |
| Test coverage | > 80% |

#### Processing Time Breakdown

Target time allocation for a 400-page RHP (< 90 minutes total):

| Phase | Allocation | Time Target | Description |
|-------|------------|-------------|-------------|
| **Ingestion** | 25% | ~22 minutes | PDF parsing, text extraction, table detection |
| **Embedding** | 20% | ~18 minutes | Chunking, vector generation, indexing |
| **Analysis** | 45% | ~40 minutes | Agent processing, LLM calls, RAG queries |
| **Reporting** | 10% | ~9 minutes | Report generation, PDF rendering |

---

## 3. Requirements

### 3.1 Functional Requirements

#### FR-1: CLI Interface
- Accept RHP PDF file path as input
- Support configuration via YAML file
- Provide progress feedback during processing
- Output report to specified directory

#### FR-2: Document Processing
- Extract text from all pages
- Identify and extract tables (especially financial)
- Map document sections (ToC-aware)
- Extract named entities (companies, people, amounts)

#### FR-3: Analysis Pipeline
- Semantic chunking with overlap
- Vector embedding generation
- Multi-agent analysis with RAG
- Self-critique and validation

#### FR-4: Report Generation
- Structured markdown output
- PDF conversion with styling
- Page references for claims
- Executive summary and detailed sections

### 3.2 Non-Functional Requirements

#### NFR-1: Logging
- Console output with color-coded levels
- Daily rotating log files in `/logs/YYYY-MM-DD.log`
- Structured logging with context (document ID, phase, agent)
- Error logs with full tracebacks

#### NFR-2: Configuration
- YAML-based configuration
- Environment variable overrides
- Sensible defaults for all settings
- Validation on startup

#### NFR-3: Error Handling
- Graceful degradation (continue with warnings)
- Checkpoint/resume capability for long runs
- Clear error messages with remediation hints

#### NFR-4: Performance
- Memory-efficient streaming for large PDFs
- Batch processing for embeddings
- API rate limiting for HuggingFace

#### NFR-5: Testability
- All core functions unit-testable
- Integration tests for pipelines
- Test fixtures with sample data

---

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI LAYER                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Typer CLI  │  Config Loader (YAML)  │  Progress Display    ││
│  └─────────────────────────────────────────────────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      LOGGING MODULE                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Console Sink│  │ File Sink    │  │ Context Management     │ │
│  │ (colored)   │  │ (daily rot.) │  │ (doc_id, phase, agent) │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    INGESTION TIER                                │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐ │
│  │PDF Parser  │ │Table       │ │Section     │ │Entity        │ │
│  │(PyMuPDF)   │ │Extractor   │ │Mapper      │ │Extractor     │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Specialized Extractors: Promoter, Pre-IPO, Order Book,    │ │
│  │ Debt Structure, Contingent Liability, Objects of Issue    │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  STORAGE & INDEXING                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────────┐ │
│  │ Qdrant (embed)  │ │ SQLite          │ │ File Storage      │ │
│  │ Vector search   │ │ Structured data │ │ Raw + processed   │ │
│  └─────────────────┘ └─────────────────┘ └───────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  ANALYTICS TIER (Phase 2.5)                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────────┐ │
│  │ Projection      │ │ Valuation       │ │ Governance        │ │
│  │ Engine          │ │ Normalization   │ │ Rulebook          │ │
│  │ (Base/Bull/     │ │ (Peer Multiples)│ │ (Pre-flagging)    │ │
│  │  Stress)        │ │                 │ │                   │ │
│  └─────────────────┘ └─────────────────┘ └───────────────────┘ │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────────┐ │
│  │ Working Capital │ │ Cash Flow       │ │ Float             │ │
│  │ Analyzer        │ │ Analyzer        │ │ Calculator        │ │
│  └─────────────────┘ └─────────────────┘ └───────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  INTELLIGENCE TIER                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  LangGraph Orchestrator                    │ │
│  │                                                            │ │
│  │  ANALYSIS AGENTS (Parallel where possible):                │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │ │
│  │  │Business  │ │Industry  │ │Management│ │Capital   │     │ │
│  │  │Analyst   │ │Analyst   │ │Agent     │ │Structure │     │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │ │
│  │  │Forensic  │ │Red Flag  │ │Governance│ │Legal     │     │ │
│  │  │Accountant│ │Agent     │ │Agent     │ │Agent     │     │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                  │ │
│  │  │Valuation │ │Utilizat- │ │Promoter  │                  │ │
│  │  │Agent     │ │ion Agent │ │DD Agent  │                  │ │
│  │  └──────────┘ └──────────┘ └──────────┘                  │ │
│  │                                                            │ │
│  │  VALIDATION & SYNTHESIS:                                   │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐          │ │
│  │  │Self-     │ │Summarizer│ │Investment        │          │ │
│  │  │Critic    │ │Agent     │ │Committee (FINAL) │          │ │
│  │  └──────────┘ └──────────┘ └──────────────────┘          │ │
│  │                                                            │ │
│  │  ON-DEMAND:                                                │ │
│  │  ┌──────────┐                                             │ │
│  │  │ Q&A Agent│ ←── Interactive question answering          │ │
│  │  └──────────┘                                             │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              LLM CONFIGURATION                             │ │
│  │  • Context Model: Qwen2.5 32B / Mixtral 8x22B            │ │
│  │  • Reasoning Model: Meta-Llama-3.3-70B-Instruct          │ │
│  │  • Summarizer: Llama-3.2-8B-Instruct                     │ │
│  │  • Embeddings: nomic-embed-text-v1.5 (768 dim)           │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                 REPORT GENERATION                                │
│  ┌────────────────┐ ┌────────────────┐ ┌─────────────────────┐ │
│  │Template Engine │ │Markdown Builder│ │PDF Generator        │ │
│  │(Jinja2)        │ │                │ │(WeasyPrint)         │ │
│  └────────────────┘ └────────────────┘ └─────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Investment Committee Memo with Scorecard & Verdict         │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Development Roadmap

---

## Phase 1: Foundation & Infrastructure (Weeks 1-2)

**Goal:** Set up project structure, development environment, logging, configuration, and basic CLI skeleton.

---

### Milestone 1.1: Project Setup & Environment

**Deliverables:** Working Python environment with all dev dependencies, pre-commit hooks configured

#### Subtask 1.1.1: Create Project Structure

**What to do:**
- Create root folder structure following Python best practices
- Set up `src/rhp_analyzer/` as the main package
- Create placeholder `__init__.py` files
- Add `.gitignore` for Python projects

**Folder structure to create:**
```
rhp-analyzer/
├── src/
│   └── rhp_analyzer/
│       ├── __init__.py
│       ├── cli/
│       ├── config/
│       ├── ingestion/
│       ├── storage/
│       ├── agents/
│       ├── reporting/
│       └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── logs/
├── data/
│   ├── input/
│   ├── processed/
│   └── output/
├── templates/
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── config.yaml
```

**Verification:**
- All folders exist
- `python -c "import src.rhp_analyzer"` works (after setup)

---

#### Subtask 1.1.2: Initialize Virtual Environment

**What to do:**
- Create venv using `python -m venv .venv`
- Create `requirements.txt` with core dependencies
- Create `requirements-dev.txt` with testing/linting tools
- Document activation steps for Windows

**Key dependencies to include:**
- Core: typer, pyyaml, pydantic, loguru
- Dev: pytest, pytest-cov, ruff, pre-commit

**Verification:**
- `.venv` folder exists
- `pip list` shows installed packages
- `python --version` confirms 3.12

---

#### Subtask 1.1.3: Configure Pre-commit Hooks

**What to do:**
- Create `.pre-commit-config.yaml`
- Configure Ruff for linting and formatting
- Add trailing whitespace and end-of-file fixers
- Install hooks with `pre-commit install`

**Hooks to configure:**
- ruff (lint + format)
- trailing-whitespace
- end-of-file-fixer
- check-yaml
- check-added-large-files

**Verification:**
- `pre-commit run --all-files` passes
- Git commits trigger hooks

---

#### Subtask 1.1.4: Create pyproject.toml

**What to do:**
- Define project metadata
- Configure Ruff settings
- Configure pytest settings
- Set up package build configuration

**Sections to include:**
- `[project]` - name, version, dependencies
- `[tool.ruff]` - line length, rules, ignore patterns
- `[tool.pytest.ini_options]` - test paths, markers

**Verification:**
- `pip install -e .` works
- `ruff check .` runs without config errors

---

### Milestone 1.2: Logging Infrastructure

**Deliverables:** Centralized logging module with console and file sinks, context management

#### Subtask 1.2.1: Create Logging Module

**What to do:**
- Create `src/rhp_analyzer/utils/log_setup.py`
- Configure Loguru with console sink (colored, INFO level)
- Configure file sink with daily rotation (`logs/YYYY-MM-DD.log`)
- Create setup function called at app startup

**Features to implement:**
- Color-coded log levels for console
- Daily rotating file logs with 30-day retention
- Log format: `{time} | {level} | {module}:{function}:{line} | {message}`
- Separate error log file (`logs/errors.log`)

**Verification:**
- Run test script that logs at all levels
- Check `logs/` folder for date-stamped file
- Verify console shows colored output

---

#### Subtask 1.2.2: Add Context Management

**What to do:**
- Create context manager for adding contextual info to logs
- Support fields: `document_id`, `phase`, `agent_name`
- Use Loguru's `bind()` and `contextualize()` features

**Usage pattern:**
```python
with log_context(document_id="DOC001", phase="ingestion"):
    logger.info("Processing started")
    # All logs in this block include context
```

**Verification:**
- Logs show context fields
- Context doesn't leak between operations

---

#### Subtask 1.2.3: Create Logging Tests

**What to do:**
- Create `tests/unit/test_logging.py`
- Test log file creation
- Test log rotation (mock time)
- Test context management

**Verification:**
- `pytest tests/unit/test_logging.py`
- Coverage > 90% for logging module

---

### Milestone 1.3: Configuration System

**Deliverables:** YAML-based configuration with validation, environment variable support

#### Subtask 1.3.1: Create Configuration Schema

**What to do:**
- Create `src/rhp_analyzer/config/schema.py`
- Define Pydantic models for all config sections
- Add validation rules and defaults
- Support nested configuration

**Config sections to define:**
- `paths`: input_dir, output_dir, logs_dir, data_dir
- `llm`: model_names, api_key (from env), timeouts
- `ingestion`: chunk_size, overlap, batch_size
- `agents`: enabled_agents, temperature, max_tokens
- `reporting`: output_formats, template_path

**Verification:**
- Invalid config raises clear validation errors
- Defaults work when config is minimal

---

#### Subtask 1.3.2: Create Configuration Loader

**What to do:**
- Create `src/rhp_analyzer/config/loader.py`
- Load from `config.yaml` in project root
- Override with environment variables (prefix: `RHP_`)
- Support CLI argument overrides

**Priority order:**
1. CLI arguments (highest)
2. Environment variables
3. YAML file
4. Defaults (lowest)

**Verification:**
- Create sample `config.yaml` with test values
- Override via env var, confirm precedence
- Missing required values raise helpful errors

---

#### Subtask 1.3.3: Create Sample Configuration File

**What to do:**
- Create `config.yaml` with documented defaults
- Create `config.example.yaml` for distribution
- Create `.env.example` for sensitive values

**Include comments explaining:**
- Each section's purpose
- Valid values/ranges
- Required vs optional fields

**Verification:**
- Fresh clone + copy example configs = working setup

---

#### Subtask 1.3.4: Create Configuration Tests

**What to do:**
- Create `tests/unit/test_config.py`
- Test loading from file
- Test environment variable override
- Test validation errors

**Verification:**
- `pytest tests/unit/test_config.py` passes
- Edge cases covered (missing file, invalid YAML)

---

### Milestone 1.4: CLI Skeleton

**Deliverables:** Basic Typer CLI with help, version, and placeholder commands

#### Subtask 1.4.1: Create CLI Entry Point

**What to do:**
- Create `src/rhp_analyzer/cli/main.py`
- Set up Typer app with metadata
- Add `--version` flag
- Add `--config` option for custom config path

**Commands to stub:**
- `analyze` - Main analysis command (placeholder)
- `validate` - Validate an RHP without full analysis
- `config` - Show current configuration

**Verification:**
- `python -m rhp_analyzer --help` shows commands
- `python -m rhp_analyzer --version` shows version

---

#### Subtask 1.4.2: Create Analyze Command Structure

**What to do:**
- Create `src/rhp_analyzer/cli/commands/analyze.py`
- Accept PDF path as argument
- Accept optional output directory
- Add `--dry-run` flag for testing

**Arguments/options:**
- `pdf_path`: Path (required, validated)
- `--output-dir`: Path (optional, defaults from config)
- `--dry-run`: bool (skip actual processing)
- `--verbose`: bool (increase log level)

**Verification:**
- Invalid PDF path shows clear error
- Dry run completes without processing

---

#### Subtask 1.4.3: Add Progress Display

**What to do:**
- Create `src/rhp_analyzer/utils/progress.py`
- Integrate Rich library for progress bars
- Create wrapper for phase/step progress
- Support both TTY and non-TTY output

**Progress to track:**
- Overall phases (6 phases)
- Current phase progress (pages, chunks, etc.)
- Time elapsed and ETA

**Verification:**
- Progress bar displays during mock operation
- Non-TTY mode shows text updates instead

---

#### Subtask 1.4.4: Create CLI Tests

**What to do:**
- Create `tests/unit/test_cli.py`
- Test help output
- Test version output
- Test argument validation

**Use Typer's testing utilities:**
- `CliRunner` for invoking commands
- Check exit codes and output

**Verification:**
- `pytest tests/unit/test_cli.py` passes
- All commands have help text

---

### Milestone 1.5: Phase 1 Checkpoint

**Deliverables:** Working foundation ready for Phase 2

#### Subtask 1.5.1: Integration Verification

**What to do:**
- Run full test suite: `pytest tests/`
- Check coverage: `pytest --cov=src/rhp_analyzer tests/`
- Verify logging works end-to-end
- Confirm config loading works

**Acceptance criteria:**
- [ ] All tests pass
- [ ] Coverage > 80% for Phase 1 code
- [ ] CLI responds to all commands
- [ ] Logs written to files correctly

---

#### Subtask 1.5.2: Documentation Update

**What to do:**
- Update `README.md` with setup instructions
- Document CLI usage
- Add development guide section

**README sections:**
- Installation steps
- Configuration guide
- Basic usage examples

---

#### Subtask 1.5.3: Git Checkpoint

**What to do:**
- Ensure all changes committed
- Create tag: `v0.1.0-foundation`
- Suggested commit messages for this phase

**Commit message examples:**
- `feat(cli): add typer-based CLI skeleton`
- `feat(logging): implement daily rotating file logs`
- `feat(config): add YAML configuration with validation`

---

## Phase 2: Document Ingestion Pipeline (Weeks 3-4)

**Goal:** Build robust PDF processing, text extraction, table extraction, and section mapping.

---

### Milestone 2.1: PDF Processing Core

**Deliverables:** PDF parser that extracts text from all pages with metadata

#### Subtask 2.1.1: Create PDF Processor Module

**What to do:**
- Create `src/rhp_analyzer/ingestion/pdf_processor.py`
- Use PyMuPDF (fitz) for text extraction
- Extract page-by-page text with metadata
- Handle embedded fonts and encodings

**Data to extract per page:**
- Page number
- Raw text content
- Character count / word count
- Has images flag
- Font information (for section detection)

**Verification:**
- Process sample RHP PDF
- All pages extracted
- Text is readable (not garbled)

---

#### Subtask 2.1.2: Add Page Analysis

**What to do:**
- Analyze page layout (single column, multi-column)
- Detect page types (text, table, cover, blank)
- Extract headers/footers for removal
- Identify page margins

**Page type detection:**
- Cover pages (first few pages with large fonts)
- Table of contents
- Text-heavy pages
- Table-heavy pages
- Appendix pages

**Verification:**
- Page types correctly identified for sample RHP
- Headers/footers detected

---

#### Subtask 2.1.3: Create PDF Validation

**What to do:**
- Validate PDF integrity before processing
- Check for encryption/password protection
- Detect if PDF is scanned (image-based)
- Estimate processing complexity

**Validation checks:**
- File exists and readable
- Valid PDF format
- Page count within limits
- Not encrypted
- Warn if scanned (but proceed for digital)

**Verification:**
- Corrupted PDF raises clear error
- Encrypted PDF detected and reported
- Scanned PDF shows warning

---

#### Subtask 2.1.4: Create PDF Processing Tests

**What to do:**
- Create `tests/unit/test_pdf_processor.py`
- Use small test PDFs in fixtures
- Test text extraction accuracy
- Test edge cases (empty pages, unicode)

**Test fixtures needed:**
- Small digital PDF (5-10 pages)
- PDF with tables
- PDF with unicode text
- PDF with images

**Verification:**
- `pytest tests/unit/test_pdf_processor.py` passes
- Known content extracted correctly

---

#### Subtask 2.1.5: OCR Integration for Scanned Pages

**What to do:**
- Install and configure Tesseract OCR engine
- Create `src/rhp_analyzer/ingestion/ocr_processor.py`
- Integrate pytesseract with PDF processor
- Configure language support (English + Hindi for Indian RHPs)
- Handle mixed documents (digital + scanned pages)

**Installation steps (Windows):**
- Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
- Add Tesseract to PATH
- Install Hindi language pack (`tesseract-ocr-hin`)
- Verify with `tesseract --version` and `tesseract --list-langs`

**Implementation:**
```python
import pytesseract
from pdf2image import convert_from_path

class OCRProcessor:
    """OCR for scanned/image-based PDF pages"""

    def __init__(self, languages: str = "eng+hin"):
        self.languages = languages
        pytesseract.pytesseract.tesseract_cmd = self._find_tesseract()

    def is_scanned_page(self, page) -> bool:
        """Detect if page is scanned (low text, high images)"""
        pass

    def extract_text_ocr(self, page_image) -> str:
        """Apply OCR to page image"""
        return pytesseract.image_to_string(
            page_image,
            lang=self.languages,
            config='--psm 6'  # Assume uniform block of text
        )

    def process_scanned_pdf(self, pdf_path: str, page_numbers: List[int]) -> Dict[int, str]:
        """Process specific scanned pages"""
        images = convert_from_path(pdf_path, first_page=min(page_numbers), last_page=max(page_numbers))
        results = {}
        for i, img in enumerate(images):
            page_num = page_numbers[i]
            results[page_num] = self.extract_text_ocr(img)
        return results
```

**Integration with PDFProcessor:**
- Auto-detect scanned pages during extraction
- Apply OCR only to scanned pages (performance optimization)
- Merge OCR text with digital text in page order
- Flag OCR-processed pages in metadata (lower confidence)

**Verification:**
- Tesseract installed and accessible
- Hindi text extracted correctly
- Mixed PDF (digital + scanned) processed seamlessly
- OCR pages flagged in output

---

### Milestone 2.2: Table Extraction

**Deliverables:** Extract structured tables, especially financial statements

#### Subtask 2.2.1: Create Table Extractor Module

**What to do:**
- Create `src/rhp_analyzer/ingestion/table_extractor.py`
- Use pdfplumber as primary table detector
- Use camelot as fallback for complex tables
- Return structured table data (list of dicts)

**Table detection strategy:**
1. Run pdfplumber table detection on each page
2. For low-confidence tables, try camelot
3. Merge results, deduplicate

**Verification:**
- Financial tables extracted from sample RHP
- Column headers correctly identified

---

#### Subtask 2.2.2: Add Table Classification

**What to do:**
- Classify tables by type (financial, shareholding, etc.)
- Use header text and structure patterns
- Add confidence scores
- Tag tables with metadata

**Table types to classify:**
- Income Statement / P&L
- Balance Sheet
- Cash Flow Statement
- Shareholding Pattern
- Object of Issue / Use of Proceeds
- Related Party Transactions
- Key Metrics
- Other / Unknown

**Verification:**
- Financial statements classified correctly
- Confidence scores reasonable

---

#### Subtask 2.2.3: Add Financial Table Parser

**What to do:**
- Create specialized parser for financial statements
- Handle Indian accounting formats (lakhs, crores)
- Extract year-over-year data
- Calculate derived metrics (growth rates)

**Special handling:**
- Currency conversion (lakhs → actual values)
- Negative numbers in parentheses
- Merged header rows
- Multi-year columns

**Verification:**
- Revenue figures extracted correctly
- YoY growth calculated accurately

---

#### Subtask 2.2.4: Create Table Extraction Tests

**What to do:**
- Create `tests/unit/test_table_extractor.py`
- Test with known table structures
- Test classification accuracy
- Test financial parsing

**Verification:**
- `pytest tests/unit/test_table_extractor.py` passes
- Edge cases handled (empty tables, malformed)

---

### Milestone 2.3: Section Mapping

**Deliverables:** Hierarchical document structure with section boundaries

#### Subtask 2.3.1: Create Section Mapper Module

**What to do:**
- Create `src/rhp_analyzer/ingestion/section_mapper.py`
- Analyze font sizes to detect headers
- Use regex for common RHP section patterns
- Build hierarchical tree structure

**Detection methods:**
1. Font size analysis (larger = higher level)
2. Bold/caps text patterns
3. Common section title regex
4. Table of contents parsing (if available)

**Verification:**
- Major sections identified (Risk Factors, Business, etc.)
- Section boundaries accurate (±2 pages)

---

#### Subtask 2.3.2: Define Standard RHP Sections

**What to do:**
- Create section taxonomy for Indian RHPs
- Define aliases for section names
- Map variations to standard names

**Standard sections:**
1. Summary / Definitions
2. Risk Factors
3. Introduction
4. The Issue
5. Capital Structure
6. Objects of the Issue
7. Basis for Issue Price
8. About Our Company (Business)
9. Industry Overview
10. Our Management
11. Financial Information
12. Legal and Other Information
13. Other Regulatory Disclosures
14. Main Provisions of the Articles

**Verification:**
- 80%+ sections mapped for sample RHP
- Aliases handled (e.g., "Risk Factors" vs "RISK FACTORS")

---

#### Subtask 2.3.3: Add Section Content Extraction

**What to do:**
- Extract full text content for each section
- Preserve section hierarchy
- Link sections to page ranges
- Store section metadata

**Metadata per section:**
- Section title
- Level (1, 2, 3...)
- Start page / End page
- Word count
- Has tables flag
- Subsections list

**Verification:**
- Section content matches document
- Page ranges accurate

---

#### Subtask 2.3.4: Create Section Mapping Tests

**What to do:**
- Create `tests/unit/test_section_mapper.py`
- Test with sample RHP structure
- Test regex patterns
- Test hierarchy building

**Verification:**
- `pytest tests/unit/test_section_mapper.py` passes
- Known sections detected

---

### Milestone 2.4: Entity Extraction

**Deliverables:** Extract key entities (companies, people, amounts)

#### Subtask 2.4.1: Create Entity Extractor Module

**What to do:**
- Create `src/rhp_analyzer/ingestion/entity_extractor.py`
- Use spaCy for base NER
- Add custom patterns for Indian entities
- Deduplicate and resolve coreferences

**Entity types:**
- COMPANY (issuer, subsidiaries, competitors)
- PERSON (promoters, directors, KMPs)
- MONEY (amounts in crores/lakhs)
- DATE (fiscal years, issue dates)
- LOCATION (offices, plants)
- ORG (regulators, underwriters)

**Verification:**
- Company name correctly extracted
- Promoter names identified
- Monetary amounts parsed

---

#### Subtask 2.4.2: Add Financial Entity Patterns

**What to do:**
- Create custom patterns for Indian financial terms
- Handle crores, lakhs, percentages
- Extract price bands, valuations
- Parse date formats (Indian style)

**Custom patterns:**
- `₹ XX crores` → structured amount
- `Rs. XX lakhs` → structured amount
- `FY 2023-24` → fiscal year
- `XX%` → percentage
- `DD/MM/YYYY` → date

**Verification:**
- `₹500 crores` extracted as 5,000,000,000
- FY references parsed correctly

---

#### Subtask 2.4.3: Create Entity Resolution

**What to do:**
- Deduplicate entities (same entity, different mentions)
- Resolve abbreviations to full names
- Link entities across document
- Build entity relationship graph

**Resolution rules:**
- "XYZ Ltd" = "XYZ Limited" = "XYZ"
- "Mr. John Doe" = "John Doe"
- First mention establishes canonical name

**Verification:**
- Single entity for company despite variations
- Aliases linked correctly

---

#### Subtask 2.4.4: Create Entity Extraction Tests

**What to do:**
- Create `tests/unit/test_entity_extractor.py`
- Test NER accuracy
- Test custom patterns
- Test entity resolution

**Verification:**
- `pytest tests/unit/test_entity_extractor.py` passes
- Known entities extracted

---

### Milestone 2.5: Ingestion Pipeline Integration

**Deliverables:** Unified ingestion pipeline combining all extractors

#### Subtask 2.5.1: Create Pipeline Orchestrator ✅ COMPLETE

**What to do:**
- Create `src/rhp_analyzer/ingestion/pipeline.py`
- Orchestrate PDF → Text → Tables → Sections → Entities
- Add checkpoints for resume capability
- Emit progress events

**Pipeline stages:**
1. Validate PDF
2. Extract all pages
3. Extract tables
4. Map sections
5. Extract entities
6. Save processed data

**Verification:**
- Full pipeline runs on sample RHP
- Intermediate results saved

**Completed:** Pipeline orchestrator implemented with PipelineStage enum, progress tracking, error handling, and all 5 stages.

---

#### Subtask 2.5.2: Add Checkpoint/Resume ✅ COMPLETE

**What to do:**
- Save state after each major stage
- Allow resume from last checkpoint
- Clear checkpoints on success

**Checkpoint data:**
- Completed stages
- Extracted data per stage
- Timestamp and duration

**Verification:**
- Kill process mid-run, resume works
- Completed stages not re-run

**Completed:** Checkpoint system with PipelineCheckpoint dataclass, JSON persistence, stage-aware resume, and automatic cleanup on success.

---

#### Subtask 2.5.3: Create Integration Tests ✅ COMPLETE

**What to do:**
- Create `tests/integration/test_ingestion_pipeline.py`
- Test full pipeline on sample PDF
- Verify all outputs generated
- Check data consistency

**Verification:**
- `pytest tests/integration/` passes
- Output files exist and valid

**Completed:** 13 integration tests covering full pipeline run, stage execution, checkpoint/resume, error handling, result serialization, and progress callbacks.

---

### Milestone 2.5A: Specialized RHP Extractors

**Deliverables:** Domain-specific extractors for IPO-relevant data

#### Subtask 2.5A.0: Create Financial Parser Module

**What to do:**
- Create `src/rhp_analyzer/ingestion/financial_parser.py`
- Parse extracted financial tables into structured data
- Calculate key financial ratios automatically
- Detect window dressing and divergence signals
- Identify trends and anomalies in time series

**Separation from Table Extractor:**
- Table Extractor: Raw extraction of table data from PDF
- Financial Parser: Semantic understanding and computation on financial tables

**Metrics to extract and calculate:**
- Revenue, EBITDA, PAT (3-5 years)
- Total Assets, Equity, Debt
- ROE, ROCE, Debt/Equity
- Current Ratio, Quick Ratio
- Working Capital metrics
- Growth rates (CAGR for revenue, PAT, EBITDA)

**New-Age Metrics (for loss-making/startup IPOs):**
- Contribution Margin: `(Revenue - Variable Costs) / Revenue`
- CAC/LTV Ratio: Customer Acquisition Cost vs Lifetime Value
- Burn Rate: Monthly cash consumption rate
- Runway: Months of cash remaining at current burn
- Unit Economics: Revenue per user/order metrics
- GMV to Revenue conversion (for marketplace businesses)

```python
# New Age Metrics calculation
def calc_new_age_metrics(self, financials: FinancialMetrics) -> Dict[str, float]:
    """Calculate metrics for loss-making/startup IPOs"""
    return {
        'contribution_margin': self.calc_contribution_margin(financials),
        'cac_ltv': self.calc_cac_ltv(financials),
        'burn_rate': self.calc_burn_rate(financials),
        'runway_months': self.calc_runway(financials),
        'revenue_per_user': self.calc_unit_economics(financials),
    }
```

**Window dressing detection:**
- Revenue Growth vs Trade Receivables Growth (channel stuffing signal)
- EBITDA vs CFO comparison (paper profits signal)
- Inventory Days trend (inventory piling)

**Implementation:**
```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class FinancialMetrics:
    fiscal_year: str
    revenue: float
    ebitda: float
    pat: float
    total_assets: float
    total_equity: float
    total_debt: float
    cfo: float
    trade_receivables: float
    inventory: float
    # Calculated ratios
    roe: float
    roce: float
    debt_equity: float
    ebitda_margin: float
    pat_margin: float
    receivable_days: float
    inventory_days: float

class FinancialParser:
    """
    Extracts and computes financial metrics from parsed tables
    """
    def __init__(self, table_extractor):
        self.table_extractor = table_extractor

    def parse_financial_statements(self, tables: List[Table]) -> List[FinancialMetrics]:
        """Parse P&L, Balance Sheet, Cash Flow into structured metrics"""
        pass

    def calculate_ratios(self, metrics: FinancialMetrics) -> Dict[str, float]:
        """Calculate key financial ratios"""
        return {
            'roe': self._calc_roe(metrics),
            'roce': self._calc_roce(metrics),
            'debt_equity': self._calc_debt_equity(metrics),
            'current_ratio': self._calc_current_ratio(metrics),
            'ebitda_margin': (metrics.ebitda / metrics.revenue * 100) if metrics.revenue else 0,
            'pat_margin': (metrics.pat / metrics.revenue * 100) if metrics.revenue else 0,
        }

    def detect_divergences(self, metrics_list: List[FinancialMetrics]) -> List[str]:
        """
        Identify 'Window Dressing' signals:
        1. Revenue growth vs Receivables growth (Channel Stuffing)
        2. EBITDA vs CFO (Paper profits)
        3. Inventory piling
        """
        warnings = []
        for i in range(1, len(metrics_list)):
            current = metrics_list[i]
            prior = metrics_list[i-1]

            # Channel stuffing check
            revenue_growth = (current.revenue / prior.revenue - 1) * 100 if prior.revenue else 0
            receivable_growth = (current.trade_receivables / prior.trade_receivables - 1) * 100 if prior.trade_receivables else 0

            if receivable_growth > revenue_growth + 10:
                warnings.append(f"{current.fiscal_year}: Receivables growing {receivable_growth:.1f}% vs Revenue {revenue_growth:.1f}% - CHANNEL STUFFING RISK")

            # Paper profits check
            if current.ebitda > 0 and current.cfo / current.ebitda < 0.5:
                warnings.append(f"{current.fiscal_year}: CFO/EBITDA = {current.cfo/current.ebitda:.1%} - PAPER PROFITS RISK")

        return warnings

    def detect_trends(self, metrics_list: List[FinancialMetrics]) -> Dict[str, str]:
        """Identify growth trends and anomalies"""
        if len(metrics_list) < 2:
            return {}

        # Calculate CAGRs
        years = len(metrics_list) - 1
        first, last = metrics_list[0], metrics_list[-1]

        revenue_cagr = ((last.revenue / first.revenue) ** (1/years) - 1) * 100 if first.revenue else 0
        pat_cagr = ((last.pat / first.pat) ** (1/years) - 1) * 100 if first.pat > 0 else 0

        return {
            'revenue_cagr': f"{revenue_cagr:.1f}%",
            'pat_cagr': f"{pat_cagr:.1f}%",
            'margin_trend': 'expanding' if last.ebitda_margin > first.ebitda_margin else 'contracting'
        }
```

**Verification:**
- Financial metrics extracted correctly from sample RHP
- Ratios calculated accurately (spot check against manual calculation)
- Divergence warnings triggered for known problematic financials
- CAGR calculations verified

---

#### Subtask 2.5A.1: Create Promoter Extractor Module

**What to do:**
- Create `src/rhp_analyzer/ingestion/promoter_extractor.py`
- Extract promoter profiles from "Our Promoters" section
- Parse other directorships and common pursuits
- Extract promoter-specific litigation
- Calculate skin-in-the-game metrics

**Data to extract:**
- Promoter names, DIN, age, qualification
- Years of experience in industry
- Other directorship count and details
- Group companies in same line of business
- Pre/post-IPO shareholding percentages
- Loans and guarantees from company
- Remuneration trends (last 3 years)
- Promoter-specific litigation (count, amounts)

**Implementation**:
```python
class PromoterExtractor:
    """Extract comprehensive promoter profiles"""
    def extract_promoters(self, state) -> List[PromoterDossier]:
        pass
```

**Verification:**
- All promoters identified
- Shareholding data accurate
- Litigation counts correct

---

#### Subtask 2.5A.2: Create Pre-IPO Investor Analyzer

**What to do:**
- Create `src/rhp_analyzer/ingestion/pre_ipo_analyzer.py`
- Parse "Capital Structure" → "History of Equity Share Capital"
- Extract entry prices and dates for all pre-IPO investors
- Parse OFS (Offer for Sale) participation
- Extract lock-in schedules
- Calculate implied returns at floor and cap prices
- Calculate IRR based on holding period

**Data to extract:**
- Investor name and category (PE/VC, Strategic, ESOP Trust)
- Entry date and price per share
- Number of shares held pre-IPO
- Shares being sold via OFS
- Lock-in period and expiry date
- Implied return multiple at floor/cap price
- Annualized IRR at floor/cap price

**Implementation**:
```python
class PreIPOInvestorAnalyzer:
    """Analyze pre-IPO investors and exit economics"""
    def analyze_investors(self, state, ipo_details) -> List[PreIPOInvestor]:
        pass
```

**Verification:**
- Returns calculated correctly (spot check against manual calc)
- Lock-in dates extracted
- OFS participation identified

---

#### Subtask 2.5A.3: Create Order Book Analyzer

**What to do:**
- Create `src/rhp_analyzer/ingestion/order_book_analyzer.py`
- Check if order book disclosure is applicable (B2B/EPC/Defense sectors)
- Extract total order book value and as-of date
- Parse top 5 orders concentration
- Extract executable-in-12-months amount
- Calculate order book to LTM revenue ratio
- Compare YoY order book growth

**Applicable sectors:**
- EPC, Defense, Infrastructure
- IT Services, Capital Goods
- Engineering, Construction

**Data to extract:**
- Total unexecuted order book
- Order book as-of date
- Top 5 orders value and concentration %
- Amount executable in next 12 months
- Government orders percentage
- Prior year order book (for YoY growth)

**Implementation**:
```python
class OrderBookAnalyzer:
    """Extract and analyze order book disclosure"""
    def analyze_order_book(self, state, sector) -> OrderBookAnalysis:
        pass
```

**Verification:**
- Order book extracted for applicable sectors
- Ratios calculated correctly
- Flags non-applicable sectors appropriately

---

#### Subtask 2.5A.4: Create Debt Structure Analyzer

**What to do:**
- Create `src/rhp_analyzer/ingestion/debt_analyzer.py`
- Parse "Indebtedness" section tables
- Extract secured vs unsecured debt
- Parse short-term vs long-term split
- Extract interest rates (weighted average, range)
- Build debt maturity waterfall (0-1yr, 1-3yr, 3-5yr, 5yr+)
- Extract financial covenants from material contracts
- Calculate post-IPO debt after repayment

**Data to extract:**
- Total debt (as of latest balance sheet date)
- Secured/unsecured split
- Short-term/long-term split
- Weighted average interest rate
- Highest and lowest interest rates
- Maturity profile by time buckets
- Covenant details (debt/equity limits, DSCR minimums)
- Debt repayment amount from IPO proceeds

**Implementation**:
```python
class DebtStructureAnalyzer:
    """Analyze comprehensive debt structure"""
    def analyze_debt(self, state, objects_analysis) -> DebtStructure:
        pass
```

**Verification:**
- Debt totals match financial statements
- Maturity buckets sum to total debt
- Post-IPO debt calculated correctly

---

#### Subtask 2.5A.5: Create Contingent Liability Categorizer

**What to do:**
- Create `src/rhp_analyzer/ingestion/contingent_liability_analyzer.py`
- Extract contingent liabilities table from notes
- Categorize by type (Tax, Customs, Legal, Bank Guarantees, etc.)
- Assign probability weights (High 75%, Medium 50%, Low 25%)
- Calculate risk-weighted contingent liability
- Express as % of post-IPO net worth
- Check if any amounts earmarked in Objects of Issue

**Categories:**
- Income Tax / GST / Sales Tax disputes
- Customs / Excise disputes
- Civil litigation
- Labor/employee claims
- Bank guarantees
- Environmental claims

**Data to extract:**
- Total contingent liabilities
- Breakdown by category and entity
- Probability assessment
- Risk-weighted amount
- % of post-IPO net worth
- Mapping to Objects of Issue

**Implementation**:
```python
class ContingentLiabilityCategorizer:
    """Categorize and risk-weight contingent liabilities"""
    def categorize_contingencies(self, state, post_ipo_networth) -> ContingentLiabilityAnalysis:
        pass
```

**Verification:**
- All contingent categories identified
- Risk weights applied
- % of net worth calculated

---

#### Subtask 2.5A.6: Create Objects of Issue Tracker

**What to do:**
- Create `src/rhp_analyzer/ingestion/objects_tracker.py`
- Parse "Objects of the Issue" section
- Extract each use category with amounts and percentages
- Identify deployment timeline (from prospectus tables)
- Check appraiser reports for capex items
- Flag any "General Corporate Purposes" without detail
- Cross-reference debt repayment with Indebtedness section

**Use categories to track:**
- Debt repayment (amount, lenders)
- Capital expenditure (items, timelines, capex breakup)
- Working capital augmentation
- General corporate purposes
- Offer expenses

**Data to extract:**
- Fresh issue amount and OFS amount
- Breakdown of fresh issue usage (with %)
- Deployment timeline for capex
- Land acquisition status (acquired/to be acquired)
- Regulatory approvals pending
- Funding gap after IPO

**Implementation**:
```python
class ObjectsOfIssueTracker:
    """Parse use of proceeds with deployment timeline"""
    def track_objects(self, state) -> ObjectsOfIssueAnalysis:
        pass
```

**Verification:**
- Uses sum to 100% of fresh issue
- Deployment timelines extracted
- Red flags identified (vague usage, delays)

---

#### Subtask 2.5A.7: Create Stub Period Analyzer

**What to do:**
- Create `src/rhp_analyzer/ingestion/stub_analyzer.py`
- Identify if RHP includes stub/interim period financials
- Extract stub period metrics (revenue, EBITDA, PAT, margins)
- Compare to prior comparable period (YoY)
- Flag significant deviations (>20% margin change)
- Check for seasonality explanations in RHP

**Data to extract:**
- Stub period date range (e.g., "6 months ended Sep 30, 2024")
- Stub period financials (revenue, EBITDA, PAT, margins)
- Prior period comparatives
- YoY growth rates
- Margin expansion/contraction
- Seasonality notes
- One-time items disclosed

**Implementation**:
```python
class StubPeriodAnalyzer:
    """Analyze interim/stub period financials"""
    def analyze_stub(self, state, historical_financials) -> StubPeriodAnalysis:
        pass
```

**Verification:**
- Stub period identified if present
- YoY comparisons accurate
- Red flags surfaced

---

#### Subtask 2.5A.8: Create Specialized Extractor Tests

**What to do:**
- Create `tests/unit/test_specialized_extractors.py`
- Test each extractor with sample data
- Verify calculations (returns, ratios, percentages)
- Test edge cases (missing data, unusual formats)

**Tests to include:**
- Promoter extraction from sample text
- Pre-IPO IRR calculation verification
- Order book ratio calculations
- Debt maturity bucketing
- Contingent liability risk weighting
- Objects of issue sum-to-100 check
- Stub period YoY calc

**Verification:**
- All extractor tests pass
- Known calculations match manual verification

---

### Milestone 2.6: Phase 2 Checkpoint

#### Subtask 2.6.1: End-to-End Ingestion Test

**What to do:**
- Run ingestion on real RHP PDF
- Verify all components work together
- Measure processing time
- Document any issues

**Acceptance criteria:**
- [ ] 300+ page RHP processes without error
- [ ] Text extraction > 95% accuracy (spot check)
- [ ] Tables extracted (financial statements found)
- [ ] Sections mapped (major sections identified)
- [ ] Entities extracted (company, promoters found)
- [ ] Processing time < 15 minutes

---

#### Subtask 2.6.2: Update README & Tests

**What to do:**
- Document ingestion commands
- Update test coverage report
- Create sample output documentation

---

#### Subtask 2.6.3: Git Checkpoint

**What to do:**
- Commit all changes
- Create tag: `v0.2.0-ingestion`

**Commit messages:**
- `feat(ingestion): add PDF text extraction with PyMuPDF`
- `feat(ingestion): add table extraction with pdfplumber`
- `feat(ingestion): add section mapping with font analysis`
- `feat(ingestion): add entity extraction with spaCy`

---

## Phase 2.5: Financial Modeling & Valuation Prep (Week 4.5)

**Goal:** Build scenario projections and valuation normalization before agent analysis to provide structured data for downstream agents.

---

### Milestone 2.5A: Historical Normalization

**Deliverables:** Normalized historical financials ready for projection and analysis

#### Subtask 2.5A.1: Create Historical Normalizer Module

**What to do:**
- Create `src/rhp_analyzer/analytics/normalizer.py`
- Align consolidated vs standalone statements
- Convert all figures to ₹ crore (from lakhs/actual values)
- Reconcile restatements and carve-outs
- Handle different fiscal year endings

**Normalization tasks:**
- Convert lakhs to crores (divide by 100)
- Align standalone and consolidated statements
- Flag material restatements
- Handle partial year data (stub periods)

**Verification:**
- All figures in consistent units (₹ crores)
- Restatements documented
- Consolidated vs standalone clearly marked

---

### Milestone 2.5B: Projection Engine

**Deliverables:** Scenario-based financial forecasts using only RHP data

#### Subtask 2.5B.1: Create Projection Engine Module

**What to do:**
- Create `src/rhp_analyzer/analytics/projection_engine.py`
- Build Base/Bull/Stress scenario models using only RHP data
- Apply rule-based growth drivers from management guidance
- Calculate post-issue metrics (diluted EPS, ROE, ROIC, coverage ratios)
- Run dilution math (pre-issue shares + fresh issue + OFS + ESOP)
- Surface sensitivities to price band (floor vs cap)

**Scenarios to build:**
- **Base Case**: Management guidance, stated capacity utilization, disclosed order book
- **Bull Case**: Optimistic assumptions (faster ramp-up, pricing power, order wins)
- **Stress Case**: Conservative (execution delays, margin compression, demand weakness)

**Post-issue metrics to calculate:**
- Diluted EPS (3-year forward)
- Post-issue ROE and ROIC
- Net Debt/EBITDA
- Interest coverage ratio
- Cash conversion cycle

**Implementation:**
```python
@dataclass
class ProjectionScenario:
    scenario_name: str  # Base, Bull, Stress
    fiscal_years: List[str]
    revenue: List[float]
    ebitda: List[float]
    pat: List[float]
    diluted_eps: List[float]
    roe: List[float]
    roic: List[float]
    assumptions: Dict[str, str]
    citations: List[str]

class ProjectionEngine:
    """Build scenario-based financial forecasts from RHP"""
    def build_scenarios(self, state, ipo_details) -> List[ProjectionScenario]:
        pass
```

**Verification:**
- All 3 scenarios generated
- Share dilution math correct
- Metrics calculated for floor and cap prices

---

### Milestone 2.5C: Valuation Normalization

**Deliverables:** Peer-normalized valuation analysis

#### Subtask 2.5C.1: Create Valuation Normalization Module

**What to do:**
- Create `src/rhp_analyzer/analytics/valuation_module.py`
- Extract all peers from "Basis for Issue Price" section
- Extract additional peers from "Industry Overview" section
- Normalize peer financials (fiscal year, accounting standard differences)
- Calculate consistent multiples (P/E, P/B, EV/EBITDA, PEG)
- Flag "missing peers" mentioned elsewhere but excluded from pricing justification
- Build valuation ladders at floor, cap, and median prices
- Calculate premium/discount vs peer median

**Peer analysis tasks:**
- Identify all disclosed peers
- Flag cherry-picked peers (excluded competitors)
- Adjust for fiscal year mismatches
- Handle Ind-AS vs IFRS differences
- Compare consolidated vs standalone

**Valuation metrics:**
- P/E ratio at floor, cap, and implied fair value
- P/B ratio
- EV/EBITDA
- PEG ratio (P/E to growth)
- Premium/discount vs peer median (%)

**Implementation:**
```python
@dataclass
class ValuationSnapshot:
    price_point: str  # Floor, Cap, Fair
    p_e_ratio: float
    p_b_ratio: float
    ev_ebitda: float
    peer_median_p_e: float
    premium_discount_vs_peers: float  # %
    missing_peers: List[str]
    citations: List[str]

class ValuationNormalization:
    """Normalize peer valuations and calculate fair value ranges"""
    def normalize_peers(self, state, ipo_details) -> List[ValuationSnapshot]:
        pass
```

**Verification:**
- All disclosed peers extracted
- Multiples calculated consistently
- Premium/discount accurate

---

### Milestone 2.5D: Governance Rulebook Service

**Deliverables:** Pre-flagged governance and forensic issues for downstream agents

#### Subtask 2.5D.1: Create Governance Rulebook Module

**What to do:**
- Create `src/rhp_analyzer/analytics/governance_rules.py`
- Define YAML-based rulebook with SEBI-aligned guardrails
- Implement rule engine that evaluates RHP inputs
- Emit structured findings (severity, rationale, RHP citation)
- Feed findings to Governance, Forensic, and Red Flag agents

**Rule families (configurable in YAML):**

**Shareholder Skin-in-the-Game:**
- Promoter post-issue <51% → Critical alert
- OFS > Fresh Issue → Major alert

**Pledge & Encumbrance:**
- Any pledge >0% → Major alert
- Pledge >25% → Critical alert (VETO)

**Related-Party Concentration:**
- RPT revenue >20% of total → Major alert
- RPT expenses >20% of total → Major alert

**Audit Quality:**
- Auditor resignations → Critical alert
- CARO/NCF qualifications → Major alert
- Modified opinions → Critical alert

**Litigation Materiality:**
- Total litigation >10% of post-issue net worth → Critical (VETO)

**Working Capital Stress:**
- Receivable days growth > Revenue CAGR by 10pp → Major alert

**Severity levels:**
- **Critical**: Potential veto flag
- **Major**: Significant concern
- **Minor**: Worth noting

**Implementation:**
```python
@dataclass
class RuleViolation:
    rule_id: str
    rule_family: str
    severity: str  # Critical, Major, Minor
    description: str
    actual_value: str
    threshold: str
    citation: str

class GovernanceRulebook:
    """SEBI-aligned rule engine for governance red flags"""
    def __init__(self, rules_yaml_path: str):
        self.rules = self._load_rules(rules_yaml_path)

    def evaluate_rules(self, state) -> List[RuleViolation]:
        pass
```

**Verification:**
- Rules loaded from YAML
- All rule types evaluated
- Findings include RHP citations

---

### Milestone 2.5E: Phase 2.5 Checkpoint

#### Subtask 2.5E.1: Financial Modeling Integration Test

**What to do:**
- Run all financial modeling modules on sample RHP
- Verify scenario projections generated
- Validate valuation calculations
- Check rulebook evaluations

**Acceptance criteria:**
- [ ] 3 projection scenarios generated (Base/Bull/Stress)
- [ ] Peer valuations normalized correctly
- [ ] Rulebook pre-flags governance issues
- [ ] All citations tracked
- [ ] Processing time < 10 minutes

---

#### Subtask 2.5E.2: Git Checkpoint

**What to do:**
- Commit all changes
- Create tag: `v0.2.5-financial-modeling`

**Commit messages:**
- `feat(analytics): add historical normalizer`
- `feat(analytics): add projection engine with Base/Bull/Stress scenarios`
- `feat(analytics): add valuation normalization with peer analysis`
- `feat(analytics): add governance rulebook service`

---

## Phase 3: Storage & Indexing (Weeks 5-6)

**Goal:** Implement vector database, SQL storage, and semantic chunking.

---

### Milestone 3.1: Semantic Chunking

**Deliverables:** Smart text chunking that preserves context and semantics

#### Subtask 3.1.1: Create Chunking Module

**What to do:**
- Create `src/rhp_analyzer/storage/chunker.py`
- Implement section-aware chunking
- Respect paragraph boundaries
- Add configurable chunk size and overlap

**Chunking strategy:**
1. First split by section boundaries
2. Then split by paragraphs within sections
3. Merge small paragraphs, split large ones
4. Target 500-1000 tokens per chunk
5. Add 100-token overlap

**Verification:**
- Chunks don't break mid-sentence
- Section metadata preserved
- Overlap correctly applied

---

#### Subtask 3.1.2: Add Chunk Metadata

**What to do:**
- Attach rich metadata to each chunk
- Include source location (section, page)
- Add chunk type (narrative, table, list)
- Calculate token counts

**Metadata per chunk:**
- chunk_id (unique)
- document_id
- section_name
- page_range (start, end)
- chunk_type (narrative, table, mixed)
- token_count
- char_count
- preceding_chunk_id
- following_chunk_id

**Verification:**
- Metadata accurate
- Chunk chain reconstructable

---

#### Subtask 3.1.3: Handle Tables Specially

**What to do:**
- Don't chunk tables - keep whole or summarize
- Create table summaries for embedding
- Store full table data separately
- Link chunks to tables

**Table handling:**
- Small tables (< 500 tokens): embed as-is
- Large tables: create structured summary
- Store original table in separate collection

**Verification:**
- Tables not broken
- Summaries are meaningful

---

#### Subtask 3.1.4: Create Chunking Tests

**What to do:**
- Create `tests/unit/test_chunker.py`
- Test chunk size compliance
- Test section boundary respect
- Test table handling

**Verification:**
- `pytest tests/unit/test_chunker.py` passes
- Edge cases covered

---

### Milestone 3.2: Embedding Generation

**Deliverables:** Generate vector embeddings for all chunks

#### Subtask 3.2.1: Create Embedding Module

**What to do:**
- Create `src/rhp_analyzer/storage/embeddings.py`
- Use sentence-transformers library
- Load `nomic-ai/nomic-embed-text-v1.5` model
- Generate embeddings in batches

**Configuration:**
- Model: nomic-embed-text-v1.5 (768 dim)
- Fallback: all-MiniLM-L6-v2 (384 dim)
- Batch size: 32 chunks
- Device: CPU (no GPU)

**Verification:**
- Embeddings generated for all chunks
- Dimensions correct
- Batch processing works

---

#### Subtask 3.2.2: Add Embedding Cache

**What to do:**
- Cache embeddings to disk (numpy format)
- Check cache before regenerating
- Invalidate cache on content change

**Cache structure:**
```
data/embeddings/{document_id}/
├── chunks.jsonl
├── embeddings.npy
└── metadata.json
```

**Verification:**
- Second run uses cache
- Modified content triggers regen

---

#### Subtask 3.2.3: Create Embedding Tests

**What to do:**
- Create `tests/unit/test_embeddings.py`
- Test embedding dimensions
- Test batch processing
- Test caching logic

**Verification:**
- `pytest tests/unit/test_embeddings.py` passes
- Similar texts have similar embeddings

---

### Milestone 3.3: Vector Database (Qdrant)

**Deliverables:** Set up Qdrant for semantic search

#### Subtask 3.3.1: Create Vector Store Module

**What to do:**
- Create `src/rhp_analyzer/storage/vector_store.py`
- Use Qdrant in embedded (local file) mode
- Create collection per document
- Configure vector parameters

**Qdrant setup:**
- Embedded mode (no server needed)
- Storage path: `data/qdrant/`
- Distance: Cosine similarity
- Vector size: 768 (matches embedding model)

**Verification:**
- Collection created successfully
- Data persists across restarts

---

#### Subtask 3.3.2: Implement CRUD Operations

**What to do:**
- Add chunks with embeddings and metadata
- Query by vector similarity
- Filter by metadata (section, page, type)
- Delete document collections

**Operations:**
- `add_chunks(doc_id, chunks, embeddings)`
- `search(query_embedding, top_k, filters)`
- `get_by_section(doc_id, section_name)`
- `delete_document(doc_id)`

**Verification:**
- Add and retrieve works
- Filtering narrows results correctly

---

#### Subtask 3.3.3: Add Search Utilities

**What to do:**
- Create high-level search function
- Accept text query (auto-embed)
- Return formatted results with context
- Support hybrid search (keyword + semantic)

**Search features:**
- Text query → embed → search
- Return top K with scores
- Include surrounding context
- Highlight matching portions

**Verification:**
- Relevant chunks returned for queries
- Scores correlate with relevance

---

#### Subtask 3.3.4: Create Vector Store Tests

**What to do:**
- Create `tests/unit/test_vector_store.py`
- Test collection creation
- Test add/search/delete
- Test filtering

**Verification:**
- `pytest tests/unit/test_vector_store.py` passes
- Search returns expected results

---

### Milestone 3.4: SQL Database (SQLite)

**Deliverables:** Structured storage for metadata and analysis results

#### Subtask 3.4.1: Create Database Schema

**What to do:**
- Create `src/rhp_analyzer/storage/database.py`
- Define SQLAlchemy models
- Create migration with Alembic
- Initialize database on first run

**Tables:**
- `documents` - RHP metadata
- `sections` - Section details
- `tables` - Extracted table data
- `entities` - Named entities
- `financial_data` - Parsed financials
- `agent_outputs` - Analysis results
- `chunks` - Chunk metadata (not content)

**Verification:**
- Database file created
- Tables have correct schema

---

#### Subtask 3.4.2: Implement Repository Pattern

**What to do:**
- Create repository classes for each entity
- Implement CRUD operations
- Add query methods for common access patterns
- Handle transactions

**Repositories:**
- `DocumentRepository`
- `SectionRepository`
- `FinancialDataRepository`
- `AgentOutputRepository`

**Verification:**
- CRUD operations work
- Queries return expected results

---

#### Subtask 3.4.3: Add Data Migration Support

**What to do:**
- Set up Alembic for migrations
- Create initial migration
- Document migration commands

**Migration workflow:**
- `alembic revision --autogenerate -m "message"`
- `alembic upgrade head`
- `alembic downgrade -1`

**Verification:**
- Fresh install runs migrations
- Schema changes tracked

---

#### Subtask 3.4.4: Create Database Tests

**What to do:**
- Create `tests/unit/test_database.py`
- Test CRUD operations
- Test relationships
- Use test database (temp file)

**Verification:**
- `pytest tests/unit/test_database.py` passes
- No test pollution between runs

---

### Milestone 3.5: File Storage Organization

**Deliverables:** Organized file structure for documents and outputs

#### Subtask 3.5.1: Create File Manager Module

**What to do:**
- Create `src/rhp_analyzer/storage/file_manager.py`
- Define storage path conventions
- Create helper functions for path resolution
- Handle directory creation

**Path structure:**
```
data/
├── input/{doc_id}/original.pdf
├── processed/{doc_id}/
│   ├── pages/
│   ├── tables/
│   ├── sections/
│   └── entities/
├── embeddings/{doc_id}/
├── qdrant/
└── output/{doc_id}/
    ├── report.md
    ├── report.pdf
    └── metadata.json
```

**Verification:**
- Paths resolved correctly
- Directories created as needed

---

#### Subtask 3.5.2: Add Document ID Generation

**What to do:**
- Generate unique document IDs
- Use combination of filename + timestamp
- Ensure filesystem-safe characters
- Support custom ID override

**ID format:** `{sanitized_filename}_{YYYYMMDD_HHMMSS}`

**Verification:**
- IDs are unique
- No invalid filesystem characters

---

### Milestone 3.5A: Financial Modeling & Analytical Services ✅ COMPLETED

**Status:** Completed - All 9 subtasks implemented and tested (62 tests passing)

**Deliverables:** Forward-looking financial models and specialized analytical modules

#### Subtask 3.5A.1: Create Projection Engine

**What to do:**
- Create `src/rhp_analyzer/analytics/projection_engine.py`
- Build Base/Bull/Stress scenario models using only RHP data
- Normalize historical financials (crore/lakh conversions, restatements)
- Apply rule-based growth drivers from management guidance
- Calculate post-issue metrics (diluted EPS, ROE, ROIC, coverage ratios)
- Run dilution math (pre-issue shares + fresh issue + OFS + ESOP)
- Surface sensitivities to price band (floor vs cap)

**Scenarios to build:**
- **Base Case**: Management guidance, stated capacity utilization
- **Bull Case**: Optimistic assumptions (faster ramp-up, pricing power)
- **Stress Case**: Conservative (delays, margin compression)

**Post-issue metrics:**
- Diluted EPS (3-year forward)
- Post-issue ROE and ROIC
- Net Debt/EBITDA
- Interest coverage ratio
- Cash conversion cycle

**Implementation:**
```python
class ProjectionEngine:
    """Build scenario-based financial forecasts from RHP"""
    def build_scenarios(self, state, ipo_details) -> List[ProjectionScenario]:
        pass
```

**Verification:**
- All 3 scenarios generated
- Share dilution math correct
- Metrics calculated for floor and cap prices

---

#### Subtask 3.5A.2: Create Valuation Normalization Module

**What to do:**
- Create `src/rhp_analyzer/analytics/valuation_module.py`
- Extract all peers from "Basis for Issue Price" section
- Extract additional peers from "Industry Overview" section
- Normalize peer financials (fiscal year, accounting standard differences)
- Calculate consistent multiples (P/E, P/B, EV/EBITDA, PEG)
- Flag "missing peers" mentioned elsewhere but excluded from pricing justification
- Build valuation ladders at floor, cap, and median prices
- Link promoter WACA, pre-IPO placements from Capital Structure
- Calculate premium/discount vs peer median

**Peer analysis:**
- Identified peers vs cherry-picked peers
- Fiscal year mismatches
- Ind-AS vs IFRS adjustments
- Consolidated vs standalone differences

**Valuation metrics:**
- P/E ratio at floor, cap, and implied fair value
- P/B ratio
- EV/EBITDA
- PEG ratio (P/E to growth)
- Premium/discount vs peer median (%)

**Implementation:**
```python
class ValuationNormalization:
    """Normalize peer valuations and calculate fair value ranges"""
    def normalize_peers(self, state, ipo_details) -> List[ValuationSnapshot]:
        pass
```

**Verification:**
- All disclosed peers extracted
- Multiples calculated consistently
- Premium/discount accurate

---

#### Subtask 3.5A.3: Create Governance Rulebook Service

**What to do:**
- Create `src/rhp_analyzer/analytics/governance_rules.py`
- Define YAML-based rulebook with SEBI-aligned guardrails
- Implement rule engine that evaluates RHP inputs
- Emit structured findings (severity, rationale, RHP citation)
- Feed findings to Governance, Forensic, and Red Flag agents

**Rule families (configurable):**
- **Shareholder Skin-in-the-Game**:
  - Promoter post-issue <51% → Critical alert
  - OFS > Fresh Issue → Major alert
- **Pledge & Encumbrance**:
  - Any pledge >0% → Major alert
  - Pledge >25% → Critical alert
- **Related-Party Concentration**:
  - RPT revenue >20% of total → Major alert
  - RPT expenses >20% of total → Major alert
- **Audit Quality**:
  - Auditor resignations → Critical alert
  - CARO/NCF qualifications → Major alert
  - Modified opinions → Critical alert
- **Litigation Materiality**:
  - Total litigation >10% of post-issue net worth → Critical
- **Working Capital Stress**:
  - Receivable days growth > Revenue CAGR by 10pp → Major alert

**Severity levels:**
- **Critical**: Potential veto flag
- **Major**: Significant concern
- **Minor**: Worth noting

**Implementation:**
```python
class GovernanceRulebook:
    """SEBI-aligned rule engine for governance red flags"""
    def __init__(self, rules_yaml_path: str):
        self.rules = self._load_rules(rules_yaml_path)

    def evaluate_rules(self, state) -> List[RuleViolation]:
        pass
```

**Verification:**
- Rules loaded from YAML
- All rule types evaluated
- Findings include RHP citations

---

#### Subtask 3.5A.4: Create Risk & Litigation Quantification Module

**What to do:**
- Create `src/rhp_analyzer/analytics/risk_quant.py`
- Parse "Outstanding Litigation" section
- Parse "Material Developments" section
- Parse "Contingent Liabilities" notes
- Aggregate litigation by entity (Company, Promoters, Directors, Subsidiaries)
- Count cases and sum amounts
- Calculate % of post-issue net worth
- Flag matters with hearing dates within 12 months
- Check if proceeds earmarked for settlement in Objects

**Litigation aggregation:**
- By entity type (Company/Promoters/Directors/Subsidiaries)
- By case type (Civil/Criminal/Tax/Regulatory)
- By amount (total claimed)
- By status (pending/resolved)

**Contingent liabilities:**
- By category (Tax, Customs, Legal, etc.)
- Probability buckets (High/Medium/Low)
- Risk-weighted amounts
- Mapping to business segments

**Implementation:**
```python
class RiskLitigationQuantifier:
    """Quantify and categorize all legal and contingent risks"""
    def quantify_risks(self, state, post_issue_networth) -> List[RiskExposure]:
        pass
```

**Verification:**
- All litigation tables parsed
- Amounts aggregated correctly
- % of net worth calculated

---

#### Subtask 3.5A.5: Create Citation & Audit Trail Manager

**What to do:**
- Create `src/rhp_analyzer/analytics/citation_manager.py`
- Enforce machine-readable citations for all claims
- Store citation records (claim_id, section, page, paragraph anchor)
- Provide validation hooks for Self-Critic agent
- Auto-generate footnotes for report

**Citation format:**
- Section name
- Page number(s)
- Paragraph or table identifier
- Text snippet (optional)

**Features:**
- Citation storage per document
- Lookup by claim ID
- Validation (check if citation exists)
- Report footnote generation

**Implementation:**
```python
class CitationManager:
    """Manage audit trail for all sourced claims"""
    def __init__(self):
        self.citations = {}

    def add_citation(self, claim_id: str, section: str, page: int, snippet: str):
        pass

    def validate_claim(self, claim_id: str) -> bool:
        pass

    def generate_footnotes(self) -> str:
        pass
```

**Verification:**
- Citations stored and retrieved correctly
- Validation rejects uncited claims
- Footnotes generated properly

---

#### Subtask 3.5A.6: Create Working Capital Analyzer with Sector Benchmarks

**What to do:**
- Create `src/rhp_analyzer/analytics/working_capital_analyzer.py`
- Calculate working capital cycle metrics (receivable days, inventory days, payable days, CCC)
- Compare to predefined sector benchmarks
- Flag significant deviations (>20% vs sector avg)
- Track trends over multiple years

**Sector benchmarks (predefined):**
- FMCG: CCC ~40 days
- Pharma: CCC ~180 days
- IT Services: CCC ~50 days
- Auto: CCC ~80 days
- Textiles: CCC ~120 days
- Capital Goods: CCC ~180 days
- Real Estate: CCC ~400 days
- Steel: CCC ~80 days
- Cement: CCC ~40 days
- Chemicals: CCC ~110 days

**Metrics to calculate:**
- Days Sales Outstanding (DSO)
- Days Inventory Outstanding (DIO)
- Days Payable Outstanding (DPO)
- Cash Conversion Cycle (CCC = DSO + DIO - DPO)
- Variance vs sector benchmark
- YoY trends

**Implementation:**
```python
class WorkingCapitalAnalyzer:
    """Analyze working capital with sector benchmarks"""
    SECTOR_BENCHMARKS = {...}

    def analyze(self, financials, sector) -> List[WorkingCapitalAnalysis]:
        pass
```

**Verification:**
- Cycle metrics calculated correctly
- Sector benchmarks applied
- Deviations flagged

---

#### Subtask 3.5A.7: Create Enhanced Cash Flow Analyzer

**What to do:**
- Create `src/rhp_analyzer/analytics/cashflow_analyzer.py`
- Calculate Free Cash Flow (FCF = CFO - Capex)
- Categorize capex (maintenance vs growth)
- Calculate FCF margin and FCF yield
- Identify cash-burning companies
- Calculate runway months for cash burners
- Assess cash flow quality (CFO/EBITDA, CFO/PAT ratios)

**Metrics to calculate:**
- Free Cash Flow (FCF)
- FCF margin (%)
- FCF yield vs market cap at floor/cap price
- Is cash burning? (FCF < 0)
- Monthly burn rate
- Cash runway (months)
- Capex intensity (capex/revenue %)
- Capex to depreciation ratio
- Maintenance capex estimate
- Growth capex estimate
- Working capital change impact

**Implementation:**
```python
class EnhancedCashFlowAnalyzer:
    """Enhanced cash flow analysis with FCF and burn rate"""
    def analyze(self, financials, ipo_details) -> List[CashFlowAnalysis]:
        pass
```

**Verification:**
- FCF calculated correctly
- Capex categorization reasonable
- Runway months accurate for burners

---

#### Subtask 3.5A.8: Create Float Calculator

**What to do:**
- Create `src/rhp_analyzer/analytics/float_calculator.py`
- Calculate tradeable float at listing (Day 1)
- Calculate float post anchor unlock (Day 90)
- Build lock-in expiry calendar
- Calculate retail quota shares
- Estimate implied daily trading volume

**Float calculations:**
- Total shares post-issue
- Promoter locked shares and %
- Anchor locked shares (90-day lock-in)
- Pre-IPO investor locked shares
- Day 1 free float (shares and %)
- Day 90 free float (shares and %)
- Retail quota shares
- Implied daily volume (Day 1 float / 250 trading days)

**Lock-in calendar:**
- Date, shares unlocking, investor name, % of float

**Implementation:**
```python
class FloatCalculator:
    """Calculate free float at different time horizons"""
    def calculate_float(self, ipo_details, investors, promoters) -> FloatAnalysis:
        pass
```

**Verification:**
- Float percentages sum correctly
- Lock-in calendar accurate
- Daily volume estimate reasonable

---

#### Subtask 3.5A.9: Create Analytical Services Tests

**What to do:**
- Create `tests/unit/test_analytics.py`
- Test each analytical module with known inputs
- Verify calculation accuracy
- Test edge cases

**Tests to include:**
- Projection engine scenario generation
- Valuation multiple calculations
- Governance rule evaluation
- Citation manager validation
- Working capital cycle math
- FCF and cash runway calculations
- Float percentage calculations

**Verification:**
- All analytics tests pass
- Calculations match manual verification

---

### Milestone 3.6: Phase 3 Checkpoint

#### Subtask 3.6.1: Storage Integration Test

**What to do:**
- Run ingestion → chunking → embedding → storage
- Verify all data stored correctly
- Test search functionality
- Measure storage size

**Acceptance criteria:**
- [ ] Chunks stored in Qdrant
- [ ] Metadata stored in SQLite
- [ ] Search returns relevant results
- [ ] Storage size reasonable (< 500MB per RHP)

---

#### Subtask 3.6.2: Git Checkpoint

**What to do:**
- Commit all changes
- Create tag: `v0.3.0-storage`

**Commit messages:**
- `feat(storage): add semantic chunking with section awareness`
- `feat(storage): add embedding generation with caching`
- `feat(storage): add Qdrant vector store integration`
- `feat(storage): add SQLite database with SQLAlchemy`
- `feat(analytics): add Projection Engine with Base/Bull/Stress scenarios`
- `feat(analytics): add Valuation Normalization Module`
- `feat(analytics): add Governance Rulebook Service`
- `feat(analytics): add Risk & Litigation Quantification`
- `feat(analytics): add Citation Manager for audit trails`
- `feat(analytics): add Working Capital Analyzer with sector benchmarks`
- `feat(analytics): add Enhanced Cash Flow Analyzer (FCF, burn rate)`
- `feat(analytics): add Float Calculator for liquidity analysis`

---

## Phase 4: Intelligence Tier - Agents (Weeks 7-9)

**Goal:** Build the multi-agent analysis system using LangGraph, with RAG-powered agents for different analysis domains.

---

### Milestone 4.1: LLM Integration Layer

**Deliverables:** Unified interface to Hugging Face Inference API with rate limiting and error handling

#### Subtask 4.1.1: Create LLM Client Module

**What to do:**
- Create `src/rhp_analyzer/agents/llm_client.py`
- Use `huggingface_hub.InferenceClient`
- Configure for multiple models (context, reasoning)
- Add retry logic with exponential backoff

**Models to configure:**
- Context Model: `Qwen/Qwen2.5-32B-Instruct` (large context)
- Reasoning Model: `meta-llama/Llama-3.3-70B-Instruct` (deep analysis)
- Fallback: `mistralai/Mixtral-8x22B-Instruct-v0.1`

**Features:**
- Automatic model selection based on task
- Token counting before requests
- Response streaming support
- Cost/usage tracking

**Verification:**
- Simple prompt returns response
- Retry works on transient failures
- Token limits respected

---

#### Subtask 4.1.2: Add Rate Limiting

**What to do:**
- Implement rate limiter for API calls
- Respect Hugging Face Pro limits
- Queue requests during high load
- Log rate limit events

**Rate limiting strategy:**
- Token bucket algorithm
- Per-model limits
- Automatic backoff on 429 errors
- Configurable limits in config.yaml

**Verification:**
- Burst requests don't fail
- 429 errors handled gracefully

---

#### Subtask 4.1.3: Create Prompt Templates

**What to do:**
- Create `src/rhp_analyzer/agents/prompts/` directory
- Define base prompt template structure
- Create system prompts for each agent role
- Support variable injection

**Template structure:**
```
prompts/
├── base.py          # Base template class
├── architect.py     # Architect agent prompts
├── forensic.py      # Forensic accountant prompts
├── red_flag.py      # Red flag agent prompts
├── governance.py    # Governance agent prompts
├── legal.py         # Legal agent prompts
└── critic.py        # Self-critic prompts
```

**Verification:**
- Templates render correctly
- Variables substituted properly

---

#### Subtask 4.1.4: Create LLM Client Tests

**What to do:**
- Create `tests/unit/test_llm_client.py`
- Mock API responses for testing
- Test retry logic
- Test rate limiting

**Verification:**
- `pytest tests/unit/test_llm_client.py` passes
- No actual API calls in tests (mocked)

---

#### Subtask 4.1.5: Local Summarizer Model Setup

**What to do:**
- Set up local LLM for fast, repetitive summarization tasks
- Install and configure Ollama (recommended for Windows)
- Pull `Llama-3.2-8B-Instruct` model locally
- Create `src/rhp_analyzer/agents/local_summarizer.py`
- Integrate with fallback to HuggingFace API if local fails

**Why Local Summarizer?**
- Reduce API costs for repetitive chunk-level summaries
- Faster response times (no network latency)
- No rate limiting concerns for batch operations
- Works offline after initial model download

**Installation Steps (Windows):**
```bash
# Option 1: Using Ollama (Recommended)
# Download from https://ollama.ai/download
# Install and run Ollama
ollama pull llama3.2:8b-instruct-fp16

# Verify installation
ollama list
ollama run llama3.2:8b-instruct-fp16 "Hello, test"

# Option 2: Using transformers (requires GPU with 8GB+ VRAM)
pip install transformers torch accelerate
# Model will be downloaded on first use
```

**Implementation:**
```python
import os
from typing import Optional

class LocalSummarizer:
    """
    Local LLM for fast summarization tasks.
    Uses Ollama by default, falls back to HuggingFace API.
    """
    def __init__(self, use_ollama: bool = True, fallback_to_api: bool = True):
        self.use_ollama = use_ollama
        self.fallback_to_api = fallback_to_api
        self.ollama_model = "llama3.2:8b-instruct-fp16"
        self.hf_model = "meta-llama/Llama-3.2-8B-Instruct"

        if use_ollama:
            self._init_ollama()
        else:
            self._init_transformers()

    def _init_ollama(self):
        """Initialize Ollama client"""
        try:
            import ollama
            self.client = ollama.Client()
            # Verify model is available
            self.client.show(self.ollama_model)
            self.is_available = True
        except Exception as e:
            print(f"Ollama not available: {e}. Will use fallback.")
            self.is_available = False

    def _init_transformers(self):
        """Initialize transformers pipeline (requires GPU)"""
        try:
            from transformers import pipeline
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = pipeline(
                "text-generation",
                model=self.hf_model,
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            self.is_available = True
        except Exception as e:
            print(f"Transformers not available: {e}. Will use API fallback.")
            self.is_available = False

    def summarize(self, text: str, max_words: int = 150) -> str:
        """
        Summarize text using local model.
        Falls back to API if local not available.
        """
        prompt = f"""Summarize the following text in {max_words} words or less.
Be concise and capture the key points.

Text:
{text}

Summary:"""

        if self.use_ollama and self.is_available:
            return self._summarize_ollama(prompt)
        elif hasattr(self, 'pipeline') and self.is_available:
            return self._summarize_transformers(prompt)
        elif self.fallback_to_api:
            return self._summarize_api(prompt)
        else:
            raise RuntimeError("No summarization backend available")

    def _summarize_ollama(self, prompt: str) -> str:
        """Use Ollama for summarization"""
        response = self.client.generate(
            model=self.ollama_model,
            prompt=prompt,
            options={
                "temperature": 0.3,
                "num_predict": 300,  # ~200 words
            }
        )
        return response['response'].strip()

    def _summarize_transformers(self, prompt: str) -> str:
        """Use local transformers for summarization"""
        result = self.pipeline(
            prompt,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True
        )
        return result[0]['generated_text'].split("Summary:")[-1].strip()

    def _summarize_api(self, prompt: str) -> str:
        """Fallback to HuggingFace API"""
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        response = client.text_generation(
            prompt,
            model=self.hf_model,
            max_new_tokens=300,
            temperature=0.3
        )
        return response.strip()

    def batch_summarize(self, texts: list, max_words: int = 150) -> list:
        """Summarize multiple texts efficiently"""
        return [self.summarize(text, max_words) for text in texts]
```

**Configuration in config.yaml:**
```yaml
local_summarizer:
  enabled: true
  backend: ollama  # or 'transformers'
  model: llama3.2:8b-instruct-fp16
  fallback_to_api: true
  max_words_default: 150
```

**Use Cases:**
- Chunk-level summaries during ingestion
- Section summaries for RAG context
- Quick entity extraction
- Table content summarization

**Verification:**
- Ollama installed and model pulled successfully
- `ollama run llama3.2:8b-instruct-fp16 "Summarize: Test text"` works
- LocalSummarizer class initializes without errors
- Summarization produces coherent output
- Fallback to API works when local unavailable
- Batch summarization completes within reasonable time

---

### Milestone 4.2: RAG (Retrieval-Augmented Generation) System

**Deliverables:** Context retrieval system that feeds relevant chunks to agents

#### Subtask 4.2.1: Create RAG Module

**What to do:**
- Create `src/rhp_analyzer/agents/rag.py`
- Implement query → retrieve → augment pipeline
- Support multi-query retrieval
- Add context window management

**RAG pipeline:**
1. Receive analysis question
2. Generate embedding for question
3. Retrieve top-K relevant chunks
4. Rerank by relevance (optional)
5. Format context for LLM
6. Track source citations

**Verification:**
- Relevant context retrieved
- Context fits within token limits

---

#### Subtask 4.2.2: Add Context Formatting

**What to do:**
- Format retrieved chunks for LLM consumption
- Include source metadata (page, section)
- Handle context window overflow
- Prioritize most relevant content

**Format structure:**
```
[Source: Section Name, Pages X-Y]
<chunk content>

[Source: Section Name, Pages A-B]
<chunk content>
...
```

**Verification:**
- Context well-formatted
- Sources traceable

---

#### Subtask 4.2.3: Add Citation Tracking

**What to do:**
- Track which chunks contributed to answers
- Generate source citations for claims
- Map claims to page numbers
- Support citation verification

**Citation format:**
- `[Page 45]` for single page
- `[Pages 45-47]` for range
- `[Risk Factors, Page 67]` with section

**Verification:**
- Citations accurate
- Claims traceable to sources

---

#### Subtask 4.2.4: Create RAG Tests

**What to do:**
- Create `tests/unit/test_rag.py`
- Test retrieval accuracy
- Test context formatting
- Test citation generation

**Verification:**
- `pytest tests/unit/test_rag.py` passes
- Known queries return expected context

---

### Milestone 4.3: Base Agent Framework

**Deliverables:** Reusable agent base class with common functionality

#### Subtask 4.3.1: Create Base Agent Class

**What to do:**
- Create `src/rhp_analyzer/agents/base.py`
- Define abstract base agent interface
- Implement common methods (retrieve, analyze, format)
- Add logging and error handling

**Base agent interface:**
- `analyze(document_id, state) → AgentOutput`
- `retrieve_context(query) → List[Chunk]`
- `generate_response(prompt, context) → str`
- `validate_output(output) → bool`

**Verification:**
- Base class provides shared functionality
- Subclasses can override as needed

---

#### Subtask 4.3.2: Define Agent Output Schema

**What to do:**
- Create structured output format for all agents
- Use Pydantic models for validation
- Include confidence scores
- Support partial results

**Output schema:**
```python
class AgentOutput:
    agent_name: str
    analysis_type: str
    findings: List[Finding]
    confidence: float
    citations: List[Citation]
    warnings: List[str]
    processing_time: float
```

**Verification:**
- Outputs validate correctly
- Schema handles all agent types

---

#### Subtask 4.3.3: Add Agent Registry

**What to do:**
- Create agent registration system
- Allow enabling/disabling agents via config
- Support agent dependencies
- Manage agent execution order

**Registry features:**
- Register agents by name
- Check dependencies before execution
- Provide agent instances on demand
- Log agent lifecycle events

**Verification:**
- Agents registered correctly
- Disabled agents not executed

---

### Milestone 4.4: Core Analysis Agents

**Deliverables:** Implement the six core analysis agents

#### Subtask 4.4.1: Architect Agent

**What to do:**
- Create `src/rhp_analyzer/agents/architect.py`
- Orchestrates overall analysis flow and acts as the "conductor" of the multi-agent system
- Creates analysis plan tailored to the specific RHP
- Summarizes key document characteristics for downstream agents
- Runs FIRST before all other analysis agents

**Responsibilities:**
- Analyze document structure and completeness
- Identify key sections and assess their quality
- Create a prioritized roadmap for other agents
- Generate a document profile with key characteristics
- Determine which specialized agents are relevant (e.g., Order Book for B2B companies)
- Flag document anomalies that need special attention

**Analysis areas:**
1. **Document Completeness Check:**
   - Verify all mandatory RHP sections present
   - Flag missing or incomplete sections
   - Check for proper restated financial statements

2. **Section Quality Assessment:**
   - Rate each section's disclosure quality (Excellent/Good/Poor)
   - Identify sections with unusually brief disclosures
   - Flag boilerplate vs substantive content

3. **Complexity Rating:**
   - Business complexity (simple/moderate/complex)
   - Corporate structure complexity (single entity vs group)
   - Transaction complexity (fresh issue vs OFS-heavy)

4. **Key Metrics Identification:**
   - Extract issue size, price band, market cap
   - Identify company sector and relevant benchmarks
   - Extract promoter holding % (pre and post)
   - Flag if SME or Mainboard IPO

5. **Agent Relevance Mapping:**
   - Enable/disable Order Book Agent (only for B2B/EPC/Defense)
   - Adjust forensic focus based on sector
   - Prioritize agents based on document signals

**System Prompt:**
```python
ARCHITECT_PROMPT = """
You are the Chief Analyst orchestrating the analysis of an IPO prospectus.
Your job is to understand the document structure and create an analysis plan.

Analyze:
1. Document Structure: Which sections are present? Any missing?
2. Company Profile: Sector, size, complexity
3. Key Metrics: Extract issue size, price band, promoter holding
4. Analysis Priorities: Which areas need deepest scrutiny?
5. Agent Routing: Which specialized agents should be activated?

Output a structured analysis plan that guides other agents.

Document Metadata:
{metadata}

Section Summary:
{section_summary}

Your analysis plan:
"""
```

**Implementation:**
```python
class ArchitectAgent(BaseAgent):
    """
    Orchestrates overall analysis flow
    Runs first to create analysis roadmap
    """
    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        # 1. Analyze document structure
        section_coverage = self._check_section_completeness(state)

        # 2. Determine company profile
        company_profile = self._extract_company_profile(state)

        # 3. Create analysis plan
        analysis_plan = self._create_analysis_plan(section_coverage, company_profile)

        # 4. Determine agent relevance
        agent_config = self._determine_agent_relevance(company_profile)

        # Store in state for downstream agents
        state['analysis_plan'] = analysis_plan
        state['agent_config'] = agent_config
        state['company_profile'] = company_profile

        return self._create_analysis_output(analysis_plan)

    def _check_section_completeness(self, state) -> Dict[str, str]:
        """Check presence and quality of mandatory sections"""
        mandatory_sections = [
            'Risk Factors', 'Business Overview', 'Financial Statements',
            'Management', 'Capital Structure', 'Objects of Issue',
            'Basis for Issue Price', 'Legal Proceedings'
        ]
        # Implementation
        pass

    def _extract_company_profile(self, state) -> Dict:
        """Extract key company characteristics"""
        pass

    def _determine_agent_relevance(self, profile: Dict) -> Dict[str, bool]:
        """Determine which agents should be activated"""
        return {
            'order_book_agent': profile.get('sector') in ['EPC', 'Defense', 'Infrastructure', 'IT Services'],
            'stub_period_agent': profile.get('has_stub_period', False),
            # ... other agents
        }
```

**Output:**
- Document completeness report
- Company profile summary
- Prioritized analysis plan
- Agent activation configuration
- Complexity rating (1-5 scale)

**Verification:**
- Document profile generated with key metrics
- Analysis plan coherent and prioritized
- Agent relevance correctly determined based on sector
- Complexity rating aligns with document characteristics

---

#### Subtask 4.4.2: Forensic Accountant Agent

**What to do:**
- Create `src/rhp_analyzer/agents/forensic.py`
- Deep analysis of financial statements
- Calculate and verify financial ratios
- Identify accounting anomalies

**Analysis areas:**
- Revenue recognition patterns
- Working capital analysis
- Cash flow vs profit comparison
- Related party transaction review
- Debt structure analysis
- Auditor qualifications review

**Output:**
- Financial health score
- Key ratios with industry comparison
- Anomalies with severity ratings
- Recommendations

**Verification:**
- Financial analysis comprehensive
- Calculations verifiable

---

#### Subtask 4.4.3: Red Flag Agent

**What to do:**
- Create `src/rhp_analyzer/agents/red_flag.py`
- Identify warning signs and risk indicators
- Cross-reference multiple document sections
- Assign severity to each red flag

**Red flags to detect:**
- Frequent auditor changes
- Related party loans
- Pledged promoter shares
- Litigation exposure
- Customer/supplier concentration
- Negative working capital
- Declining margins
- Regulatory issues

**Output:**
- List of red flags with severity (High/Medium/Low)
- Evidence citations
- Risk score (1-10)

**Verification:**
- Known red flags detected in test document
- False positive rate acceptable

---

#### Subtask 4.4.4: Governance Agent

**What to do:**
- Create `src/rhp_analyzer/agents/governance.py`
- Analyze corporate governance structure
- Evaluate board composition
- Review management quality

**Analysis areas:**
- Board independence ratio
- Related party relationships
- Promoter background checks
- Key management experience
- Succession planning
- Conflict of interest review
- Compensation structure

**Output:**
- Governance score
- Board composition analysis
- Key person risk assessment
- Recommendations

**Verification:**
- Governance structure analyzed
- Independence calculated correctly

---

#### Subtask 4.4.5: Legal Agent

**What to do:**
- Create `src/rhp_analyzer/agents/legal.py`
- Analyze legal and regulatory disclosures
- Evaluate litigation risks
- Review compliance status

**Analysis areas:**
- Pending litigation summary
- Contingent liabilities
- Regulatory compliance history
- Material contracts review
- Intellectual property status
- Environmental compliance

**Output:**
- Legal risk score
- Litigation exposure summary
- Compliance status
- Key contracts analysis

**Verification:**
- Legal risks identified
- Litigation summarized accurately

---

#### Subtask 4.4.6: Summarizer Agent

**What to do:**
- Create `src/rhp_analyzer/agents/summarizer.py`
- Generate executive summaries
- Create section-wise summaries
- Produce key takeaways

**Summary types:**
- Executive summary (1 page)
- Section summaries (per major section)
- Investment thesis (bull/bear case)
- Key metrics summary

**Output:**
- Multi-level summaries
- Key points extraction
- Investment recommendation framework

**Verification:**
- Summaries accurate
- Key points captured

---

#### Subtask 4.4.7: Promoter Due Diligence Agent (NEW)

**What to do:**
- Create `src/rhp_analyzer/agents/promoter_dd.py`
- Comprehensive "Know the Jockey" analysis
- Analyze promoter background, track record, integrity
- Assess alignment with minority shareholders
- Critical for Indian markets (promoter quality = #1 predictor of value)

**Analysis areas:**
- **Background Check**: Age, qualification, experience years, past ventures
- **Track Record**: Success/failure of previous businesses, exits, bankruptcies
- **Other Directorships**: Count, competing businesses, time commitment concerns
- **Conflicts of Interest**: Group companies in same business, cross-holdings, circular ownership
- **Skin-in-the-Game**: Pre/post-IPO holding %, OFS participation, value at risk at cap price
- **Financial Interests**: Loans from company, guarantees, excessive remuneration
- **Litigation**: Criminal cases, regulatory actions, defaults, as % of post-IPO net worth
- **Selling Pressure**: If selling >20% stake or post-IPO holding <51%

**Red flag triggers:**
- Post-IPO promoter holding <51%
- OFS > 20% of pre-IPO stake
- Criminal litigation against promoter
- Loans from company >10% of net worth
- Related group companies in same business (not subsidiaries)
- Multiple recent auditor resignations in promoter's other companies
- Past defaults or SEBI debarments

**Output:**
- Promoter integrity score (0-10)
- Track record assessment
- Conflict of interest matrix
- Skin-in-the-game analysis (holding value at cap price)
- Litigation risk summary
- Verdict: Trustworthy / Questionable / Red Flag

**Verification:**
- Known promoter issues flagged
- Skin-in-game calculations accurate
- Conflicts identified

---

#### Subtask 4.4.8: Business Analyst Agent

**What to do:**
- Create `src/rhp_analyzer/agents/business.py`
- Analyze company operations in detail
- Generate SWOT analysis
- Assess client concentration risk

**Analysis areas:**
- Revenue breakdown by product, geography, segment
- Capacity utilization trends (installed vs actual production)
- Order book analysis (for B2B/EPC/Defense sectors)
  - Total order book and order book-to-revenue ratio
  - Top order concentration risk
  - Government vs private orders mix
- Manufacturing/service process overview
- Top customer dependency (concentration risk)
- SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)

**System Prompt Focus:**
```python
BUSINESS_PROMPT = """
You are a Business Analyst. Describe the company's operations in detail.
Do not just summarize; analyze the quality of the business.

Analyze:
1. Revenue Mix: Breakdown by Product, Geography, and Customer Segment.
2. Capacity Utilization: Installed capacity vs Actual production (Trends).
3. Order Book Analysis (if applicable for B2B/EPC/Defense sectors)
4. Manufacturing/Service Process
5. SWOT Analysis
6. Client Concentration: Who are the top customers? (Dependency risk)
...
"""
```

**Verification:**
- Revenue mix extracted accurately
- SWOT analysis comprehensive
- Client concentration identified

---

#### Subtask 4.4.9: Industry Analyst Agent

**What to do:**
- Create `src/rhp_analyzer/agents/industry.py`
- Extract TAM/SAM with CAGR
- Analyze competitive landscape
- Identify industry tailwinds and barriers to entry

**Analysis areas:**
- Market Size (TAM): Total Addressable Market and CAGR
- Key Drivers: Government policy, consumption trends, exports
- Competitive Landscape:
  - Listed peers and market share
  - Competitive positioning
- Barriers to Entry: Capital intensity, regulations, technology

**System Prompt Focus:**
```python
INDUSTRY_PROMPT = """
You are a Sector Specialist. Analyze the industry landscape.
Ignore the company's marketing fluff; look for hard industry data.

Analyze:
1. Market Size (TAM): Total Addressable Market and CAGR.
2. Key Drivers: What is pushing this industry forward?
3. Competitive Landscape: Listed peers and market share
4. Barriers to Entry: Is it easy for new players to enter?
...
"""
```

**Verification:**
- TAM and CAGR extracted
- Peers identified correctly
- Industry drivers captured

---

#### Subtask 4.4.10: Management Agent

**What to do:**
- Create `src/rhp_analyzer/agents/management.py`
- Profile key management personnel (MD/CEO/CFO)
- Analyze remuneration vs industry standards
- Check for recent attrition at leadership level

**Analysis areas:**
- Key Management Personnel (KMP) profiles:
  - MD/CEO: Education, experience, past track record
  - CFO: Background and credibility
- Remuneration Analysis:
  - Are salaries market standard or excessive?
  - Compare to company profitability
- Attrition: Any recent high-level exits?

**System Prompt Focus:**
```python
MANAGEMENT_PROMPT = """
You are an HR & Leadership expert. Evaluate the "Jockey" (Management).

Analyze:
1. Key Management Personnel (KMP): MD/CEO, CFO profiles
2. Remuneration: Are salaries market standard or excessive?
3. Attrition: Any recent high-level exits?
...
"""
```

**Verification:**
- All KMPs identified
- Remuneration figures extracted
- Recent changes noted

---

#### Subtask 4.4.11: Capital Structure Agent

**What to do:**
- Create `src/rhp_analyzer/agents/capital_structure.py`
- Integrate with PreIPOInvestorAnalyzer and FloatCalculator
- Analyze WACA vs IPO price
- Generate pre-IPO investor exit analysis
- Build lock-in expiry calendar

**Analysis areas:**
- Weighted Average Cost of Acquisition (WACA):
  - Compare Promoter's WACA vs IPO Price (Flag if >50x multiple)
  - Compare Pre-IPO Investors' WACA vs IPO Price
- Pre-IPO Investor Exit Analysis:
  - Entry Date, Entry Price, Exit Multiple, IRR %
  - Flag investors earning >50% IRR in <2 years
- Offer For Sale (OFS):
  - Who is selling? (Promoters vs PE Funds)
  - Flag if Promoters selling >20% of their holding
- Float & Liquidity Analysis:
  - Day-1 free float % (should be >10%)
  - Day-90 free float % (post anchor unlock)
- Lock-in Expiry Calendar:
  - Anchor unlock (Day 90)
  - Pre-IPO investor unlock (6 months/1 year)
  - Cumulative selling pressure at each milestone

**Output Tables:**
- Pre-IPO Investor Exit Table
- Float Evolution Table
- Lock-in Calendar

**Verdict Categories:**
- FAIR: Reasonable returns for early investors, adequate float
- AGGRESSIVE: >40% IRR for recent investors, low float
- EXIT VEHICLE: Dominated by OFS, promoter selling, poor float

**Verification:**
- WACA calculations verified
- IRR calculations accurate
- Float percentages sum correctly
- Lock-in calendar complete

---

#### Subtask 4.4.12: Valuation Agent

**What to do:**
- Create `src/rhp_analyzer/agents/valuation.py`
- Compare P/E, P/B, EV/EBITDA with disclosed peers
- Calculate PEG ratio
- Identify missing peers from Industry Overview section
- Generate valuation ladder at floor/cap prices

**Analysis areas:**
- Peer Comparison:
  - P/E ratio vs peer median
  - P/B ratio vs peer median
  - EV/EBITDA vs peer median
- PEG Ratio: P/E to growth (PAT CAGR)
- Missing Peers: Peers mentioned in Industry Overview but excluded from Basis for Issue Price
- Premium/Discount Analysis:
  - Calculate % premium/discount vs peer median
  - At floor price and cap price

**Output Tables:**
- Peer Comparison Table with multiples
- Valuation Ladder (Floor/Fair/Cap)

**Verdict Categories:**
- ATTRACTIVE: Trading at discount to peers with growth justification
- FAIR: Trading at par with peers
- EXPENSIVE: Significant premium with no justification
- OVERPRICED: >50% premium to peers

**Verification:**
- Peer multiples extracted correctly
- Premium/discount calculated accurately
- Missing peers identified

---

#### Subtask 4.4.13: Utilization Agent

**What to do:**
- Create `src/rhp_analyzer/agents/utilization.py`
- Integrate with ObjectsOfIssueTracker and DebtStructureAnalyzer
- Analyze Fresh Issue vs OFS split
- Check deployment timeline readiness
- Verify monitoring agency appointment

**Analysis areas:**
- Fresh Issue vs OFS Split:
  - Fresh Issue %: Money coming to company
  - OFS %: Money going to sellers
  - Flag if OFS > 50% (Pure exit play)
- Use of Fresh Issue Proceeds:
  - Debt repayment % (reducing leverage vs growth)
  - Capex % with deployment timeline
  - Working capital %
  - General corporate purposes % (flag if >15%)
- Deployment Readiness:
  - Land acquisition status (acquired vs to-be-acquired)
  - Regulatory approvals pending
  - Construction timelines
- Monitoring Agency: Required if >50% for objects other than GCP

**Red Flags to Identify:**
- GCP > 25% without specifics
- Capex on land not yet acquired
- No deployment timeline for major items
- Missing monitoring agency when required

**Verification:**
- Uses sum to 100% of fresh issue
- Deployment timelines extracted
- Red flags identified correctly

---

#### Subtask 4.4.14: Q&A Agent

**What to do:**
- Create `src/rhp_analyzer/agents/qa.py`
- Enable on-demand question answering about the RHP
- Use RAG to retrieve relevant context
- Enforce citation requirements
- Support interactive and single-question modes

**Features:**
- Accept natural language questions about the RHP
- Retrieve top-K relevant chunks using vector search
- Generate answers with page citations
- Return "Insufficient evidence in the document" when unable to find answer
- Support follow-up questions with conversation context

**Implementation:**
```python
class QAAgent(BaseAgent):
    """Answers specific questions using RAG"""
    def answer(self, question: str, state: AnalysisState) -> str:
        context_chunks = self.retrieve_context(
            query=question,
            filters={},
            top_k=5
        )
        # Build prompt with context and generate answer
        pass
```

**CLI Integration:**
- `rhp-analyzer query <doc_id>` - Interactive Q&A mode
- `rhp-analyzer query <doc_id> --question "..."` - Single question mode

**Verification:**
- Relevant context retrieved
- Answers include page citations
- Handles unanswerable questions gracefully

---

#### Subtask 4.4.15: Investment Committee Agent

**What to do:**
- Create `src/rhp_analyzer/agents/investment_committee.py`
- Act as Chief Investment Officer making final Subscribe/Avoid decision
- Synthesize all analyst reports into weighted scorecard
- Generate formal Investment Committee Memo
- This is the FINAL DECISION MAKER

**Input Sources:**
- Business Analyst report → Business Moat score
- Industry Analyst report → Industry Tailwinds score
- Management Agent report → Management quality assessment
- Capital Structure Agent report → Float & selling pressure
- Forensic Accountant report → Financial Health score
- Valuation Agent output → Valuation Comfort score
- Utilization Agent report → Use of proceeds quality
- Governance Agent report → Governance Quality score
- Promoter DD Agent report → Promoter Quality score
- Red Flag Agent report → Risk Assessment
- Legal Agent report → Legal Risk
- Projection Engine scenarios → Forward metrics
- All other analytical modules

**Weighted Scorecard (0-100):**
- Financial Health (30%): Base on Base scenario ROE bands
  - ROE >20% → 10/10, 15-20% → 8/10, 10-15% → 6/10, <10% → 3/10
  - Adjust for CFO/EBITDA gap penalties
- Valuation Comfort (20%): Peer median premium/discount
  - Discount >10% → 10/10, Fair value → 8/10, Premium <20% → 6/10, Premium >20% → 3/10
- Governance Quality (20%): From Governance Agent
  - **CRITICAL**: If <5/10, automatic VETO
- Promoter Quality (15%): From Promoter DD Agent
  - Trustworthy → 10/10, Questionable → 6/10, Red Flag → 0/10 + VETO
- Business Moat (10%): Qualitative assessment
- Industry Tailwinds (5%): TAM growth, sector trends

**Quantitative Guardrails (Must Report):**
- Base/Bull/Stress diluted EPS for next 3 years
- Base/Bull/Stress ROE and ROIC post-issue
- Price-band sensitivity (floor vs cap upside/downside)
- Implied upside/downside vs normalized peer median
- Promoter pledge % (VETO if >25%)
- RPT concentration % (Major concern if >20%)
- Litigation/Net Worth % (VETO if >10%)
- Post-issue float % (concern if <25%)

**Veto Conditions (Automatic AVOID rating):**
- Governance score <5/10
- Promoter pledge >25%
- Litigation >10% of post-issue net worth
- Promoter post-issue holding <40%
- Criminal charges against promoters
- SEBI/NCLT proceedings pending

**Final Verdict Options:**
1. **SUBSCRIBE (Listing Gains)**: Good hype, fair pricing, short-term play (exit in days/weeks)
2. **SUBSCRIBE (Long Term)**: Great business, good governance, hold for 3-5 years
3. **AVOID (Expensive)**: Business is fine but valuation unreasonable
4. **AVOID (Toxic)**: Fraud risk, governance failures, or severe business issues

**Output Format - Investment Committee Memo:**
```
## Investment Committee Memorandum
### [Company Name] IPO - [Date]

**Recommendation**: [SUBSCRIBE/AVOID] ([LISTING GAINS/LONG TERM/EXPENSIVE/TOXIC])

**Scorecard Summary** (0-100):
- Overall Score: XX/100
- Financial Health: XX/30
- Valuation Comfort: XX/20
- Governance Quality: XX/20
- Promoter Quality: XX/15
- Business Moat: XX/10
- Industry Tailwinds: XX/5

**Base Case Projections (3-year forward at cap price)**:
- Diluted EPS FY26E: ₹XX | FY27E: ₹XX | FY28E: ₹XX
- ROE: XX% | ROIC: XX%
- Net Debt/EBITDA: Xx

**Valuation**:
- At Floor (₹XX): P/E XXx | Premium/Discount: XX% vs peers
- At Cap (₹XX): P/E XXx | Premium/Discount: XX% vs peers
- Implied Upside/Downside to Fair Value: XX%

**Key Strengths**: [bullet points with citations]
**Key Concerns**: [bullet points with severity flags]

**Veto Flags** (if any): [Critical issues that trigger automatic AVOID]

**Investment Rationale**: [2-3 paragraphs justifying the verdict]

**Risk Factors to Monitor**: [post-listing tracking points]
```

**Verification:**
- Scorecard math correct (weights sum to 100%)
- Veto conditions enforced
- All quantitative guardrails reported
- Verdict aligns with scorecard

---

### Milestone 4.5: Self-Critic Agent

**Deliverables:** Quality assurance agent that validates other agents' outputs

#### Subtask 4.5.1: Create Self-Critic Agent

**What to do:**
- Create `src/rhp_analyzer/agents/critic.py`
- Review outputs from all other agents
- Verify claims against source document
- Identify inconsistencies

**Validation checks:**
- Factual accuracy (claims match sources)
- Internal consistency (no contradictions)
- Completeness (no major gaps)
- Citation verification
- Numerical accuracy

**Verification:**
- Known errors caught
- Valid claims not flagged incorrectly

---

#### Subtask 4.5.2: Add Claim Verification

**What to do:**
- Extract claims from agent outputs
- Retrieve relevant source content
- Compare claim against source
- Flag discrepancies

**Verification process:**
1. Parse claims from agent output
2. For each claim, retrieve cited source
3. Use LLM to verify claim matches source
4. Score confidence of verification

**Verification:**
- False claims detected
- True claims verified

---

#### Subtask 4.5.3: Add Feedback Loop

**What to do:**
- Route critic feedback to original agents
- Allow agents to revise based on feedback
- Limit revision iterations (max 2)
- Track revision history

**Feedback flow:**
1. Critic identifies issues
2. Issues routed to relevant agent
3. Agent revises output
4. Critic re-validates
5. Final output with revision notes

**Verification:**
- Feedback improves quality
- Infinite loops prevented

---

### Milestone 4.6: LangGraph Orchestration

**Deliverables:** State machine managing agent workflow

#### Subtask 4.6.1: Define Analysis State

**What to do:**
- Create `src/rhp_analyzer/agents/state.py`
- Define TypedDict for analysis state
- Include all shared data between agents
- Track workflow progress

**State fields:**
- document_id
- current_phase
- completed_agents
- agent_outputs (dict)
- errors
- warnings
- metadata

**Verification:**
- State correctly typed
- All agents can read/write state

---

#### Subtask 4.6.2: Create Workflow Graph

**What to do:**
- Create `src/rhp_analyzer/agents/workflow.py`
- Define LangGraph StateGraph
- Add nodes for each agent
- Define edges and conditions

**Workflow structure:**
```
START
  ↓
Architect Agent
  ↓
┌─────────────────────────────┐
│ Parallel: Forensic, RedFlag,│
│ Governance, Legal           │
└─────────────────────────────┘
  ↓
Self-Critic Agent
  ↓ (if revisions needed)
Revision Loop (max 2x)
  ↓
Summarizer Agent
  ↓
END
```

**Verification:**
- Graph executes in correct order
- Parallel agents run concurrently
- Critic loop works

---

#### Subtask 4.6.3: Add Workflow Persistence

**What to do:**
- Save workflow state after each node
- Allow resume from any checkpoint
- Handle agent failures gracefully
- Log state transitions

**Persistence:**
- Save to JSON file after each agent
- Include timestamp and duration
- Store partial results

**Verification:**
- Kill mid-workflow, resume works
- Completed agents not re-run

---

#### Subtask 4.6.4: Create Workflow Tests

**What to do:**
- Create `tests/unit/test_workflow.py`
- Test state transitions
- Test error handling
- Mock agent execution

**Verification:**
- `pytest tests/unit/test_workflow.py` passes
- Workflow behaves correctly

---

### Milestone 4.7: Phase 4 Checkpoint

#### Subtask 4.7.1: Agent Integration Test

**What to do:**
- Run full agent workflow on sample RHP
- Verify all agents produce output
- Check self-critic catches issues
- Measure total processing time

**Acceptance criteria:**
- [ ] All 14 agents execute successfully:
  - Analysis: Business, Industry, Management, Capital Structure, Forensic, Red Flag, Governance, Legal, Valuation, Utilization, Promoter DD
  - Validation: Self-Critic, Summarizer
  - Decision: Investment Committee
- [ ] Self-critic validates outputs
- [ ] Citations traceable to sources
- [ ] Processing time < 60 minutes (with API)
- [ ] No hallucinated facts in spot check
- [ ] Investment Committee scorecard calculated correctly
- [ ] Veto conditions enforced properly

---

#### Subtask 4.7.2: Git Checkpoint

**What to do:**
- Commit all changes
- Create tag: `v0.4.0-agents`

**Commit messages:**
- `feat(agents): add HuggingFace LLM client with retry`
- `feat(agents): add RAG retrieval system`
- `feat(agents): implement Business Analyst Agent`
- `feat(agents): implement Industry Analyst Agent`
- `feat(agents): implement Management Agent`
- `feat(agents): implement Capital Structure Agent`
- `feat(agents): implement Forensic Accountant Agent (enhanced)`
- `feat(agents): implement Red Flag Agent`
- `feat(agents): implement Governance Agent`
- `feat(agents): implement Legal Agent`
- `feat(agents): implement Valuation Agent`
- `feat(agents): implement Utilization Agent`
- `feat(agents): add Promoter Due Diligence Agent`
- `feat(agents): add Summarizer Agent`
- `feat(agents): add Self-Critic validation agent`
- `feat(agents): add Q&A Agent for interactive queries`
- `feat(agents): add Investment Committee Agent (final decision maker with scorecard)`
- `feat(agents): add LangGraph workflow orchestration`

---

## Phase 5: Report Generation (Weeks 10-11)

**Goal:** Generate professional markdown and PDF reports from agent analysis.

---

### Milestone 5.1: Report Data Aggregation

**Deliverables:** Consolidate all agent outputs into unified report structure

#### Subtask 5.1.1: Create Report Builder Module

**What to do:**
- Create `src/rhp_analyzer/reporting/builder.py`
- Aggregate outputs from all agents
- Resolve conflicts between agents
- Create hierarchical report structure

**Aggregation tasks:**
- Collect all agent outputs from state
- Merge overlapping findings
- Prioritize by severity/importance
- Create unified finding list

**Verification:**
- All agent outputs included
- No duplicate findings

---

#### Subtask 5.1.2: Define Report Schema

**What to do:**
- Create `src/rhp_analyzer/reporting/schema.py`
- Define Pydantic models for report sections
- Support multiple report formats
- Include metadata

**Report structure:**
```python
class Report:
    metadata: ReportMetadata
    executive_summary: ExecutiveSummary
    company_overview: CompanyOverview
    financial_analysis: FinancialAnalysis
    risk_assessment: RiskAssessment
    governance_review: GovernanceReview
    legal_review: LegalReview
    investment_thesis: InvestmentThesis
    appendices: List[Appendix]
```

**Verification:**
- Schema covers all sections
- Validation works

---

#### Subtask 5.1.3: Add Data Validation

**What to do:**
- Validate report completeness
- Check for required sections
- Verify citation integrity
- Flag missing data

**Validation checks:**
- All required sections present
- Citations resolve to pages
- No empty sections
- Numerical data formatted

**Verification:**
- Incomplete reports flagged
- Validation messages helpful

---

### Milestone 5.2: Markdown Generation

**Deliverables:** Generate well-formatted markdown report

#### Subtask 5.2.1: Create Markdown Template System

**What to do:**
- Create `src/rhp_analyzer/reporting/markdown.py`
- Use Jinja2 for templating
- Create master report template
- Support section templates

**Template location:** `templates/report/`

**Templates to create:**
- `report.md.j2` - Master template
- `sections/executive_summary.md.j2`
- `sections/financial_analysis.md.j2`
- `sections/risk_assessment.md.j2`
- `sections/governance.md.j2`
- `sections/legal.md.j2`
- `sections/investment_thesis.md.j2`

**Verification:**
- Templates render correctly
- Markdown valid

---

#### Subtask 5.2.2: Design Report Format

**What to do:**
- Define visual hierarchy
- Create consistent formatting rules
- Add table formatting
- Include charts/diagrams (text-based)

**Format elements:**
- Headers (H1 for title, H2 for sections, H3 for subsections)
- Tables for financial data
- Bullet lists for findings
- Blockquotes for citations
- Severity badges (emoji-based)
- Rating scales (★★★☆☆)

**Verification:**
- Report visually appealing
- Consistent styling throughout

---

#### Subtask 5.2.3: Add Table Formatting

**What to do:**
- Format financial tables in markdown
- Support large tables (scrollable in PDF)
- Add table captions
- Align numbers correctly

**Table types:**
- Financial summary (key metrics)
- Year-over-year comparison
- Risk factor matrix
- Governance scorecard

**Verification:**
- Tables render correctly
- Numbers aligned

---

#### Subtask 5.2.4: Create Markdown Tests

**What to do:**
- Create `tests/unit/test_markdown.py`
- Test template rendering
- Test with sample data
- Validate markdown output

**Verification:**
- `pytest tests/unit/test_markdown.py` passes
- Output is valid markdown

---

### Milestone 5.3: PDF Generation

**Deliverables:** Convert markdown to styled PDF

#### Subtask 5.3.1: Set Up WeasyPrint

**What to do:**
- Create `src/rhp_analyzer/reporting/pdf.py`
- Install WeasyPrint with GTK (Windows-specific steps)
- Create markdown → HTML → PDF pipeline
- Handle installation issues gracefully

**Windows installation:**
- Install GTK3 runtime
- Add to PATH
- Verify with test conversion

**Fallback if WeasyPrint fails:**
- Use `md2pdf` or `pandoc`
- Document alternative

**Verification:**
- Simple PDF generates correctly
- Fonts render properly

---

#### Subtask 5.3.2: Create PDF Stylesheet

**What to do:**
- Create `templates/report/styles.css`
- Define professional styling
- Set page layout (A4, margins)
- Style headers, tables, lists

**CSS elements:**
- Page size and margins
- Font families (system fonts)
- Header/footer with page numbers
- Table styling
- Color scheme (professional)
- Page break controls

**Verification:**
- PDF looks professional
- Multi-page layout correct

---

#### Subtask 5.3.3: Add PDF Features

**What to do:**
- Add table of contents
- Add page numbers
- Add headers/footers
- Add cover page

**Features:**
- Auto-generated ToC from headers
- Page X of Y numbering
- Document title in header
- Generation date in footer
- Cover page with company logo placeholder

**Verification:**
- ToC links work
- Page numbers correct

---

#### Subtask 5.3.4: Create PDF Tests

**What to do:**
- Create `tests/unit/test_pdf.py`
- Test PDF generation
- Verify page count
- Check file validity

**Verification:**
- `pytest tests/unit/test_pdf.py` passes
- PDF opens in readers

---

### Milestone 5.4: Report CLI Commands

**Deliverables:** CLI commands for report generation

#### Subtask 5.4.1: Add Generate Command

**What to do:**
- Add `generate` subcommand to CLI
- Accept document ID or analysis results
- Support format selection (md, pdf, both)
- Specify output location

**Command options:**
- `rhp-analyzer generate <doc_id>`
- `--format md|pdf|both`
- `--output-dir <path>`
- `--template <template_name>`

**Verification:**
- Command generates report
- Format selection works

---

#### Subtask 5.4.2: Add Report Customization

**What to do:**
- Support custom templates
- Allow section selection
- Support branding options
- Add verbosity levels

**Customization:**
- Include/exclude sections
- Custom CSS for PDF
- Report title override
- Author name

**Verification:**
- Customizations applied
- Partial reports work

---

### Milestone 5.5: Phase 5 Checkpoint

#### Subtask 5.5.1: Report Generation Test

**What to do:**
- Generate full report from sample analysis
- Verify markdown quality
- Verify PDF quality
- Check all sections present

**Acceptance criteria:**
- [ ] Markdown report complete
- [ ] PDF generates without errors
- [ ] All sections have content
- [ ] Citations present and formatted
- [ ] Tables render correctly
- [ ] Report length 15-30 pages

---

#### Subtask 5.5.2: Git Checkpoint

**What to do:**
- Commit all changes
- Create tag: `v0.5.0-reporting`

**Commit messages:**
- `feat(reporting): add Jinja2 template system`
- `feat(reporting): add markdown report generation`
- `feat(reporting): add PDF generation with WeasyPrint`
- `feat(cli): add generate command for reports`

---

## Phase 6: Production Polish (Weeks 12-13)

**Goal:** Error handling, testing, documentation, and production readiness.

---

### Milestone 6.1: Comprehensive Error Handling

**Deliverables:** Robust error handling throughout application

#### Subtask 6.1.1: Define Exception Hierarchy

**What to do:**
- Create `src/rhp_analyzer/exceptions.py`
- Define custom exception classes
- Add error codes and messages
- Support error context

**Exception classes:**
- `RHPAnalyzerError` (base)
- `PDFProcessingError`
- `ExtractionError`
- `LLMError`
- `StorageError`
- `ReportGenerationError`
- `ConfigurationError`

**Verification:**
- Exceptions raised appropriately
- Error messages helpful

---

#### Subtask 6.1.2: Add Global Error Handler

**What to do:**
- Create centralized error handler
- Log errors with context
- Provide user-friendly messages
- Support error recovery hints

**Features:**
- Catch and log all exceptions
- Convert to user-friendly messages
- Suggest remediation steps
- Support --debug for full tracebacks

**Verification:**
- Errors don't crash CLI
- Helpful messages shown

---

#### Subtask 6.1.3: Add Graceful Degradation

**What to do:**
- Continue processing on non-fatal errors
- Mark sections as incomplete
- Warn user about issues
- Generate partial reports if possible

**Degradation scenarios:**
- Table extraction fails → skip tables, warn
- One agent fails → continue with others
- PDF generation fails → output markdown only

**Verification:**
- Partial failures don't stop entire run
- Warnings logged and shown

---

### Milestone 6.2: Comprehensive Testing

**Deliverables:** High test coverage with integration tests

#### Subtask 6.2.1: Add Integration Tests

**What to do:**
- Create `tests/integration/` test suite
- Test full pipelines end-to-end
- Use real sample RHP (small subset)
- Test CLI commands

**Integration tests:**
- `test_ingestion_pipeline.py`
- `test_analysis_pipeline.py`
- `test_report_pipeline.py`
- `test_full_workflow.py`

**Verification:**
- `pytest tests/integration/` passes
- Full workflow completes

---

#### Subtask 6.2.2: Add Edge Case Tests

**What to do:**
- Test unusual inputs
- Test error scenarios
- Test resource limits
- Test timeout handling

**Edge cases:**
- Empty PDF
- Very large PDF (500+ pages)
- PDF with no tables
- API timeout during analysis
- Disk full during save

**Verification:**
- Edge cases handled gracefully
- No crashes on bad input

---

#### Subtask 6.2.3: Add Performance Tests

**What to do:**
- Measure processing time per phase
- Profile memory usage
- Identify bottlenecks
- Set performance baselines

**Metrics to measure:**
- Ingestion time (pages/second)
- Embedding time (chunks/second)
- LLM latency (per agent)
- Total end-to-end time

**Verification:**
- Performance within targets
- No memory leaks

---

#### Subtask 6.2.4: Achieve Coverage Target

**What to do:**
- Run coverage analysis
- Identify gaps
- Add tests for uncovered code
- Target 80%+ coverage

**Command:**
```bash
pytest --cov=src/rhp_analyzer --cov-report=html tests/
```

**Verification:**
- Coverage > 80%
- Critical paths 100% covered

---

### Milestone 6.3: Documentation

**Deliverables:** Comprehensive documentation for users and developers

#### Subtask 6.3.1: Create User Guide

**What to do:**
- Create `docs/user-guide.md`
- Document installation steps
- Document CLI usage
- Provide examples

**Sections:**
- Installation
- Configuration
- Basic usage
- Advanced options
- Troubleshooting

**Verification:**
- New user can follow guide
- All commands documented

---

#### Subtask 6.3.2: Create Developer Guide

**What to do:**
- Create `docs/developer-guide.md`
- Document architecture
- Explain extension points
- Describe contribution process

**Sections:**
- Architecture overview
- Module descriptions
- Adding new agents
- Testing guidelines
- Code style guide

**Verification:**
- Developer can understand codebase
- Extension process clear

---

#### Subtask 6.3.3: Add API Documentation

**What to do:**
- Add docstrings to all public functions
- Generate API docs (pdoc or mkdocs)
- Document data schemas
- Include code examples

**Documentation coverage:**
- All public classes
- All public functions
- All Pydantic models

**Verification:**
- API docs generate without errors
- Docstrings complete

---

#### Subtask 6.3.4: Create README

**What to do:**
- Update `README.md` comprehensively
- Add badges (tests, coverage, version)
- Include quick start guide
- Link to full documentation

**README sections:**
- Project description
- Features list
- Quick start (5 steps)
- Configuration overview
- Contributing
- License

**Verification:**
- README is clear and complete
- Quick start works for new users

---

### Milestone 6.4: Final Polish

**Deliverables:** Production-ready application

#### Subtask 6.4.1: Add Version Management

**What to do:**
- Set up semantic versioning
- Add version to CLI output
- Create CHANGELOG.md
- Document release process

**Version format:** `MAJOR.MINOR.PATCH`

**Verification:**
- Version displays correctly
- CHANGELOG up to date

---

#### Subtask 6.4.2: Create Sample Outputs

**What to do:**
- Process sample RHP
- Save example outputs
- Include in repository (or link)
- Document expected output

**Sample outputs:**
- `examples/sample_report.md`
- `examples/sample_report.pdf`
- `examples/sample_config.yaml`

**Verification:**
- Examples demonstrate capabilities
- Users understand expected output

---

#### Subtask 6.4.3: Performance Optimization

**What to do:**
- Profile slow sections
- Optimize bottlenecks
- Add caching where beneficial
- Document performance tips

**Optimization areas:**
- Embedding batching
- LLM prompt optimization
- File I/O optimization
- Memory usage reduction

**Verification:**
- Processing time improved
- Memory usage stable

---

#### Subtask 6.4.4: Security Review

**What to do:**
- Audit API key handling
- Check for sensitive data logging
- Review file permissions
- Document security considerations

**Security checks:**
- API keys not logged
- Temp files cleaned up
- Output permissions appropriate
- No hardcoded secrets

**Verification:**
- No secrets in logs or outputs
- Keys loaded from env only

---

### Milestone 6.5: Phase 6 Checkpoint (Final)

#### Subtask 6.5.1: Full System Test

**What to do:**
- Process real RHP end-to-end
- Verify complete workflow
- Check all outputs
- Measure final performance

**Acceptance criteria:**
- [ ] Full RHP processes without errors
- [ ] Report quality meets expectations
- [ ] Processing time < 90 minutes
- [ ] All tests pass
- [ ] Coverage > 80%
- [ ] Documentation complete

---

#### Subtask 6.5.2: Final Git Checkpoint

**What to do:**
- Commit all changes
- Create tag: `v1.0.0`
- Create GitHub release (if applicable)

**Commit messages:**
- `feat(error): add comprehensive error handling`
- `test: add integration test suite`
- `docs: add user and developer guides`
- `chore: prepare v1.0.0 release`

---

## 6. Project Checkpoints

### Checkpoint Summary Table

| Checkpoint | Version Tag | Key Deliverables | Acceptance Criteria |
|------------|-------------|------------------|---------------------|
| Phase 1 Complete | v0.1.0-foundation | CLI skeleton, logging, config | CLI runs, logs to file, config loads |
| Phase 2 Complete | v0.2.0-ingestion | PDF processing, table extraction, 8 specialized extractors | RHP text + tables + specialized data extracted |
| Phase 2.5 Complete | v0.2.5-financial-modeling | Projection Engine, Valuation Module, Governance Rulebook | Scenarios generated, valuations calculated, rules evaluated |
| Phase 3 Complete | v0.3.0-storage | Vector DB, SQL, chunking, 9 analytical modules | Search works, analytics pipeline complete |
| Phase 4 Complete | v0.4.0-agents | 14 agents (incl. all analysis + Investment Committee) + workflow | Analysis completes with scorecard and final verdict |
| Phase 5 Complete | v0.5.0-reporting | Markdown + PDF reports with Investment Committee Memo | Professional report with actionable verdict generated |
| Phase 6 Complete | v1.0.0 | Production ready | All tests pass, docs complete |

---

### Git Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Adding tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

**Scopes:**
- `cli`: CLI commands
- `config`: Configuration
- `logging`: Logging system
- `ingestion`: PDF processing
- `storage`: Databases
- `agents`: LLM agents
- `reporting`: Report generation

---

## 7. Appendices

### Appendix A: Key Data Models & Schemas

**Note**: This appendix lists the main data classes defined in blueprint.md that need to be implemented during development.

#### Core Document Models
```python
@dataclass
class Page:
    """PDF page representation"""
    page_number: int
    text: str
    tables: List[str]
    has_images: bool
    is_scanned: bool

@dataclass
class Section:
    """Document section hierarchy"""
    section_id: str
    title: str
    level: int
    start_page: int
    end_page: int
    content: str
    subsections: List['Section']

@dataclass
class Chunk:
    """Semantic chunk for embedding"""
    chunk_id: str
    document_id: str
    section_name: str
    page_range: tuple
    chunk_type: str  # narrative, table, mixed
    content: str
    token_count: int
    embedding: Optional[List[float]]
```

#### Financial Data Models
```python
@dataclass
class FinancialData:
    """Financial metrics for a fiscal year"""
    fiscal_year: str
    revenue: float
    ebitda: float
    pat: float
    total_assets: float
    total_equity: float
    total_debt: float
    roe: float
    roce: float
    # ... additional financial metrics

@dataclass
class ProjectionScenario:
    """Forward projections from Projection Engine"""
    scenario_name: str  # Base, Bull, Stress
    fiscal_years: List[str]
    revenue: List[float]
    ebitda: List[float]
    pat: List[float]
    diluted_eps: List[float]
    roe: List[float]
    roic: List[float]
    assumptions: Dict[str, str]
    citations: List[str]
```

#### Specialized Analysis Models
```python
@dataclass
class PromoterDossier:
    """Comprehensive promoter profile"""
    name: str
    din: str
    age: int
    qualification: str
    experience_years: int
    other_directorships: List[str]
    other_directorships_count: int
    group_companies_in_same_line: List[str]
    shareholding_pre_ipo: float
    shareholding_post_ipo: float
    shares_selling_via_ofs: int
    skin_in_game_post_ipo: float  # Value at cap price
    litigation_as_defendant: List[str]
    litigation_count: int
    litigation_amount_cr: float
    loans_from_company: float
    remuneration_last_3_years: List[float]

@dataclass
class PreIPOInvestor:
    """Pre-IPO investor analysis"""
    name: str
    category: str  # PE/VC, Strategic, ESOP Trust, Other
    entry_date: str
    entry_price: float
    shares_held_pre_ipo: int
    shares_selling_via_ofs: int
    ofs_amount: float
    implied_return_multiple_at_floor: float
    implied_return_multiple_at_cap: float
    implied_irr_at_floor: float
    implied_irr_at_cap: float
    holding_period_months: int
    lock_in_period: str
    lock_in_expiry_date: Optional[str]
    shares_locked: int

@dataclass
class OrderBookAnalysis:
    """Order book analysis for B2B/EPC companies"""
    applicable: bool
    total_order_book: float  # in crores
    order_book_as_of_date: str
    order_book_to_ltm_revenue: float
    top_5_orders_value: float
    top_5_orders_concentration: float  # %
    executable_in_12_months: float
    executable_in_12_months_percent: float
    government_orders_percent: float
    order_book_1yr_ago: Optional[float]
    order_book_growth_yoy: Optional[float]
    citations: List[str]

@dataclass
class DebtStructure:
    """Comprehensive debt analysis"""
    total_debt: float
    secured_debt: float
    unsecured_debt: float
    short_term_debt: float
    long_term_debt: float
    weighted_avg_interest_rate: float
    highest_interest_rate: float
    lowest_interest_rate: float
    maturing_within_1_year: float
    maturing_1_to_3_years: float
    maturing_3_to_5_years: float
    maturing_beyond_5_years: float
    debt_repayment_from_ipo: float
    post_ipo_debt: float
    debt_to_equity_pre_ipo: float
    debt_to_equity_post_ipo: float
    debt_to_ebitda: float
    has_financial_covenants: bool
    covenant_details: List[str]
    citations: List[str]

@dataclass
class FloatAnalysis:
    """Free float and liquidity analysis"""
    total_shares_post_issue: int
    fresh_issue_shares: int
    promoter_locked_shares: int
    promoter_locked_percent: float
    anchor_locked_shares: int
    pre_ipo_locked_shares: int
    day_1_free_float_shares: int
    day_1_free_float_percent: float
    day_90_free_float_percent: float
    retail_quota_shares: int
    retail_quota_percent: float
    implied_daily_volume: float
    lock_in_calendar: List[Dict]  # Timeline of unlocks

@dataclass
class ValuationSnapshot:
    """Peer valuation comparison"""
    price_point: str  # Floor, Cap, Fair
    p_e_ratio: float
    p_b_ratio: float
    ev_ebitda: float
    peer_median_p_e: float
    premium_discount_vs_peers: float  # %
    missing_peers: List[str]
    citations: List[str]
```

#### Decision & Scoring Models
```python
@dataclass
class Scorecard:
    """Investment Committee final scorecard"""
    overall_score: float  # 0-100
    financial_health: float  # 0-30
    valuation_comfort: float  # 0-20
    governance_quality: float  # 0-20
    promoter_quality: float  # 0-15
    business_moat: float  # 0-10
    industry_tailwinds: float  # 0-5

    # Veto metrics
    promoter_pledge_percent: float
    rpt_concentration_percent: float
    litigation_to_networth_percent: float
    post_issue_promoter_holding: float

    # Verdict
    verdict: str  # SUBSCRIBE/AVOID
    verdict_type: str  # LISTING GAINS, LONG TERM, EXPENSIVE, TOXIC
    veto_flags: List[str]

@dataclass
class RiskExposure:
    """Quantified risk item"""
    category: str  # Litigation, Contingent, Regulatory
    entity: str  # Company, Promoter, Director, Subsidiary
    description: str
    amount: float
    probability: str  # High, Medium, Low
    risk_weighted_amount: float
    percent_of_networth: float
    hearing_date: Optional[str]
    citation: str

@dataclass
class ContingentLiabilityAnalysis:
    """Categorized contingent liabilities"""
    total_contingent: float
    tax_disputes: float
    customs_excise_disputes: float
    legal_claims: float
    bank_guarantees: float
    other_categories: Dict[str, float]
    risk_weighted_total: float
    percent_of_post_ipo_networth: float
    earmarked_in_objects: bool
    citations: List[str]

@dataclass
class ObjectsOfIssueAnalysis:
    """Use of proceeds breakdown"""
    fresh_issue_amount: float
    ofs_amount: float
    debt_repayment_amount: float
    debt_repayment_percent: float
    capex_amount: float
    capex_percent: float
    working_capital_amount: float
    working_capital_percent: float
    general_corp_purposes_amount: float
    general_corp_purposes_percent: float
    offer_expenses: float
    # Deployment details
    capex_items: List[Dict]  # {item, amount, timeline}
    land_acquired_percent: float
    approvals_pending: List[str]
    deployment_risks: List[str]
    citations: List[str]
```

#### Agent Output Models
```python
@dataclass
class AgentAnalysis:
    """Standard agent output format"""
    agent_name: str
    analysis_type: str
    findings: List[str]
    confidence: float
    citations: List[str]
    warnings: List[str]
    processing_time: float

@dataclass
class CitationRecord:
    """Audit trail for sourced claims"""
    claim_id: str
    section: str
    page_numbers: List[int]
    paragraph_id: str
    text_snippet: str
```

#### Additional Financial Models
```python
@dataclass
class CashFlowAnalysis:
    """Enhanced cash flow analysis"""
    fiscal_year: str
    cfo: float  # Cash from Operations
    cfi: float  # Cash from Investing
    cff: float  # Cash from Financing
    capex: float
    fcf: float  # Free Cash Flow (CFO - Capex)
    fcf_margin: float  # FCF / Revenue %
    capex_to_revenue: float  # %
    cfo_to_ebitda: float  # % (should be >70%)
    cfo_to_pat: float  # %
    is_cash_burning: bool
    monthly_burn_rate: Optional[float]  # If burning
    runway_months: Optional[int]  # If burning
    maintenance_capex_estimate: float
    growth_capex_estimate: float
    working_capital_change: float
    citations: List[str]

@dataclass
class WorkingCapitalAnalysis:
    """Working capital with sector benchmarks"""
    fiscal_year: str
    receivable_days: float
    inventory_days: float
    payable_days: float
    cash_conversion_cycle: float  # DSO + DIO - DPO
    sector: str
    sector_avg_ccc: Optional[float]
    variance_vs_sector: float  # %
    receivable_days_change_yoy: float
    inventory_days_change_yoy: float
    receivable_growth_vs_revenue_growth: float  # % difference
    is_channel_stuffing_risk: bool  # If receivables growing faster than revenue
    citations: List[str]

@dataclass
class StubPeriodAnalysis:
    """Interim period financial analysis"""
    stub_period: str  # e.g., "6 months ended Sep 30, 2024"
    comparable_prior_period: str
    stub_revenue: float
    prior_revenue: float
    revenue_growth_yoy: float
    stub_ebitda: float
    prior_ebitda: float
    ebitda_growth_yoy: float
    stub_ebitda_margin: float
    prior_ebitda_margin: float
    margin_expansion: float  # positive = expansion
    annualized_revenue: float
    annualized_ebitda: float
    stub_growth_below_historical_cagr: bool
    margin_compression_in_stub: bool
    one_time_items: List[str]
    seasonality_notes: str
    citations: List[str]

@dataclass
class IPODetails:
    """Core IPO parameters"""
    issue_size: float  # Total issue size in Cr
    fresh_issue_amount: float
    ofs_amount: float
    price_band_floor: float
    price_band_cap: float
    shares_pre_issue: int
    shares_post_issue: int
    fresh_issue_shares: int
    anchor_quota_shares: int
    retail_quota_shares: int
    issue_open_date: str
    issue_close_date: str
    listing_date: str
    face_value: float
    lot_size: int
    minimum_investment: float

@dataclass
class RiskFactor:
    """Individual risk factor"""
    risk_id: str
    category: str  # Operational, Financial, Regulatory, Industry, Legal
    description: str
    severity: str  # High, Medium, Low
    page_reference: int
    quantifiable_impact: Optional[float]
    mitigation_mentioned: bool
    section: str

@dataclass
class Report:
    """Final report structure"""
    metadata: Dict
    executive_summary: str
    investment_verdict: str  # SUBSCRIBE/AVOID
    verdict_type: str  # LISTING GAINS, LONG TERM, EXPENSIVE, TOXIC
    scorecard: 'Scorecard'
    scenario_dashboard: Dict  # Base/Bull/Stress projections
    valuation_analysis: 'ValuationSnapshot'
    company_overview: str
    financial_analysis: str
    risk_assessment: str
    governance_review: str
    legal_review: str
    investment_thesis: str
    appendices: List[Dict]
    citations: List['CitationRecord']
```

#### Scorecard Implementation
```python
@dataclass
class Scorecard:
    """Investment Committee final scorecard"""
    overall_score: float  # 0-100
    financial_health: float  # 0-30
    valuation_comfort: float  # 0-20
    governance_quality: float  # 0-20
    promoter_quality: float  # 0-15
    business_moat: float  # 0-10
    industry_tailwinds: float  # 0-5

    # Veto metrics
    promoter_pledge_percent: float
    rpt_concentration_percent: float
    litigation_to_networth_percent: float
    post_issue_promoter_holding: float

    # Verdict
    verdict: str  # SUBSCRIBE/AVOID
    verdict_type: str  # LISTING GAINS, LONG TERM, EXPENSIVE, TOXIC
    veto_flags: List[str]

# Scoring Rubrics
FINANCIAL_HEALTH_RUBRIC = {
    "roe_above_20": 10,      # 30% weight of section
    "roe_15_to_20": 8,
    "roe_10_to_15": 6,
    "roe_below_10": 3
}
# Penalty: -2 if CFO/EBITDA < 50%

VALUATION_COMFORT_RUBRIC = {
    "discount_gt_10": 10,    # >10% discount to peers
    "fair_value": 8,         # Within ±10% of peers
    "premium_lt_20": 6,      # <20% premium
    "premium_gt_20": 3       # >20% premium
}

GOVERNANCE_RUBRIC = {
    "base_score": 10,        # Start at 10
    "independent_directors_gt_50": 0,  # No deduction
    "independent_directors_lt_50": -1,
}

GOVERNANCE_DEDUCTIONS = {
    "criminal_case": -3,
    "group_conflict_gt_3": -2,
    "post_ipo_holding_lt_51": -2,
    "ofs_gt_20_pct_stake": -1,
    "independent_directors_lt_50": -1,
    "rpt_revenue_gt_20": -2,
    "pledge_gt_25": -3,
    "pledge_10_to_25": -2,
    "modified_audit_opinion": -2,
    "auditor_resignation": -2
}

# VETO CONDITIONS (Automatic AVOID rating)
VETO_CONDITIONS = [
    "governance_score < 5",
    "promoter_pledge > 25%",
    "litigation > 10% of post-issue net worth",
    "promoter_post_ipo_holding < 40%",
    "criminal_charges_against_promoters",
    "sebi_nclt_proceedings_pending"
]
```

**Implementation Note**: All these dataclasses should be defined in `src/rhp_analyzer/models/` package. Use Pydantic for validation if runtime type checking is needed.

---

### Appendix B: Windows-Specific Setup Notes

#### Python Installation
```cmd
# Verify Python version
python --version

# Create virtual environment
python -m venv .venv

# Activate (Command Prompt)
.venv\Scripts\activate.bat

# Activate (PowerShell)
.venv\Scripts\Activate.ps1
```

#### WeasyPrint GTK Installation
```cmd
# Option 1: MSYS2 (recommended)
# Download from https://www.msys2.org/
# Install GTK3: pacman -S mingw-w64-x86_64-gtk3

# Option 2: Standalone GTK
# Download from https://github.com/nickverschueren/gtk-windows
# Add to PATH: C:\Program Files\GTK3-Runtime\bin
```

#### Tesseract OCR (if needed later)
```cmd
# Download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

---

### Appendix C: Sample Configuration File

```yaml
# config.yaml - RHP Analyzer Configuration

# Paths
paths:
  input_dir: "./data/input"
  output_dir: "./data/output"
  logs_dir: "./logs"
  data_dir: "./data"

# LLM Configuration
llm:
  provider: "huggingface"
  context_model: "Qwen/Qwen2.5-32B-Instruct"
  reasoning_model: "meta-llama/Llama-3.3-70B-Instruct"
  temperature: 0.1
  max_tokens: 4096
  timeout: 120

# Ingestion Settings
ingestion:
  chunk_size: 1000
  chunk_overlap: 100
  min_chunk_size: 200
  batch_size: 32

# Agent Configuration
agents:
  enabled:
    - architect
    - forensic
    - red_flag
    - governance
    - legal
    - summarizer
    - critic
  max_revisions: 2
  parallel_execution: true

# Reporting
reporting:
  formats:
    - markdown
    - pdf
  template: "default"
  include_appendices: true

# Logging
logging:
  level: "INFO"
  console: true
  file: true
  retention_days: 30
```

---

### Appendix D: Sample CLI Commands

```cmd
# Basic analysis
rhp-analyzer analyze "C:\RHPs\company_rhp.pdf"

# With custom output directory
rhp-analyzer analyze "C:\RHPs\company_rhp.pdf" --output-dir "C:\Reports"

# Dry run (validate without processing)
rhp-analyzer analyze "C:\RHPs\company_rhp.pdf" --dry-run

# Verbose logging
rhp-analyzer analyze "C:\RHPs\company_rhp.pdf" --verbose

# Generate report from existing analysis
rhp-analyzer generate company_rhp_20260103 --format both

# Show configuration
rhp-analyzer config show

# Validate configuration
rhp-analyzer config validate

# Q&A Mode - Interactive session
rhp-analyzer query company_rhp_20260103
> What is the company's revenue growth?
> What are the main risk factors?
> Who are the promoters and their shareholding?
> What is the debt-equity ratio?
> exit

# Q&A Mode - Single question
rhp-analyzer query company_rhp_20260103 --question "What is the debt/equity ratio?"

# Q&A Mode - With specific section focus
rhp-analyzer query company_rhp_20260103 --question "What are the key risks?" --section "Risk Factors"

# List processed documents
rhp-analyzer list

# Show document summary
rhp-analyzer info company_rhp_20260103

# Export analysis results to JSON
rhp-analyzer export company_rhp_20260103 --format json --output analysis.json
```

---

### Appendix E: Expected Report Structure

```markdown
# RHP Analysis Report: [Company Name]
## Generated: [Date] | Document Version: 1.0

---

## Investment Committee Memorandum

### Recommendation: [SUBSCRIBE/AVOID] ([LISTING GAINS/LONG TERM/EXPENSIVE/TOXIC])

### Scorecard Summary (0-100)

| Category | Score | Weight | Weighted |
|----------|-------|--------|----------|
| Financial Health | XX/10 | 30% | XX/30 |
| Valuation Comfort | XX/10 | 20% | XX/20 |
| Governance Quality | XX/10 | 20% | XX/20 |
| Promoter Quality | XX/10 | 15% | XX/15 |
| Business Moat | XX/10 | 10% | XX/10 |
| Industry Tailwinds | XX/10 | 5% | XX/5 |
| **Overall Score** | | | **XX/100** |

### Quantitative Guardrails

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Promoter Pledge % | X% | <25% | ✓/⚠️ |
| RPT Concentration | X% | <20% | ✓/⚠️ |
| Litigation/Net Worth | X% | <10% | ✓/⚠️ |
| Post-IPO Promoter Holding | X% | >40% | ✓/⚠️ |
| Day-1 Float | X% | >10% | ✓/⚠️ |

### Veto Flags
- [List any critical issues that trigger automatic AVOID, or "None"]

---

## Scenario Dashboard

### Base Case Projections (3-year forward at Cap Price ₹XX)

| Metric | FY26E | FY27E | FY28E |
|--------|-------|-------|-------|
| Revenue (₹ Cr) | X | X | X |
| EBITDA (₹ Cr) | X | X | X |
| PAT (₹ Cr) | X | X | X |
| Diluted EPS (₹) | X | X | X |
| ROE (%) | X% | X% | X% |
| ROIC (%) | X% | X% | X% |

### Bull/Stress Sensitivity

| Scenario | FY28E EPS | FY28E ROE | Implied P/E |
|----------|-----------|-----------|-------------|
| Bull | ₹X | X% | Xx |
| Base | ₹X | X% | Xx |
| Stress | ₹X | X% | Xx |

---

## Valuation Ladder

| Price Point | P/E | P/B | EV/EBITDA | vs Peer Median |
|-------------|-----|-----|-----------|----------------|
| Floor (₹XX) | Xx | Xx | Xx | -X% discount |
| Cap (₹XX) | Xx | Xx | Xx | +X% premium |
| Fair Value | Xx | Xx | Xx | At par |

### Peer Comparison

| Company | P/E | P/B | ROE | Notes |
|---------|-----|-----|-----|-------|
| [Issuer] | Xx | Xx | X% | At Cap |
| Peer 1 | Xx | Xx | X% | |
| Peer 2 | Xx | Xx | X% | |
| Peer Median | Xx | Xx | X% | |

---

## Executive Summary
- Company overview (2-3 sentences)
- Key investment highlights
- Critical risk factors
- Recommendation summary with price targets

---

## Company Overview
- Business description
- Industry position
- Key products/services
- Geographic presence
- Competitive advantages

---

## Financial Analysis

### Key Metrics Trend

| Metric | FY22 | FY23 | FY24 | Stub | CAGR |
|--------|------|------|------|------|------|
| Revenue (₹ Cr) | X | X | X | X | X% |
| EBITDA (₹ Cr) | X | X | X | X | X% |
| PAT (₹ Cr) | X | X | X | X | X% |
| EBITDA Margin | X% | X% | X% | X% | |
| PAT Margin | X% | X% | X% | X% | |
| ROE | X% | X% | X% | X% | |
| ROCE | X% | X% | X% | X% | |

### Cash Flow Quality

| Metric | FY22 | FY23 | FY24 |
|--------|------|------|------|
| CFO (₹ Cr) | X | X | X |
| FCF (₹ Cr) | X | X | X |
| CFO/EBITDA | X% | X% | X% |
| Capex/Revenue | X% | X% | X% |

### Working Capital Efficiency
- Cash Conversion Cycle: X days (Sector Avg: Y days)
- Receivable Days Trend: [Improving/Worsening]
- Window Dressing Flags: [None / List issues]

---

## Risk Assessment

### High Severity Risks
1. [Risk description] [Page X] ⚠️
2. [Risk description] [Page Y] ⚠️

### Medium Severity Risks
1. [Risk description] [Page X]
2. [Risk description] [Page Y]

### Red Flags Identified
- [ ] [Red flag with severity and citation]

---

## Governance Review

### Board Composition
| Role | Name | Independent | Experience |
|------|------|-------------|------------|
| Chairman | X | Yes/No | X years |
| MD/CEO | X | No | X years |
| ... | ... | ... | ... |

### Promoter Assessment
- Promoter Quality Score: X/10
- Skin-in-the-Game: ₹X Cr at cap price (X% holding)
- Track Record: [Assessment]
- Conflicts of Interest: [None / List]
- Litigation: X cases, ₹X Cr exposure

### Related Party Concerns
- RPT as % of Revenue: X%
- Key RPT relationships: [List]

---

## Legal Review

### Litigation Exposure

| Entity | Criminal | Civil | Tax | Total (₹ Cr) | % of NW |
|--------|----------|-------|-----|--------------|---------|
| Company | X | X | X | X | X% |
| Promoters | X | X | X | X | X% |
| **Total** | X | X | X | X | X% |

### Contingent Liabilities
- Total: ₹X Cr (X% of post-IPO net worth)
- Risk-weighted: ₹X Cr

---

## Investment Thesis

### Bull Case
- [Key positive factors with citations]

### Bear Case
- [Key negative factors with citations]

### Key Risks to Monitor Post-Listing
1. [Risk 1]
2. [Risk 2]
3. [Risk 3]

---

## Appendices
A. Detailed Financial Tables
B. Complete Peer Analysis
C. Full Litigation Details
D. Objects of Issue Breakdown
E. Lock-in Expiry Calendar

---

## Source Citations
[^1]: [Section Name, Page X] - "Quote from RHP"
[^2]: [Section Name, Page Y] - "Quote from RHP"
...
```

---

### Appendix F: Development Timeline Summary

```
Week 1-2:     Phase 1 - Foundation (CLI, logging, config)
Week 3-4:     Phase 2 - Ingestion (PDF, tables, sections, specialized extractors)
Week 4.5:     Phase 2.5 - Financial Modeling (projections, valuations, rulebook)
Week 5-6:     Phase 3 - Storage (vector DB, SQL, chunking, analytics)
Week 7-9:     Phase 4 - Agents (LLM, RAG, all 14 agents)
Week 10-11:   Phase 5 - Reporting (markdown, PDF, Investment Committee Memo)
Week 12-13:   Phase 6 - Polish (testing, docs, release)

Total: ~13 weeks to v1.0.0
```

---

### Appendix G: Troubleshooting Guide

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Package not installed | Run `pip install -e .` |
| PDF extraction fails | Corrupted PDF | Try different PDF, check file |
| Embedding OOM | Large batch size | Reduce `batch_size` in config |
| LLM timeout | Slow API response | Increase `timeout` in config |
| PDF generation fails | Missing GTK | Install GTK3 runtime |
| Rate limit errors | Too many API calls | Enable rate limiting |

---

## Document Complete

This implementation plan covers all phases from project setup to production release. Each milestone has specific subtasks with clear deliverables and verification steps.

**Total Milestones:** 32
**Total Subtasks:** ~120
**Total Agents:** 14

**Agent Summary:**
- Analysis Agents (11): Business Analyst, Industry Analyst, Management, Capital Structure, Forensic Accountant, Red Flag, Governance, Legal, Valuation, Utilization, Promoter Due Diligence
- Validation Agents (2): Self-Critic, Summarizer
- Decision Agent (1): Investment Committee
- On-Demand Agent (1): Q&A Agent

**Next Steps:**
1. Start with Phase 1, Milestone 1.1, Subtask 1.1.1
2. Request implementation for specific subtasks as needed
3. Follow the checkpoint process after each phase

---

*Generated: January 3, 2026*
*Document Version: 1.0*
