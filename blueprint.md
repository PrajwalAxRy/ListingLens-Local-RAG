# RHP Analyzer and Subscription decision intelligence - Technical Design Document
## Version 1.0

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Detailed Architecture](#3-detailed-architecture)
4. [Component Specifications](#4-component-specifications)
5. [Data Models & Schemas](#5-data-models--schemas)
6. [Technology Stack](#6-technology-stack)
7. [Agent System Design](#7-agent-system-design)
8. [Workflow & Processing Pipeline](#8-workflow--processing-pipeline)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Testing Strategy](#10-testing-strategy)
11. [Performance Considerations](#11-performance-considerations)
12. [Error Handling & Logging](#12-error-handling--logging)
13. [Future Enhancements](#13-future-enhancements)

---

## 1. Executive Summary

### 1.1 Project Purpose
A local, AI-powered system to analyze Indian IPO Red Herring Prospectus (RHP) documents and generate comprehensive investment analysis reports in markdown/PDF format. The system uses a multi-agent architecture with state-of-the-art LLMs to extract insights, identify risks, and provide detailed financial analysis.

### 1.2 Key Objectives
- Process 300-500 page RHP documents locally
- Generate detailed analyst-grade reports
- Extract and analyze financial data automatically
- Identify red flags and governance issues
- Provide investment thesis and risk assessment
- Output publication-ready markdown/PDF reports

### 1.3 Core Constraints
- **Environment**: Local system (personal machine)
- **Concurrency**: Single RHP processing at a time
- **Budget**: Unlimited API budget for Hugging Face models
- **User**: Single analyst (personal project)
- **Data Sources**: Self-contained (no external financial APIs)

---

## 2. System Overview

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                    (Command Line Tool)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                            │
│                    (LangGraph State Machine)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Workflow Mgr │  │ Agent Router │  │ State Tracker│         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION TIER                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐│
│  │  PDF Parser      │  │ Table Extractor  │  │ Section Mapper││
│  │ (PyMuPDF/pdfpl.) │  │  (unstructured)  │  │  (Layout-Aware)││
│  └──────────────────┘  └──────────────────┘  └───────────────┘│
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Entity Extractor │  │ Financial Parser │                    │
│  │    (NER Model)   │  │  (Custom Rules)  │                    │
│  └──────────────────┘  └──────────────────┘                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE & INDEXING                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐│
│  │ Vector Database  │  │   SQL Database   │  │  File Storage ││
│  │ (Qdrant/Chroma)  │  │   (SQLite)       │  │  (Local FS)   ││
│  └──────────────────┘  └──────────────────┘  └───────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLIGENCE TIER                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              MULTI-AGENT SYSTEM                            │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │ │
│  │  │  Architect   │ │   Forensic   │ │  Red Flag    │      │ │
│  │  │    Agent     │ │  Accountant  │ │   Agent      │      │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘      │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │ │
│  │  │  Governance  │ │    Legal     │ │ Self-Critic  │      │ │
│  │  │    Agent     │ │    Agent     │ │    Agent     │      │ │
│  │  └──────────────┘ └──────────────┘ └──────────────┘      │ │
│  │  ┌──────────────┐ ┌──────────────┐                       │ │
│  │  │   Q&A Agent  │ │  Summarizer  │                       │ │
│  │  │              │ │    Agent     │                       │ │
│  │  └──────────────┘ └──────────────┘                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              LLM CONFIGURATION                             │ │
│  │  • Context Model: Qwen2.5 32B / Mixtral 8x22B            │ │
│  │  • Reasoning Model: Meta-Llama-3.3-70B-Instruct          │ │
│  │  • Summarizer: Llama-3.2-8B-Instruct (local)             │ │
│  │  • Embeddings: gte-large-en-v1.5 / nomic-embed-text-v1.5 │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REPORT GENERATION                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐│
│  │ Markdown Builder │  │  PDF Generator   │  │ Template Mgr  ││
│  │                  │  │  (WeasyPrint)    │  │               ││
│  └──────────────────┘  └──────────────────┘  └───────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Processing Flow

```
RHP PDF Input
    ↓
[Phase 1: Ingestion] → Parse PDF, Extract Tables, Map Sections
    ↓
[Phase 2: Preprocessing] → Entity Extraction, Financial Parsing
    ↓
[Phase 3: Chunking] → Semantic Chunking, Vectorization
    ↓
[Phase 4: Analysis] → Multi-Agent Processing
    ↓
[Phase 5: Critique] → Self-Critic Review & Verification
    ↓
[Phase 6: Report Assembly] → Generate Markdown/PDF
    ↓
Final Report Output
```

---

## 3. Detailed Architecture

### 3.1 Ingestion Tier

#### 3.1.1 PDF Processing Module
**Purpose**: Extract text, images, and structure from RHP PDF

**Components**:
- **Primary Parser**: PyMuPDF (fitz)
  - Fast, handles both digital and scanned PDFs
  - Extracts text with position coordinates
  - Handles embedded images

- **Fallback Parser**: pdfplumber
  - Better table detection
  - More accurate layout preservation
  - Used when PyMuPDF struggles with complex layouts

- **OCR Engine**: Tesseract (via pytesseract)
  - For scanned/image-based pages
  - Language: English + Hindi (Indian RHPs may have Hindi text)

**Implementation Details**:
```python
class PDFProcessor:
    """
    Handles PDF parsing with multiple strategies
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.metadata = {}
        self.pages = []

    def extract_with_pymupdf(self) -> List[Page]:
        """Primary extraction method"""
        pass

    def extract_with_pdfplumber(self) -> List[Page]:
        """Fallback for complex layouts"""
        pass

    def detect_scanned_pages(self) -> List[int]:
        """Identify pages needing OCR"""
        pass

    def apply_ocr(self, page_num: int) -> str:
        """Apply OCR to scanned pages"""
        pass
```

#### 3.1.2 Table Extraction Module
**Purpose**: Extract financial tables and structured data

**Strategy**:
- Use `unstructured.io` for initial detection
- Use `pdfplumber` for precise extraction
- Use `camelot-py` as fallback for complex tables

**Table Types to Extract**:
1. Financial Statements (P&L, Balance Sheet, Cash Flow)
2. Shareholding Pattern
3. Object of the Issue
4. Use of Proceeds
5. Key Metrics & Ratios
6. Risk Factors Summary
7. Promoter Details
8. Related Party Transactions
9. Basis for Issue Price (Peer Comparison)
10. Contingent Liabilities
11. Capital Structure (WACA, Pre-IPO placements)
12. Top Customers & Suppliers (Concentration)

**Implementation**:
```python
class TableExtractor:
    """
    Multi-strategy table extraction
    """
    def __init__(self):
        self.strategies = ['unstructured', 'pdfplumber', 'camelot']

    def extract_tables(self, pdf_path: str, page_range: tuple) -> List[Table]:
        """Extract tables with confidence scores"""
        pass

    def classify_table(self, table: Table) -> str:
        """Classify table type using heuristics + LLM"""
        pass

    def parse_financial_statement(self, table: Table) -> FinancialData:
        """Parse specific financial statement types"""
        pass
```

#### 3.1.3 Section Mapping Module
**Purpose**: Build hierarchical structure of RHP document

**Approach**:
- Analyze font sizes and styles to identify headers
- Use regex patterns for common section headers
- Build tree structure of sections

**Key Sections to Identify**:
1. Summary of Prospectus
2. Risk Factors
3. Business Overview
4. Financial Statements
5. Management & Governance
6. Legal & Regulatory
7. Objects of the Issue
8. Issue Details
9. Promoter & Shareholders

**Implementation**:
```python
class SectionMapper:
    """
    Creates hierarchical document structure
    """
    SECTION_PATTERNS = {
        'risk': r'RISK\s+FACTORS?',
        'business': r'(BUSINESS|OUR\s+COMPANY)',
        'financial': r'FINANCIAL\s+(STATEMENTS?|INFORMATION)',
        # ... more patterns
    }

    def build_hierarchy(self, pages: List[Page]) -> SectionTree:
        """Build document section tree"""
        pass

    def extract_section_boundaries(self) -> Dict[str, Tuple[int, int]]:
        """Get page ranges for each section"""
        pass
```

#### 3.1.4 Entity Extraction Module
**Purpose**: Extract key entities (companies, people, locations)

**Models to Use**:
- **Primary**: `dslim/bert-base-NER` (Hugging Face)
- **Alternative**: `xlm-roberta-large-finetuned-conll03-english`
- **Indian Entities**: Fine-tune on Indian company names

**Entities to Extract**:
- Company name and subsidiaries
- Promoter names
- Directors and KMPs
- Underwriters and intermediaries
- Legal advisors
- Auditors
- Locations (office, manufacturing)

**Implementation**:
```python
class EntityExtractor:
    """
    NER-based entity extraction
    """
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.ner_pipeline = pipeline("ner", model=model_name)
        self.entity_cache = {}

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract and categorize entities"""
        pass

    def resolve_coreferences(self, entities: List[Entity]) -> List[Entity]:
        """Resolve entity mentions to canonical forms"""
        pass
```

#### 3.1.5 Financial Parser Module
**Purpose**: Extract financial metrics and calculate ratios

**Metrics to Extract**:
- Revenue, EBITDA, PAT (3-5 years)
- Total Assets, Equity, Debt
- ROE, ROCE, Debt/Equity
- Working Capital metrics
- Growth rates (CAGR)

**Implementation**:
```python
class FinancialParser:
    """
    Extracts and computes financial metrics
    """
    def parse_financial_statements(self, tables: List[Table]) -> FinancialData:
        """Parse P&L, Balance Sheet, Cash Flow"""
        pass

    def calculate_ratios(self, financials: FinancialData) -> Dict[str, float]:
        """Calculate key financial ratios"""
        ratios = {
            'roe': self.calc_roe(financials),
            'roce': self.calc_roce(financials),
            'debt_equity': self.calc_debt_equity(financials),
            'current_ratio': self.calc_current_ratio(financials),
            # New Age Metrics
            'contribution_margin': self.calc_contribution_margin(financials),
            'cac_ltv': self.calc_cac_ltv(financials),
            # Risk Metrics
            'contingent_liabilities_to_nw': self.calc_contingent_liabilities_ratio(financials),
        }
        return ratios

    def detect_divergences(self, financials: FinancialData) -> List[str]:
        """
        Identify 'Window Dressing' signals:
        1. Revenue growth vs Receivables growth (Channel Stuffing)
        2. EBITDA vs CFO (Paper profits)
        """
        pass

    def detect_trends(self, time_series: Dict[str, List[float]]) -> TrendAnalysis:
        """Identify growth trends and anomalies"""
        pass
```

### 3.2 Storage & Indexing Layer

#### 3.2.1 Vector Database (Qdrant)
**Purpose**: Semantic search across document chunks

**Why Qdrant**:
- Easy local setup (Docker or standalone)
- Excellent filtering capabilities
- Good performance for single-user scenarios
- Persistent storage

**Schema**:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class VectorStore:
    """
    Manages vector storage for semantic search
    """
    def __init__(self, collection_name: str = "rhp_chunks"):
        self.client = QdrantClient(path="./qdrant_storage")
        self.collection_name = collection_name

    def create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Add document chunks with embeddings"""
        points = [
            PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "section": chunk.section,
                    "page_num": chunk.page_num,
                    "chunk_type": chunk.type,  # 'narrative' or 'table'
                    "metadata": chunk.metadata
                }
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: List[float], filters: Dict, top_k: int = 5):
        """Semantic search with filters"""
        pass
```

**Chunking Strategy**:
- **Chunk Size**: 500-1500 tokens
- **Overlap**: 100 tokens
- **Chunking Method**: Semantic chunking (split on section boundaries)
- **Separate Storage**: Narrative text vs. tables

#### 3.2.2 SQL Database (SQLite)
**Purpose**: Store structured data and metadata

**Schema**:
```sql
-- Document metadata
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    filename TEXT,
    company_name TEXT,
    upload_date TIMESTAMP,
    total_pages INTEGER,
    processing_status TEXT,
    issue_size REAL,
    price_band TEXT
);

-- Extracted sections
CREATE TABLE sections (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    section_name TEXT,
    start_page INTEGER,
    end_page INTEGER,
    word_count INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

-- Financial data
CREATE TABLE financial_data (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    fiscal_year TEXT,
    revenue REAL,
    ebitda REAL,
    pat REAL,
    total_assets REAL,
    total_equity REAL,
    total_debt REAL,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

-- Extracted entities
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    entity_type TEXT,  -- 'company', 'person', 'location'
    entity_name TEXT,
    mentions INTEGER,
    context TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

-- Risk factors
CREATE TABLE risk_factors (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    category TEXT,  -- 'operational', 'financial', 'regulatory', etc.
    description TEXT,
    severity TEXT,  -- 'high', 'medium', 'low'
    page_reference INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

-- Agent analysis results
CREATE TABLE agent_outputs (
    id INTEGER PRIMARY KEY,
    document_id INTEGER,
    agent_name TEXT,
    analysis_result TEXT,  -- JSON blob
    confidence_score REAL,
    created_at TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);
```

#### 3.2.3 File Storage
**Purpose**: Store original PDFs and generated reports

**Structure**:
```
./data/
├── input/
│   └── {document_id}/
│       ├── original.pdf
│       └── metadata.json
├── processed/
│   └── {document_id}/
│       ├── text/
│       │   ├── page_001.txt
│       │   ├── page_002.txt
│       │   └── ...
│       ├── tables/
│       │   ├── table_001.csv
│       │   ├── table_002.csv
│       │   └── ...
│       ├── images/
│       │   └── extracted_images/
│       └── chunks/
│           └── chunks.jsonl
├── embeddings/
│   └── {document_id}/
│       └── embeddings.npy
└── outputs/
    └── {document_id}/
        ├── report.md
        ├── report.pdf
        └── analysis_summary.json
```

### 3.3 Orchestration Layer (LangGraph)

#### 3.3.1 State Machine Design

**State Definition**:
```python
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END

class AnalysisState(TypedDict):
    """
    Global state maintained throughout processing
    """
    # Document info
    document_id: str
    pdf_path: str

    # Ingestion outputs
    pages: List[Dict]
    sections: Dict[str, Dict]
    tables: List[Dict]
    entities: Dict[str, List[str]]
    financial_data: Dict[str, Dict]

    # Vector search
    chunks: List[Dict]
    embeddings: Optional[List[List[float]]]

    # Agent outputs
    architect_analysis: Optional[str]
    valuation_analysis: Optional[str]
    utilization_analysis: Optional[str]
    forensic_analysis: Optional[str]
    red_flag_analysis: Optional[str]
    governance_analysis: Optional[str]
    legal_analysis: Optional[str]

    # Critique & verification
    self_critique: Optional[str]
    verification_results: Dict[str, bool]

    # Final output
    final_report: Optional[str]

    # Status & errors
    current_phase: str
    errors: List[str]
    warnings: List[str]
```

**Workflow Graph**:
```python
class RHPAnalysisWorkflow:
    """
    LangGraph-based orchestration
    """
    def __init__(self):
        self.graph = StateGraph(AnalysisState)
        self._build_graph()

    def _build_graph(self):
        # Define nodes
        self.graph.add_node("ingest_pdf", self.ingest_pdf_node)
        self.graph.add_node("extract_tables", self.extract_tables_node)
        self.graph.add_node("map_sections", self.map_sections_node)
        self.graph.add_node("extract_entities", self.extract_entities_node)
        self.graph.add_node("parse_financials", self.parse_financials_node)
        self.graph.add_node("create_chunks", self.create_chunks_node)
        self.graph.add_node("generate_embeddings", self.generate_embeddings_node)
        self.graph.add_node("architect_agent", self.architect_agent_node)
        self.graph.add_node("forensic_agent", self.forensic_agent_node)
        self.graph.add_node("red_flag_agent", self.red_flag_agent_node)
        self.graph.add_node("valuation_agent", self.valuation_agent_node)
        self.graph.add_node("utilization_agent", self.utilization_agent_node)
        self.graph.add_node("governance_agent", self.governance_agent_node)
        self.graph.add_node("legal_agent", self.legal_agent_node)
        self.graph.add_node("self_critic_agent", self.self_critic_agent_node)
        self.graph.add_node("generate_report", self.generate_report_node)

        # Define edges (workflow sequence)
        self.graph.set_entry_point("ingest_pdf")
        self.graph.add_edge("ingest_pdf", "extract_tables")
        self.graph.add_edge("extract_tables", "map_sections")
        self.graph.add_edge("map_sections", "extract_entities")
        self.graph.add_edge("extract_entities", "parse_financials")
        self.graph.add_edge("parse_financials", "create_chunks")
        self.graph.add_edge("create_chunks", "generate_embeddings")
        self.graph.add_edge("generate_embeddings", "architect_agent")

        # Parallel agent execution (can be sequential too)
        self.graph.add_edge("architect_agent", "forensic_agent")
        self.graph.add_edge("forensic_agent", "red_flag_agent")
        self.graph.add_edge("legal_agent", "valuation_agent")
        self.graph.add_edge("valuation_agent", "utilization_agent")

        # Critique and finalization
        self.graph.add_edge("utilization
        # Critique and finalization
        self.graph.add_edge("legal_agent", "self_critic_agent")
        self.graph.add_edge("self_critic_agent", "generate_report")
        self.graph.add_edge("generate_report", END)

        self.compiled_graph = self.graph.compile()

    def run(self, pdf_path: str) -> Dict:
        """Execute the workflow"""
        initial_state = {
            "document_id": self._generate_doc_id(),
            "pdf_path": pdf_path,
            "current_phase": "initialization",
            "errors": [],
            "warnings": []
        }

        result = self.compiled_graph.invoke(initial_state)
        return result
```

---

## 4. Component Specifications

### 4.1 Embedding Model

**Model Selection**: `nomic-ai/nomic-embed-text-v1.5`
- **Reasoning**: Optimized for long documents, good performance
- **Alternative**: `Alibaba-NLP/gte-large-en-v1.5`
- **Context Length**: 8192 tokens
- **Dimension**: 768 or 1024

**Usage**:
```python
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """
    Generates embeddings for document chunks
    """
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model = SentenceTransformer(model_name)

    def generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with batching"""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()
```

### 4.2 LLM Configuration

#### 4.2.1 Context Model (Large Context Window)
**Model**: `Qwen/Qwen2.5-32B-Instruct` or `mistralai/Mixtral-8x22B-Instruct-v0.1`
- **Purpose**: Handle large context for full section analysis
- **Context Window**: 32K-64K tokens
- **API**: Hugging Face Inference API

**Configuration**:
```python
from huggingface_hub import InferenceClient

class ContextLLM:
    """
    Large context window model for section-level analysis
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-32B-Instruct"):
        self.client = InferenceClient(token=os.getenv("HF_TOKEN"))
        self.model_id = model_id

    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = self.client.text_generation(
            prompt,
            model=self.model_id,
            max_new_tokens=max_tokens,
            temperature=0.3,  # Low temperature for factual analysis
            top_p=0.9
        )
        return response
```

#### 4.2.2 Reasoning Model (Deep Analysis)
**Model**: `meta-llama/Llama-3.3-70B-Instruct`
- **Purpose**: Complex reasoning, red flag identification
- **Context Window**: 8K-128K tokens (depending on variant)
- **API**: Hugging Face Inference API

**Configuration**:
```python
class ReasoningLLM:
    """
    Advanced reasoning model for critical analysis
    """
    def __init__(self, model_id: str = "meta-llama/Llama-3.3-70B-Instruct"):
        self.client = InferenceClient(token=os.getenv("HF_TOKEN"))
        self.model_id = model_id

    def analyze(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=4096,
            temperature=0.2  # Very low for analytical tasks
        )
        return response.choices[0].message.content
```

#### 4.2.3 Local Summarizer Model
**Model**: `meta-llama/Llama-3.2-8B-Instruct` (local)
- **Purpose**: Fast chunk-level summaries
- **Deployment**: Run locally via `transformers` or `ollama`
- **Context Window**: 4K tokens

**Why Local**:
- Reduce API costs for repetitive summarization
- Faster for batch operations
- No network dependency

**Setup**:
```bash
# Option 1: Using Ollama (recommended for ease)
ollama pull llama3.2:8b-instruct-fp16

# Option 2: Using transformers
pip install transformers torch accelerate
```

```python
class LocalSummarizer:
    """
    Local model for fast summarization
    """
    def __init__(self, use_ollama: bool = True):
        if use_ollama:
            self.client = ollama.Client()
            self.model = "llama3.2:8b-instruct-fp16"
        else:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-generation",
                model="meta-llama/Llama-3.2-8B-Instruct",
                device="cuda"  # or "cpu"
            )

    def summarize(self, text: str, max_length: int = 200) -> str:
        if hasattr(self, 'client'):
            response = self.client.generate(
                model=self.model,
                prompt=f"Summarize the following in {max_length} words:\n\n{text}"
            )
            return response['response']
        else:
            # Use transformers pipeline
            pass
```

### 4.3 RHP-Derived Financial Modeling Engine

**Purpose**: Convert historical disclosures from the RHP (restated consolidated and standalone statements, capitalization tables, object deployment schedules) into forward-looking projections that tie directly into the decision scorecard.

**Key Responsibilities**:
- Normalize historical figures (convert crore/lakh units, restatement adjustments, carve-outs) into machine-usable time series.
- Generate at least three projection scenarios (Base, Bull, Stress) using rule-based drivers sourced only from RHP data (management guidance, capacity additions, order book, working-capital policies).
- Compute post-issue metrics such as diluted EPS, ROE, ROIC, net-debt/EBITDA, interest coverage, and cash conversion cycle under each scenario.
- Run dilution math: reconcile pre-issue share capital, fresh issue shares, OFS, and ESOP overhang to output post-issue outstanding shares and promoter holding.
- Surface sensitivities to price band (floor vs cap) and to deployment delays in Objects of the Issue.

**Implementation Hooks**:
```python
class ProjectionEngine:
    """Builds scenario-based financial forecasts from RHP disclosures"""
    def __init__(self, financial_parser: FinancialParser):
        self.financial_parser = financial_parser

    def build_scenarios(self, historicals: List[FinancialData], guidance: Dict) -> List["ProjectionScenario"]:
        # Use CAGR bands, capacity utilization, order book cover as drivers
        pass

    def compute_post_issue_metrics(self, ipo_details: IPODetails, scenario: "ProjectionScenario") -> Dict[str, float]:
        # Diluted EPS, ROE, ROIC, net-debt/EBITDA
        pass

    def run_sensitivities(self, scenario: "ProjectionScenario", price_band: Tuple[int, int]) -> Dict[str, Dict]:
        # Price-band, deployment delay, margin compression
        pass
```

### 4.4 Valuation Normalization & Sensitivity Module

**Need**: RHP “Basis for Issue Price” tables often cherry-pick peers. This module normalizes every peer disclosed in the RHP (including those from Industry Overview) by:
- Extracting peer financials directly from the RHP tables and adjusting for fiscal year, accounting standard (Ind-AS vs IFRS), and consolidated/standalone differences.
- Calculating comparable multiples (P/E, P/B, EV/EBITDA, PEG) on a consistent basis and highlighting “missing peers” referenced elsewhere in the RHP but excluded from pricing justification.
- Producing valuation ladders at floor and cap prices plus a premium/discount vs peer median.
- Linking promoter WACA, pre-IPO placements, and OFS pricing from the Capital Structure section to quantify mark-ups.
- Feeding scenario output from the Projection Engine to compute implied upside/downside at listing, six-month, and long-term horizons.

### 4.5 Governance & Forensic Rulebook Service

**Purpose**: Enforce SEBI-aligned guardrails using only RHP inputs so red flags are objective.

**Rule Families** (configurable YAML):
- **Shareholder Skin-in-the-Game**: promoter post-issue holding < 51% or OFS > fresh issue triggers critical alert.
- **Pledge & Encumbrance**: any disclosed pledge > 0% triggers major alert; > 25% triggers critical.
- **Related-Party Concentration**: RPT revenue or expense > 20% of totals.
- **Audit Quality**: auditor resignations, CARO/NCF qualifications, modified opinions.
- **Litigation Materiality**: total promoter/company litigation > 10% of post-issue net worth.
- **Working Capital Stress**: receivable days growth exceeding revenue CAGR by > 10 percentage points.

The rulebook emits structured findings (severity, rationale, citation) consumed by Governance, Forensic, and Red Flag agents.

### 4.6 Risk & Litigation Quantification Module

**Inputs**: “Outstanding Litigation”, “Material Developments”, “Contingent Liabilities” sections of the RHP.

**Outputs**:
- Aggregated litigation tables split by entity (Company, Promoters, Directors, Subsidiaries) with counts, amounts, and % of net worth/post-issue equity.
- Timeline flags for matters with hearing dates within 12 months of listing.
- Mapping of contingent liabilities to business segments (e.g., GST, customs, environmental) and probability buckets.
- Integration with Objects of the Issue to see if proceeds earmarked for settlement.

### 4.7 Citation & Audit Trail Manager

Institutional clients need defensible sourcing. This component enforces:
- Machine-readable citations for every numerical or qualitative claim (section, page, paragraph anchor).
- Validation hooks in Self-Critic agent to reject outputs lacking citations.
- Report footnotes auto-generated from the citation store.

```python
class CitationManager:
    def __init__(self):
        self._store: Dict[str, CitationRecord] = {}

    def attach(self, claim_id: str, citation: CitationRecord):
        self._store[claim_id] = citation

    def export_markdown(self) -> str:
        # Footnote block for report
        pass
```

### 4.8 Promoter Extractor Module

**Purpose**: Extract comprehensive promoter profiles from RHP for due diligence.

**RHP Sections to Parse**:
- "Our Promoters" / "Our Promoter and Promoter Group"
- "Common Pursuits and Interests of Promoters"
- "Interest of Promoters"
- "Payment of Benefits to Promoters"
- "Other Directorships of Promoters"
- "Litigation involving Promoters"
- "Related Party Transactions"

**Implementation**:
```python
class PromoterExtractor:
    """
    Extracts promoter details from multiple RHP sections
    """
    def __init__(self, vector_store: VectorStore, citation_mgr: CitationManager):
        self.vector_store = vector_store
        self.citation_mgr = citation_mgr

    def extract_promoters(self, state: AnalysisState) -> List[PromoterDossier]:
        """Extract all promoters with comprehensive details"""
        promoters = []

        # 1. Get basic promoter profiles from "Our Promoters" section
        promoter_context = self._retrieve_section("Our Promoters", state)
        basic_profiles = self._parse_promoter_profiles(promoter_context)

        # 2. Extract directorships
        directorships = self._extract_directorships(state)

        # 3. Extract common pursuits (conflict of interest check)
        common_pursuits = self._extract_common_pursuits(state)

        # 4. Extract financial interests
        financial_interests = self._extract_financial_interests(state)

        # 5. Extract promoter-specific litigation
        litigation = self._extract_promoter_litigation(state)

        # 6. Calculate skin-in-game
        shareholding = self._extract_shareholding(state)

        # Merge all data
        for promoter_name in basic_profiles:
            dossier = PromoterDossier(
                name=promoter_name,
                din=basic_profiles[promoter_name].get('din'),
                age=basic_profiles[promoter_name].get('age'),
                qualification=basic_profiles[promoter_name].get('qualification'),
                experience_years=basic_profiles[promoter_name].get('experience'),
                other_directorships=directorships.get(promoter_name, []),
                group_companies_in_same_line=common_pursuits.get(promoter_name, []),
                shareholding_pre_ipo=shareholding.get(promoter_name, {}).get('pre', 0),
                shareholding_post_ipo=shareholding.get(promoter_name, {}).get('post', 0),
                litigation_as_defendant=litigation.get(promoter_name, []),
                loans_from_company=financial_interests.get(promoter_name, {}).get('loans', 0),
                remuneration_last_3_years=financial_interests.get(promoter_name, {}).get('remuneration', [])
            )
            promoters.append(dossier)

        return promoters

    def _retrieve_section(self, section_name: str, state: AnalysisState) -> List[Chunk]:
        """Retrieve chunks from specific RHP section"""
        query_vector = self._embed_query(section_name)
        results = self.vector_store.search(
            query_vector=query_vector,
            filters={"section": section_name},
            top_k=20
        )
        return results

    def _parse_promoter_profiles(self, context: List[Chunk]) -> Dict:
        """Use LLM to extract structured promoter profiles"""
        # Implementation using LLM to parse narrative text
        pass

    def _extract_directorships(self, state: AnalysisState) -> Dict[str, List[str]]:
        """Extract other directorships for each promoter"""
        # Parse "Other Directorships" tables
        pass

    def _extract_common_pursuits(self, state: AnalysisState) -> Dict[str, List[str]]:
        """Extract group companies in same line of business"""
        # Parse "Common Pursuits" section
        pass

    def _extract_financial_interests(self, state: AnalysisState) -> Dict:
        """Extract loans, remuneration, guarantees"""
        # Parse "Interest of Promoters", "Payment of Benefits"
        pass

    def _extract_promoter_litigation(self, state: AnalysisState) -> Dict[str, List[RiskExposure]]:
        """Extract litigation specific to promoters"""
        # Parse litigation tables, filter by entity = "Promoter"
        pass

    def _extract_shareholding(self, state: AnalysisState) -> Dict:
        """Extract pre/post IPO shareholding"""
        # Parse "Shareholding Pattern" tables
        pass
```

### 4.9 Pre-IPO Investor Analyzer Module

**Purpose**: Analyze entry prices, returns, and lock-in schedules for all pre-IPO investors.

**RHP Sections to Parse**:
- "Capital Structure" → "Build-up of Promoters' Shareholding"
- "Capital Structure" → "History of Equity Share Capital"
- "Lock-in" section
- "Offer for Sale" details

**Implementation**:
```python
class PreIPOInvestorAnalyzer:
    """
    Analyzes pre-IPO investors and their exit economics
    """
    def __init__(self, financial_parser: FinancialParser, citation_mgr: CitationManager):
        self.financial_parser = financial_parser
        self.citation_mgr = citation_mgr

    def analyze_investors(self, state: AnalysisState, ipo_details: IPODetails) -> List[PreIPOInvestor]:
        """Analyze all pre-IPO investors"""
        investors = []

        # 1. Parse capital structure history
        cap_history = self._parse_capital_structure_history(state)

        # 2. Parse OFS details
        ofs_details = self._parse_ofs_details(state)

        # 3. Parse lock-in information
        lock_in_schedule = self._parse_lock_in_schedule(state)

        # 4. Extract price band
        floor_price = ipo_details.price_band_floor
        cap_price = ipo_details.price_band_cap

        # 5. Build investor profiles
        for investor_name, entry_records in cap_history.items():
            # Calculate weighted average entry price
            total_investment = sum(r['shares'] * r['price'] for r in entry_records)
            total_shares = sum(r['shares'] for r in entry_records)
            avg_entry_price = total_investment / total_shares if total_shares > 0 else 0

            # Get earliest entry date
            entry_date = min(r['date'] for r in entry_records)
            holding_period = self._calculate_holding_period_months(entry_date)

            # Calculate returns
            return_multiple_floor = floor_price / avg_entry_price if avg_entry_price > 0 else 0
            return_multiple_cap = cap_price / avg_entry_price if avg_entry_price > 0 else 0

            # Calculate IRR
            irr_floor = self._calculate_irr(avg_entry_price, floor_price, holding_period)
            irr_cap = self._calculate_irr(avg_entry_price, cap_price, holding_period)

            # Get OFS participation
            ofs_shares = ofs_details.get(investor_name, {}).get('shares', 0)
            ofs_amount = ofs_shares * cap_price / 10000000  # Convert to Cr

            # Get lock-in details
            lock_in_info = lock_in_schedule.get(investor_name, {})

            investor = PreIPOInvestor(
                name=investor_name,
                category=self._classify_investor(investor_name),
                entry_date=entry_date,
                entry_price=avg_entry_price,
                shares_held_pre_ipo=total_shares,
                shares_selling_via_ofs=ofs_shares,
                ofs_amount=ofs_amount,
                implied_return_multiple_at_floor=return_multiple_floor,
                implied_return_multiple_at_cap=return_multiple_cap,
                implied_irr_at_floor=irr_floor,
                implied_irr_at_cap=irr_cap,
                holding_period_months=holding_period,
                lock_in_period=lock_in_info.get('period', ''),
                lock_in_expiry_date=lock_in_info.get('expiry', None),
                shares_locked=lock_in_info.get('locked_shares', 0)
            )
            investors.append(investor)

        return investors

    def _parse_capital_structure_history(self, state: AnalysisState) -> Dict:
        """Parse equity history tables"""
        # Extract "History of Equity Share Capital" table
        pass

    def _parse_ofs_details(self, state: AnalysisState) -> Dict:
        """Parse OFS seller details"""
        # Extract "Offer for Sale" breakdown
        pass

    def _parse_lock_in_schedule(self, state: AnalysisState) -> Dict:
        """Parse lock-in requirements"""
        # Extract lock-in table
        pass

    def _calculate_holding_period_months(self, entry_date: str) -> int:
        """Calculate months between entry and IPO"""
        from datetime import datetime
        entry = datetime.strptime(entry_date, "%Y-%m-%d")
        now = datetime.now()
        return (now.year - entry.year) * 12 + now.month - entry.month

    def _calculate_irr(self, entry_price: float, exit_price: float, months: int) -> float:
        """Calculate annualized IRR"""
        if months == 0 or entry_price == 0:
            return 0.0
        years = months / 12
        return ((exit_price / entry_price) ** (1 / years) - 1) * 100

    def _classify_investor(self, name: str) -> str:
        """Classify investor type"""
        name_lower = name.lower()
        if any(kw in name_lower for kw in ['fund', 'capital', 'partners', 'ventures']):
            return "PE/VC"
        elif any(kw in name_lower for kw in ['trust', 'esop']):
            return "ESOP Trust"
        elif any(kw in name_lower for kw in ['promoter']):
            return "Promoter"
        else:
            return "Other"
```

### 4.10 Float Calculator Module

**Purpose**: Calculate tradeable float at various post-listing milestones.

**Implementation**:
```python
class FloatCalculator:
    """
    Calculates free float at different time horizons
    """
    def calculate_float_analysis(
        self,
        ipo_details: IPODetails,
        pre_ipo_investors: List[PreIPOInvestor],
        promoter_dossiers: List[PromoterDossier]
    ) -> FloatAnalysis:
        """Calculate comprehensive float analysis"""

        # Get share capital structure
        total_shares_post_issue = ipo_details.shares_post_issue
        fresh_issue_shares = ipo_details.fresh_issue_shares

        # Calculate locked shares
        promoter_locked = sum(p.shareholding_post_ipo * total_shares_post_issue / 100
                             for p in promoter_dossiers)

        anchor_locked = ipo_details.anchor_quota_shares if hasattr(ipo_details, 'anchor_quota_shares') else 0

        pre_ipo_locked = sum(inv.shares_locked for inv in pre_ipo_investors)

        # Day 1 float (only retail + unlocked portion)
        day_1_free = total_shares_post_issue - promoter_locked - anchor_locked - pre_ipo_locked
        day_1_free_percent = (day_1_free / total_shares_post_issue) * 100

        # Day 90 (anchor unlocks)
        day_90_free = day_1_free + anchor_locked
        day_90_free_percent = (day_90_free / total_shares_post_issue) * 100

        # Build lock-in calendar
        lock_in_calendar = self._build_lock_in_calendar(
            promoter_dossiers,
            pre_ipo_investors,
            anchor_locked,
            total_shares_post_issue
        )

        # Retail quota shares
        retail_shares = ipo_details.retail_quota_shares if hasattr(ipo_details, 'retail_quota_shares') else 0
        retail_percent = (retail_shares / total_shares_post_issue) * 100

        # Implied daily volume (assume 250 trading days)
        implied_daily = day_1_free / 250

        float_analysis = FloatAnalysis(
            total_shares_post_issue=total_shares_post_issue,
            fresh_issue_shares=fresh_issue_shares,
            promoter_locked_shares=int(promoter_locked),
            promoter_locked_percent=(promoter_locked / total_shares_post_issue) * 100,
            anchor_locked_shares=int(anchor_locked),
            pre_ipo_locked_shares=int(pre_ipo_locked),
            day_1_free_float_shares=int(day_1_free),
            day_1_free_float_percent=day_1_free_percent,
            day_90_free_float_percent=day_90_free_percent,
            retail_quota_shares=int(retail_shares),
            retail_quota_percent=retail_percent,
            implied_daily_volume=implied_daily,
            lock_in_calendar=lock_in_calendar
        )

        return float_analysis

    def _build_lock_in_calendar(self, promoters, investors, anchor_shares, total_shares) -> List[Dict]:
        """Build timeline of lock-in expiries"""
        calendar = []

        # Anchor unlock at 90 days
        if anchor_shares > 0:
            calendar.append({
                'date': 'Day 90',
                'shares_unlocking': int(anchor_shares),
                'investor': 'Anchor Investors',
                'percent_of_float': (anchor_shares / total_shares) * 100
            })

        # Pre-IPO investor unlocks
        for inv in investors:
            if inv.shares_locked > 0 and inv.lock_in_expiry_date:
                calendar.append({
                    'date': inv.lock_in_expiry_date,
                    'shares_unlocking': inv.shares_locked,
                    'investor': inv.name,
                    'percent_of_float': (inv.shares_locked / total_shares) * 100
                })

        # Sort by date
        calendar.sort(key=lambda x: x['date'])
        return calendar
```

### 4.11 Order Book Analyzer Module

**Purpose**: Extract and analyze order book for B2B/EPC/Defense companies.

**Applicable Sectors**: EPC, Defense, Infrastructure, IT Services, Capital Goods

**Implementation**:
```python
class OrderBookAnalyzer:
    """
    Analyzes order book disclosure in RHP
    """
    def __init__(self, vector_store: VectorStore, citation_mgr: CitationManager):
        self.vector_store = vector_store
        self.citation_mgr = citation_mgr

    def analyze_order_book(self, state: AnalysisState, sector: str) -> OrderBookAnalysis:
        """Extract and analyze order book if applicable"""

        # Check if sector typically has order book disclosure
        applicable_sectors = ['EPC', 'Defense', 'Infrastructure', 'IT Services',
                            'Capital Goods', 'Engineering', 'Construction']

        if not any(s.lower() in sector.lower() for s in applicable_sectors):
            return OrderBookAnalysis(applicable=False)

        # Search for order book section
        order_book_context = self.vector_store.search(
            query_vector=self._embed_query("order book unexecuted orders outstanding"),
            filters={},
            top_k=10
        )

        if not order_book_context:
            return OrderBookAnalysis(applicable=True)  # Expected but not found

        # Extract order book metrics using LLM
        extracted_data = self._extract_order_book_data(order_book_context)

        # Get latest revenue for ratio calculation
        ltm_revenue = state['financial_data'].get('latest_revenue', 0)

        analysis = OrderBookAnalysis(
            applicable=True,
            total_order_book=extracted_data.get('total', 0),
            order_book_as_of_date=extracted_data.get('as_of_date'),
            order_book_to_ltm_revenue=extracted_data.get('total', 0) / ltm_revenue if ltm_revenue > 0 else 0,
            top_5_orders_value=extracted_data.get('top_5_value', 0),
            executable_in_12_months=extracted_data.get('executable_12m', 0),
            government_orders_percent=extracted_data.get('govt_percent', 0),
            order_book_1yr_ago=extracted_data.get('prior_year', None)
        )

        # Calculate derived metrics
        if analysis.total_order_book > 0:
            analysis.top_5_orders_concentration = (analysis.top_5_orders_value / analysis.total_order_book) * 100
            analysis.executable_in_12_months_percent = (analysis.executable_in_12_months / analysis.total_order_book) * 100

        if analysis.order_book_1yr_ago:
            analysis.order_book_growth_yoy = ((analysis.total_order_book / analysis.order_book_1yr_ago) - 1) * 100

        return analysis

    def _extract_order_book_data(self, context: List[Chunk]) -> Dict:
        """Use LLM to extract structured order book data"""
        # Implementation with LLM
        pass
```

### 4.12 Enhanced Debt Structure Analyzer Module

**Purpose**: Build complete debt maturity waterfall and covenant analysis.

**Implementation**:
```python
class DebtStructureAnalyzer:
    """
    Analyzes debt structure from Indebtedness section
    """
    def analyze_debt(self, state: AnalysisState, objects_analysis: ObjectsOfIssueAnalysis) -> DebtStructure:
        """Comprehensive debt structure analysis"""

        # Extract indebtedness section
        debt_context = self._retrieve_section("Indebtedness", state)

        # Parse debt tables
        debt_data = self._parse_debt_tables(debt_context)

        # Get debt from financial statements
        financials = state['financial_data']
        total_debt = financials.get('latest_total_debt', 0)

        # Calculate post-IPO debt
        debt_repayment = objects_analysis.debt_repayment_amount
        post_ipo_debt = total_debt - debt_repayment

        # Extract interest rates from notes
        interest_rates = self._extract_interest_rates(debt_context)

        # Parse maturity profile
        maturity = self._parse_maturity_profile(debt_data)

        # Extract covenants from material contracts
        covenants = self._extract_covenants(state)

        debt_structure = DebtStructure(
            total_debt=total_debt,
            secured_debt=debt_data.get('secured', 0),
            unsecured_debt=debt_data.get('unsecured', 0),
            short_term_debt=debt_data.get('short_term', 0),
            long_term_debt=debt_data.get('long_term', 0),
            weighted_avg_interest_rate=interest_rates.get('weighted_avg'),
            highest_interest_rate=interest_rates.get('max'),
            lowest_interest_rate=interest_rates.get('min'),
            maturing_within_1_year=maturity.get('0-1yr', 0),
            maturing_1_to_3_years=maturity.get('1-3yr', 0),
            maturing_3_to_5_years=maturity.get('3-5yr', 0),
            maturing_beyond_5_years=maturity.get('5yr+', 0),
            debt_repayment_from_ipo=debt_repayment,
            post_ipo_debt=post_ipo_debt,
            has_financial_covenants=len(covenants) > 0,
            covenant_details=covenants
        )

        # Calculate ratios
        equity = financials.get('latest_equity', 0)
        ebitda = financials.get('latest_ebitda', 0)

        debt_structure.debt_to_equity_pre_ipo = total_debt / equity if equity > 0 else 0
        debt_structure.debt_to_equity_post_ipo = post_ipo_debt / (equity + objects_analysis.fresh_issue) if equity > 0 else 0
        debt_structure.debt_to_ebitda = total_debt / ebitda if ebitda > 0 else 0

        return debt_structure

    def _parse_debt_tables(self, context: List[Chunk]) -> Dict:
        """Parse indebtedness tables"""
        pass

    def _extract_interest_rates(self, context: List[Chunk]) -> Dict:
        """Extract interest rate information"""
        pass

    def _parse_maturity_profile(self, debt_data: Dict) -> Dict:
        """Parse debt maturity schedule"""
        pass

    def _extract_covenants(self, state: AnalysisState) -> List[str]:
        """Extract financial covenants from material contracts"""
        pass
```

### 4.13 Enhanced Cash Flow Analyzer Module

**Purpose**: Calculate FCF, cash burn rate, and capex intensity for all scenarios.

**Implementation**:
```python
class EnhancedCashFlowAnalyzer:
    """
    Enhanced cash flow analysis including FCF, burn rate, capex categorization
    """
    def __init__(self, financial_parser: FinancialParser):
        self.financial_parser = financial_parser

    def analyze_cash_flows(self, financials: List[FinancialData], ipo_details: IPODetails) -> List[CashFlowAnalysis]:
        """Analyze cash flows for each fiscal year"""
        analyses = []

        for financial_data in financials:
            # Extract core cash flows
            cfo = financial_data.cash_flow_operations or 0
            cfi = financial_data.cash_flow_investing or 0
            cff = financial_data.cash_flow_financing or 0

            # Extract capex (usually negative in CFI)
            capex = abs(financial_data.capex or 0)

            # Calculate FCF
            fcf = cfo - capex

            # Get revenue and EBITDA
            revenue = financial_data.revenue or 0
            ebitda = financial_data.ebitda or 0
            pat = financial_data.pat or 0

            # Calculate depreciation
            depreciation = financial_data.depreciation or 0

            # Capex intensity
            capex_to_revenue = (capex / revenue * 100) if revenue > 0 else 0
            capex_to_depreciation = (capex / depreciation) if depreciation > 0 else 0

            # Categorize capex
            maintenance_capex = depreciation  # Proxy
            growth_capex = max(0, capex - depreciation)

            # FCF metrics
            fcf_margin = (fcf / revenue * 100) if revenue > 0 else 0

            # Market cap for FCF yield
            market_cap_floor = ipo_details.market_cap_at_floor if hasattr(ipo_details, 'market_cap_at_floor') else 0
            market_cap_cap = ipo_details.market_cap_at_cap if hasattr(ipo_details, 'market_cap_at_cap') else 0

            fcf_yield_floor = (fcf / market_cap_floor * 100) if market_cap_floor > 0 else None
            fcf_yield_cap = (fcf / market_cap_cap * 100) if market_cap_cap > 0 else None

            # Cash burn analysis (for negative FCF companies)
            is_cash_burning = fcf < 0
            monthly_burn = abs(fcf) / 12 if is_cash_burning else 0

            cash_balance = financial_data.cash_and_equivalents or 0
            runway_months = (cash_balance / monthly_burn) if monthly_burn > 0 else float('inf')

            # Quality metrics
            cfo_to_ebitda = (cfo / ebitda * 100) if ebitda > 0 else 0
            cfo_to_pat = (cfo / pat * 100) if pat > 0 else 0

            # Working capital change
            wc_change = financial_data.working_capital_change or 0
            wc_change_to_revenue = (wc_change / revenue * 100) if revenue > 0 else 0

            analysis = CashFlowAnalysis(
                fiscal_year=financial_data.fiscal_year,
                cfo=cfo,
                cfi=cfi,
                cff=cff,
                net_cash_flow=cfo + cfi + cff,
                capex=capex,
                fcf=fcf,
                fcf_margin=fcf_margin,
                fcf_yield_at_floor=fcf_yield_floor,
                fcf_yield_at_cap=fcf_yield_cap,
                is_cash_burning=is_cash_burning,
                monthly_cash_burn=monthly_burn,
                runway_months=runway_months,
                capex_to_revenue=capex_to_revenue,
                capex_to_depreciation=capex_to_depreciation,
                maintenance_capex_estimate=maintenance_capex,
                growth_capex_estimate=growth_capex,
                wc_change=wc_change,
                wc_change_to_revenue=wc_change_to_revenue,
                cfo_to_ebitda=cfo_to_ebitda,
                cfo_to_pat=cfo_to_pat,
                cash_and_equivalents=cash_balance
            )

            analyses.append(analysis)

        return analyses
```

### 4.14 Working Capital Analyzer with Sector Benchmarks

**Purpose**: Analyze working capital cycle with sector-relative assessment.

**Implementation**:
```python
class WorkingCapitalAnalyzer:
    """
    Working capital cycle analysis with sector benchmarking
    """
    # Sector benchmarks (to be expanded)
    SECTOR_BENCHMARKS = {
        'FMCG': {'receivable_days': 30, 'inventory_days': 45, 'ccc': 40},
        'Pharma': {'receivable_days': 90, 'inventory_days': 120, 'ccc': 180},
        'IT Services': {'receivable_days': 60, 'inventory_days': 0, 'ccc': 50},
        'Auto': {'receivable_days': 45, 'inventory_days': 60, 'ccc': 80},
        'Textiles': {'receivable_days': 60, 'inventory_days': 90, 'ccc': 120},
        'Capital Goods': {'receivable_days': 120, 'inventory_days': 90, 'ccc': 180},
        'Real Estate': {'receivable_days': 180, 'inventory_days': 365, 'ccc': 400},
        'Steel': {'receivable_days': 60, 'inventory_days': 45, 'ccc': 80},
        'Cement': {'receivable_days': 30, 'inventory_days': 30, 'ccc': 40},
        'Chemicals': {'receivable_days': 60, 'inventory_days': 75, 'ccc': 110}
    }

    def analyze_working_capital(
        self,
        financials: List[FinancialData],
        sector: str
    ) -> List[WorkingCapitalAnalysis]:
        """Analyze working capital for each year"""
        analyses = []

        # Get sector benchmarks
        sector_benchmarks = self._get_sector_benchmarks(sector)

        for i, financial_data in enumerate(financials):
            # Calculate days
            revenue = financial_data.revenue or 0
            cogs = financial_data.cost_of_goods_sold or (revenue * 0.7)  # Estimate if not available

            inventory = financial_data.inventory or 0
            receivables = financial_data.trade_receivables or 0
            payables = financial_data.trade_payables or 0

            inventory_days = (inventory / cogs * 365) if cogs > 0 else 0
            receivable_days = (receivables / revenue * 365) if revenue > 0 else 0
            payable_days = (payables / cogs * 365) if cogs > 0 else 0

            ccc = inventory_days + receivable_days - payable_days

            # Net working capital
            current_assets = financial_data.current_assets or 0
            current_liabilities = financial_data.current_liabilities or 0
            nwc = current_assets - current_liabilities
            nwc_to_revenue = (nwc / revenue * 100) if revenue > 0 else 0

            # YoY trends
            if i > 0:
                prior = analyses[i-1]
                receivable_days_change = receivable_days - prior.receivable_days
                inventory_days_change = inventory_days - prior.inventory_days
                ccc_change = ccc - prior.cash_conversion_cycle

                # Red flag: receivables growing faster than revenue
                revenue_growth = ((revenue / financials[i-1].revenue) - 1) * 100 if financials[i-1].revenue > 0 else 0
                receivable_growth = ((receivables / financials[i-1].trade_receivables) - 1) * 100 if financials[i-1].trade_receivables > 0 else 0
                receivable_vs_revenue_growth = receivable_growth - revenue_growth
            else:
                receivable_days_change = 0
                inventory_days_change = 0
                ccc_change = 0
                receivable_vs_revenue_growth = 0

            # Red flags
            is_receivable_days_worsening = receivable_days_change > 10  # More than 10 days increase
            is_inventory_piling = inventory_days_change > 15

            # vs Sector
            vs_sector_ccc = None
            if sector_benchmarks:
                vs_sector_ccc = ccc - sector_benchmarks['ccc']

            analysis = WorkingCapitalAnalysis(
                fiscal_year=financial_data.fiscal_year,
                inventory_days=inventory_days,
                receivable_days=receivable_days,
                payable_days=payable_days,
                cash_conversion_cycle=ccc,
                inventory=inventory,
                trade_receivables=receivables,
                trade_payables=payables,
                net_working_capital=nwc,
                nwc_to_revenue=nwc_to_revenue,
                receivable_days_change_yoy=receivable_days_change,
                inventory_days_change_yoy=inventory_days_change,
                ccc_change_yoy=ccc_change,
                receivable_growth_vs_revenue_growth=receivable_vs_revenue_growth,
                is_receivable_days_worsening=is_receivable_days_worsening,
                is_inventory_piling=is_inventory_piling,
                sector=sector,
                sector_avg_receivable_days=sector_benchmarks.get('receivable_days') if sector_benchmarks else None,
                sector_avg_inventory_days=sector_benchmarks.get('inventory_days') if sector_benchmarks else None,
                sector_avg_ccc=sector_benchmarks.get('ccc') if sector_benchmarks else None,
                vs_sector_ccc=vs_sector_ccc
            )

            analyses.append(analysis)

        return analyses

    def _get_sector_benchmarks(self, sector: str) -> Optional[Dict]:
        """Get sector benchmarks"""
        for key in self.SECTOR_BENCHMARKS:
            if key.lower() in sector.lower():
                return self.SECTOR_BENCHMARKS[key]
        return None
```

### 4.15 Contingent Liability Categorizer Module

**Purpose**: Categorize and risk-weight contingent liabilities.

**Implementation**:
```python
class ContingentLiabilityCategorizer:
    """
    Categorizes contingent liabilities by type and risk probability
    """
    # Risk probability mapping
    PROBABILITY_WEIGHTS = {
        'tax': 0.3,  # Tax disputes typically settle at 10-30% of claim
        'bank_guarantee': 0.1,  # Low probability if business is performing
        'legal_civil': 0.5,  # Variable, use 50% as conservative
        'legal_criminal': 0.2,  # Lower probability but reputational risk
        'environmental': 0.7,  # High probability for mining/chemicals
        'regulatory': 0.4,  # Depends on nature
        'labor': 0.3
    }

    def analyze_contingent_liabilities(
        self,
        state: AnalysisState,
        net_worth: float,
        objects_analysis: ObjectsOfIssueAnalysis
    ) -> ContingentLiabilityAnalysis:
        """Categorize and analyze contingent liabilities"""

        # Extract contingent liability section
        cl_context = self._retrieve_section("Contingent Liabilities", state)

        # Parse contingent liability notes
        cl_data = self._parse_contingent_liabilities(cl_context)

        # Initialize analysis
        analysis = ContingentLiabilityAnalysis()

        # Categorize each item
        for item in cl_data:
            category = self._categorize_liability(item)
            amount = item['amount']
            count = item.get('count', 1)

            # Add to appropriate category
            if category == 'tax':
                analysis.tax_disputes += amount
                analysis.tax_disputes_count += count
            elif category == 'legal_civil':
                analysis.legal_civil += amount
                analysis.legal_civil_count += count
            elif category == 'legal_criminal':
                analysis.legal_criminal += amount
                analysis.legal_criminal_count += count
            elif category == 'bank_guarantee':
                analysis.bank_guarantees += amount
            elif category == 'environmental':
                analysis.environmental += amount
            elif category == 'regulatory':
                analysis.regulatory_fines += amount
            elif category == 'labor':
                analysis.labor_disputes += amount
            else:
                analysis.other += amount

            # Check timeline risk
            if item.get('hearing_date'):
                hearing_date = item['hearing_date']
                if self._is_within_12_months(hearing_date):
                    analysis.matters_with_hearing_in_12_months += 1
                    analysis.amount_at_risk_in_12_months += amount

            # Create RiskExposure object
            risk_exposure = RiskExposure(
                entity="Company",  # Will be updated if promoter/subsidiary
                count=count,
                amount_cr=amount,
                percent_networth=(amount / net_worth * 100) if net_worth > 0 else 0,
                category=category,
                next_hearing=item.get('hearing_date'),
                severity=self._assess_severity(amount, net_worth),
                citation=item.get('citation', '')
            )
            analysis.items.append(risk_exposure)

        # Calculate totals
        analysis.total_contingent_liabilities = sum([
            analysis.tax_disputes,
            analysis.legal_civil,
            analysis.legal_criminal,
            analysis.bank_guarantees,
            analysis.environmental,
            analysis.regulatory_fines,
            analysis.labor_disputes,
            analysis.other
        ])

        analysis.total_as_percent_networth = (analysis.total_contingent_liabilities / net_worth * 100) if net_worth > 0 else 0

        # Probability-weighted exposure
        analysis.high_probability_exposure = (
            analysis.environmental * self.PROBABILITY_WEIGHTS['environmental'] +
            analysis.legal_civil * self.PROBABILITY_WEIGHTS['legal_civil']
        )

        analysis.medium_probability_exposure = (
            analysis.tax_disputes * self.PROBABILITY_WEIGHTS['tax'] +
            analysis.regulatory_fines * self.PROBABILITY_WEIGHTS['regulatory'] +
            analysis.labor_disputes * self.PROBABILITY_WEIGHTS['labor']
        )

        analysis.low_probability_exposure = (
            analysis.bank_guarantees * self.PROBABILITY_WEIGHTS['bank_guarantee'] +
            analysis.legal_criminal * self.PROBABILITY_WEIGHTS['legal_criminal']
        )

        # Check if IPO proceeds earmarked for settlement
        # This would come from Objects of Issue analysis
        analysis.amount_earmarked_from_ipo = 0  # To be filled if disclosed

        return analysis

    def _categorize_liability(self, item: Dict) -> str:
        """Categorize liability based on description"""
        description = item.get('description', '').lower()

        if any(kw in description for kw in ['income tax', 'gst', 'service tax', 'customs', 'excise']):
            return 'tax'
        elif any(kw in description for kw in ['bank guarantee', 'letter of credit']):
            return 'bank_guarantee'
        elif any(kw in description for kw in ['environmental', 'pollution', 'waste']):
            return 'environmental'
        elif any(kw in description for kw in ['sebi', 'rbi', 'regulatory', 'compliance']):
            return 'regulatory'
        elif any(kw in description for kw in ['labor', 'employee', 'provident fund', 'esi']):
            return 'labor'
        elif any(kw in description for kw in ['criminal', 'fir', 'prosecution']):
            return 'legal_criminal'
        else:
            return 'legal_civil'

    def _assess_severity(self, amount: float, net_worth: float) -> str:
        """Assess severity based on % of net worth"""
        if net_worth == 0:
            return 'high'

        percent = (amount / net_worth) * 100

        if percent > 5:
            return 'high'
        elif percent > 2:
            return 'medium'
        else:
            return 'low'

    def _is_within_12_months(self, date_str: str) -> bool:
        """Check if date is within next 12 months"""
        from datetime import datetime, timedelta
        try:
            hearing_date = datetime.strptime(date_str, "%Y-%m-%d")
            twelve_months_out = datetime.now() + timedelta(days=365)
            return hearing_date <= twelve_months_out
        except:
            return False

    def _retrieve_section(self, section: str, state: AnalysisState) -> List[Chunk]:
        """Retrieve section chunks"""
        pass

    def _parse_contingent_liabilities(self, context: List[Chunk]) -> List[Dict]:
        """Parse contingent liability notes"""
        pass
```

### 4.16 Objects of Issue Tracker Module

**Purpose**: Parse use of proceeds with deployment timeline and readiness flags.

**Implementation**:
```python
class ObjectsOfIssueTracker:
    """
    Tracks use of proceeds with deployment timeline
    """
    def analyze_objects(self, state: AnalysisState, ipo_details: IPODetails) -> ObjectsOfIssueAnalysis:
        """Comprehensive use of proceeds analysis"""

        # Extract Objects of the Issue section
        objects_context = self._retrieve_section("Objects of the Issue", state)

        # Parse use of proceeds table
        use_breakdown = self._parse_use_of_proceeds(objects_context)

        # Calculate percentages
        fresh_issue = ipo_details.fresh_issue_cr
        ofs = ipo_details.ofs_cr
        total_issue = fresh_issue + ofs

        fresh_percent = (fresh_issue / total_issue * 100) if total_issue > 0 else 0
        ofs_percent = (ofs / total_issue * 100) if total_issue > 0 else 0

        # Parse deployment timeline
        deployment_schedule = self._parse_deployment_schedule(objects_context)

        # Check readiness indicators
        readiness = self._check_readiness_indicators(objects_context)

        # Check for monitoring agency
        has_monitoring = self._check_monitoring_agency(objects_context)

        analysis = ObjectsOfIssueAnalysis(
            total_issue_size=total_issue,
            fresh_issue=fresh_issue,
            ofs=ofs,
            fresh_issue_percent=fresh_percent,
            ofs_percent=ofs_percent,
            capex_amount=use_breakdown.get('capex', 0),
            debt_repayment_amount=use_breakdown.get('debt_repayment', 0),
            working_capital_amount=use_breakdown.get('working_capital', 0),
            acquisition_amount=use_breakdown.get('acquisition', 0),
            general_corporate_purposes=use_breakdown.get('gcp', 0),
            issue_expenses=use_breakdown.get('issue_expenses', 0),
            deployment_schedule=deployment_schedule,
            land_acquired_for_capex=readiness.get('land_acquired', False),
            approvals_in_place=readiness.get('approvals', False),
            capex_already_incurred=readiness.get('capex_incurred', 0),
            has_monitoring_agency=has_monitoring,
            monitoring_agency_name=readiness.get('monitoring_agency')
        )

        # Calculate percentages of fresh issue
        if fresh_issue > 0:
            analysis.capex_percent = (analysis.capex_amount / fresh_issue) * 100
            analysis.debt_repayment_percent = (analysis.debt_repayment_amount / fresh_issue) * 100
            analysis.working_capital_percent = (analysis.working_capital_amount / fresh_issue) * 100
            analysis.acquisition_percent = (analysis.acquisition_amount / fresh_issue) * 100
            analysis.gcp_percent = (analysis.general_corporate_purposes / fresh_issue) * 100
            analysis.issue_expenses_percent = (analysis.issue_expenses / fresh_issue) * 100

        # Assessments
        analysis.is_growth_oriented = analysis.capex_amount > analysis.debt_repayment_amount
        analysis.is_exit_oriented = ofs > fresh_issue
        analysis.is_deleveraging = analysis.debt_repayment_amount > (fresh_issue * 0.5)

        # Red flags
        analysis.gcp_exceeds_25_percent = analysis.gcp_percent > 25
        analysis.vague_deployment_timeline = len(deployment_schedule) == 0

        return analysis

    def _parse_use_of_proceeds(self, context: List[Chunk]) -> Dict:
        """Parse use of proceeds breakdown"""
        pass

    def _parse_deployment_schedule(self, context: List[Chunk]) -> List[Dict]:
        """Parse FY-wise deployment schedule"""
        pass

    def _check_readiness_indicators(self, context: List[Chunk]) -> Dict:
        """Check if land acquired, approvals in place, etc."""
        pass

    def _check_monitoring_agency(self, context: List[Chunk]) -> bool:
        """Check if monitoring agency appointed"""
        pass
```

### 4.17 Stub Period Analyzer Module

**Purpose**: Analyze interim/stub period financials vs prior period.

**Implementation**:
```python
class StubPeriodAnalyzer:
    """
    Analyzes stub period (interim) financials
    """
    def analyze_stub_period(
        self,
        state: AnalysisState,
        full_year_financials: List[FinancialData]
    ) -> Optional[StubPeriodAnalysis]:
        """Analyze stub period if available"""

        # Check if stub period exists
        stub_data = self._extract_stub_period(state)

        if not stub_data:
            return None  # No stub period disclosed

        # Get comparable prior period
        prior_data = self._extract_comparable_prior_period(state, stub_data['period'])

        if not prior_data:
            return None

        # Extract metrics
        stub_revenue = stub_data.get('revenue', 0)
        stub_ebitda = stub_data.get('ebitda', 0)
        stub_pat = stub_data.get('pat', 0)

        prior_revenue = prior_data.get('revenue', 0)
        prior_ebitda = prior_data.get('ebitda', 0)
        prior_pat = prior_data.get('pat', 0)

        # Calculate margins
        stub_ebitda_margin = (stub_ebitda / stub_revenue * 100) if stub_revenue > 0 else 0
        stub_pat_margin = (stub_pat / stub_revenue * 100) if stub_revenue > 0 else 0

        prior_ebitda_margin = (prior_ebitda / prior_revenue * 100) if prior_revenue > 0 else 0
        prior_pat_margin = (prior_pat / prior_revenue * 100) if prior_revenue > 0 else 0

        # YoY growth
        revenue_growth = ((stub_revenue / prior_revenue) - 1) * 100 if prior_revenue > 0 else 0
        ebitda_growth = ((stub_ebitda / prior_ebitda) - 1) * 100 if prior_ebitda > 0 else 0
        pat_growth = ((stub_pat / prior_pat) - 1) * 100 if prior_pat > 0 else 0

        margin_expansion = stub_ebitda_margin - prior_ebitda_margin

        # Annualize stub period
        months = stub_data.get('months', 6)
        annualization_factor = 12 / months

        annualized_revenue = stub_revenue * annualization_factor
        annualized_ebitda = stub_ebitda * annualization_factor
        annualized_pat = stub_pat * annualization_factor

        # Compare to last full year
        last_fy = full_year_financials[-1] if full_year_financials else None
        last_fy_revenue = last_fy.revenue if last_fy else 0

        implied_fy_growth = ((annualized_revenue / last_fy_revenue) - 1) * 100 if last_fy_revenue > 0 else 0

        # Calculate historical CAGR
        historical_cagr = self._calculate_historical_cagr(full_year_financials)

        # Warning flags
        stub_below_cagr = revenue_growth < historical_cagr
        margin_compression = margin_expansion < -2  # More than 2% margin drop

        analysis = StubPeriodAnalysis(
            stub_period=stub_data['period'],
            comparable_prior_period=prior_data['period'],
            stub_revenue=stub_revenue,
            stub_ebitda=stub_ebitda,
            stub_pat=stub_pat,
            stub_ebitda_margin=stub_ebitda_margin,
            stub_pat_margin=stub_pat_margin,
            prior_revenue=prior_revenue,
            prior_ebitda=prior_ebitda,
            prior_pat=prior_pat,
            prior_ebitda_margin=prior_ebitda_margin,
            prior_pat_margin=prior_pat_margin,
            revenue_growth_yoy=revenue_growth,
            ebitda_growth_yoy=ebitda_growth,
            pat_growth_yoy=pat_growth,
            margin_expansion=margin_expansion,
            annualized_revenue=annualized_revenue,
            annualized_ebitda=annualized_ebitda,
            annualized_pat=annualized_pat,
            last_full_year_revenue=last_fy_revenue,
            implied_full_year_growth=implied_fy_growth,
            stub_growth_below_historical_cagr=stub_below_cagr,
            margin_compression_in_stub=margin_compression
        )

        return analysis

    def _extract_stub_period(self, state: AnalysisState) -> Optional[Dict]:
        """Extract stub period financials from RHP"""
        pass

    def _extract_comparable_prior_period(self, state: AnalysisState, stub_period: str) -> Optional[Dict]:
        """Extract comparable prior year period"""
        pass

    def _calculate_historical_cagr(self, financials: List[FinancialData]) -> float:
        """Calculate historical revenue CAGR"""
        if len(financials) < 2:
            return 0.0

        first_revenue = financials[0].revenue
        last_revenue = financials[-1].revenue
        years = len(financials) - 1

        if first_revenue > 0 and years > 0:
            cagr = ((last_revenue / first_revenue) ** (1 / years) - 1) * 100
            return cagr
        return 0.0
```

---

## 5. Data Models & Schemas

### 5.1 Core Data Classes

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class ChunkType(Enum):
    NARRATIVE = "narrative"
    TABLE = "table"
    MIXED = "mixed"

class SectionType(Enum):
    RISK_FACTORS = "risk_factors"
    BUSINESS = "business"
    FINANCIAL = "financial"
    MANAGEMENT = "management"
    LEGAL = "legal"
    ISSUE_DETAILS = "issue_details"
    OTHER = "other"

@dataclass
class Page:
    """Represents a single PDF page"""
    page_num: int
    text: str
    images: List[str] = field(default_factory=list)
    tables: List['Table'] = field(default_factory=list)
    layout_info: Dict = field(default_factory=dict)
    is_scanned: bool = False

@dataclass
class Table:
    """Represents an extracted table"""
    table_id: str
    page_num: int
    rows: List[List[str]]
    headers: List[str]
    table_type: Optional[str] = None  # 'financial', 'shareholding', etc.
    confidence: float = 0.0

@dataclass
class Section:
    """Represents a document section"""
    section_id: str
    section_type: SectionType
    title: str
    start_page: int
    end_page: int
    subsections: List['Section'] = field(default_factory=list)
    content: str = ""
    word_count: int = 0

@dataclass
class Chunk:
    """Represents a semantic chunk"""
    chunk_id: str
    text: str
    chunk_type: ChunkType
    section: str
    page_num: int
    start_char: int
    end_char: int
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None

@dataclass
class Entity:
    """Represents an extracted entity"""
    entity_type: str  # 'ORG', 'PERSON', 'LOC', etc.
    text: str
    mentions: int
    contexts: List[str] = field(default_factory=list)
    page_references: List[int] = field(default_factory=list)

@dataclass
class FinancialData:
    """Financial metrics for a fiscal year"""
    fiscal_year: str
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    ebitda_margin: Optional[float] = None
    pat: Optional[float] = None
    pat_margin: Optional[float] = None
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None

    # Calculated ratios
    roe: Optional[float] = None
    roce: Optional[float] = None
    debt_equity_ratio: Optional[float] = None
    current_ratio: Optional[float] = None
    cash_conversion_cycle: Optional[float] = None
    cfo_to_ebitda: Optional[float] = None
    receivable_days: Optional[float] = None

@dataclass
class IPODetails:
    """Key IPO Information"""
    open_date: str
    close_date: str
    listing_date: str
    price_band: str
    lot_size: int
    issue_size_cr: float
    fresh_issue_cr: float
    ofs_cr: float
    market_cap_cr: float
    registrar: str
    lead_managers: List[str]
    qib_quota: str
    retail_quota: str
    nii_quota: str

@dataclass
class Scorecard:
    """Investment Decision Scorecard"""
    financial_health_score: float  # 0-10 (Weight 30%)
    valuation_comfort_score: float # 0-10 (Weight 20%)
    governance_score: float        # 0-10 (Weight 20%)
    business_moat_score: float     # 0-10 (Weight 15%)
    industry_tailwind_score: float # 0-10 (Weight 15%)
    total_score: float             # 0-100
    verdict: str                   # Subscribe / Avoid / Long Term
    veto_flag: bool                # True if Governance < 5
    # Deterministic drivers for transparency
    roe_band: str                  # e.g., " <10%", "10-15%", " >20%"
    pledge_percent: float          # promoter pledge % disclosed in RHP
    rpt_percent: float             # related-party revenue %
    litigation_to_networth: float  # total litigation / post-issue NW

"""Scoring Rubric""":
- Financial Health: 0-10 derived from Base scenario ROE bands (>20% =10, 15-20=8, 10-15=6, <10=3) adjusted for CFO/EBITDA gap penalties.
- Valuation Comfort: start from peer median premium/discount; subtract points for >20% premium without growth justification.
- Governance: start at 10 and subtract per rule breach (pledge, RPT %, auditor remarks, SEBI action).
- Business Moat & Industry Tailwinds: tie directly to qualitative agents but force disclosure of TAM CAGR and dependence metrics.
- Veto flag automatically set if governance <5, litigation_to_networth >10%, or pledge_percent >25%.

@dataclass
class RiskFactor:
    """Represents a risk factor"""
    risk_id: str
    category: str  # 'operational', 'financial', 'regulatory', 'market'
    description: str
    severity: str  # 'high', 'medium', 'low'
    page_reference: int
    impact_assessment: Optional[str] = None

@dataclass
class AgentAnalysis:
    """Output from an agent"""
    agent_name: str
    analysis: str
    key_findings: List[str]
    concerns: List[str]
    confidence: float
    sources: List[int]  # Page references
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Report:
    """Final analysis report"""
    document_id: str
    company_name: str
    generated_at: datetime

    # Executive summary
    tldr: str  # 200-300 words
    investment_thesis: List[str]
    key_risks: List[str]

    # Detailed sections
    business_overview: str
    financial_analysis: str
    governance_assessment: str
    risk_assessment: str
    legal_review: str
    red_flags: List[str]
    projection_scenarios: List[ProjectionScenario]
    valuation_snapshots: List[ValuationSnapshot]
    citations: List[CitationRecord]

    # Metadata
    confidence_score: float
    page_count: int
    processing_time: float

@dataclass
class ProjectionScenario:
    """Base/Bull/Stress projections derived solely from RHP disclosures"""
    name: str
    revenue: Dict[str, float]
    ebitda: Dict[str, float]
    pat: Dict[str, float]
    drivers: Dict[str, str]  # e.g., capacity utilization, order book
    diluted_eps: Dict[str, float]
    roe: Dict[str, float]
    roic: Dict[str, float]
    net_debt_to_ebitda: Dict[str, float]
    cash_conversion_cycle: Dict[str, float]
    assumptions: List[str]
    citations: List[str]

@dataclass
class ValuationSnapshot:
    """Normalized peer comparison at different price points"""
    peer_name: str
    fiscal_year: str
    pe: float
    pb: float
    ev_ebitda: float
    peg: Optional[float]
    source_section: str
    missing_peer: bool

@dataclass
class RiskExposure:
    """Quantified litigation/contingent liability item"""
    entity: str  # Company/Promoter/Subsidiary/Director
    count: int
    amount_cr: float
    percent_networth: float
    category: str  # tax, criminal, civil, regulatory
    next_hearing: Optional[str]
    severity: str
    citation: str

@dataclass
class CitationRecord:
    """Audit trail for sourced claims"""
    claim_id: str
    section: str
    page: int
    paragraph_label: Optional[str]
    text_snippet: str

@dataclass
class PromoterDossier:
    """Comprehensive promoter profile extracted from RHP"""
    name: str
    din: str
    age: Optional[int] = None
    qualification: Optional[str] = None
    experience_years: Optional[int] = None
    designation: Optional[str] = None  # MD, Chairman, etc.

    # Directorships & Conflicts
    other_directorships: List[str] = field(default_factory=list)  # From "Other Directorships"
    group_companies_in_same_line: List[str] = field(default_factory=list)  # Conflict of interest
    common_pursuits: List[str] = field(default_factory=list)  # From "Common Pursuits" section

    # Financial Interest
    shareholding_pre_ipo: float = 0.0
    shareholding_post_ipo: float = 0.0
    selling_via_ofs: float = 0.0  # ₹ Cr
    loans_from_company: float = 0.0  # ₹ Cr (from "Interest of Promoters")
    guarantees_given: float = 0.0  # ₹ Cr
    remuneration_last_3_years: List[float] = field(default_factory=list)
    other_benefits: List[str] = field(default_factory=list)  # Perquisites, ESOPs

    # Litigation specific to promoter
    litigation_as_defendant: List['RiskExposure'] = field(default_factory=list)
    criminal_cases: int = 0
    civil_cases: int = 0
    regulatory_actions: int = 0
    total_litigation_amount: float = 0.0  # ₹ Cr

    # Track record signals
    past_ventures_mentioned: List[str] = field(default_factory=list)
    disqualifications: bool = False

    # Computed metrics
    skin_in_game_post_ipo: float = 0.0  # Post-IPO holding value at cap price

@dataclass
class PreIPOInvestor:
    """Pre-IPO investor details for exit analysis"""
    name: str
    category: str  # "Promoter", "PE/VC", "Angel", "ESOP Trust", "Strategic", "HNI"

    # Entry details
    entry_date: Optional[str] = None
    entry_price: float = 0.0
    shares_acquired: int = 0
    investment_amount: float = 0.0  # ₹ Cr

    # Current holding
    shares_held_pre_ipo: int = 0
    holding_percent_pre_ipo: float = 0.0

    # OFS participation
    shares_selling_via_ofs: int = 0
    ofs_amount: float = 0.0  # ₹ Cr at cap price

    # Return calculations
    implied_return_multiple_at_floor: float = 0.0  # Issue floor / Entry price
    implied_return_multiple_at_cap: float = 0.0
    implied_irr_at_floor: Optional[float] = None  # Annualized return
    implied_irr_at_cap: Optional[float] = None
    holding_period_months: Optional[int] = None

    # Lock-in details
    lock_in_period: str = ""  # "6 months", "1 year", "3 years"
    lock_in_expiry_date: Optional[str] = None
    shares_locked: int = 0
    shares_free_post_listing: int = 0

@dataclass
class FloatAnalysis:
    """Post-listing float and liquidity analysis"""
    # Share capital structure
    total_shares_post_issue: int = 0
    fresh_issue_shares: int = 0

    # Lock-in breakdown
    promoter_locked_shares: int = 0  # 3-year lock-in
    promoter_locked_percent: float = 0.0
    anchor_locked_shares: int = 0  # 90-day lock-in
    pre_ipo_locked_shares: int = 0  # 6-month / 1-year
    esop_unvested_shares: int = 0

    # Float calculations
    day_1_free_float_shares: int = 0
    day_1_free_float_percent: float = 0.0
    day_90_free_float_percent: float = 0.0  # Post anchor unlock
    day_180_free_float_percent: float = 0.0  # Post 6-month unlock
    year_1_free_float_percent: float = 0.0

    # Liquidity indicators
    retail_quota_shares: int = 0
    retail_quota_percent: float = 0.0
    implied_daily_volume: float = 0.0  # Free float / 250

    # Lock-in expiry calendar
    lock_in_calendar: List[Dict] = field(default_factory=list)  # [{date, shares_unlocking, investor, %_of_float}]

@dataclass
class OrderBookAnalysis:
    """Order book / revenue visibility for B2B companies"""
    applicable: bool = False  # Set True for EPC, Defense, IT Services, Capital Goods

    # Order book metrics
    total_order_book: float = 0.0  # ₹ Cr
    order_book_as_of_date: Optional[str] = None
    order_book_to_ltm_revenue: float = 0.0  # "X times" or "Y months of revenue"

    # Order book composition
    top_5_orders_value: float = 0.0  # ₹ Cr
    top_5_orders_concentration: float = 0.0  # % of total order book
    largest_single_order: float = 0.0  # ₹ Cr
    largest_single_order_percent: float = 0.0

    # Execution timeline
    executable_in_12_months: float = 0.0  # ₹ Cr
    executable_in_12_months_percent: float = 0.0
    average_order_tenure_months: Optional[int] = None

    # Trend
    order_book_1yr_ago: Optional[float] = None
    order_book_growth_yoy: Optional[float] = None
    order_inflow_ltm: Optional[float] = None  # Last 12 months new orders
    book_to_bill_ratio: Optional[float] = None  # Order inflow / Revenue

    # Quality indicators
    government_orders_percent: float = 0.0
    private_orders_percent: float = 0.0
    export_orders_percent: float = 0.0
    repeat_customer_orders_percent: float = 0.0

    citations: List[str] = field(default_factory=list)

@dataclass
class DebtStructure:
    """Detailed debt analysis from RHP indebtedness section"""
    # Total debt breakdown
    total_debt: float = 0.0  # ₹ Cr
    secured_debt: float = 0.0
    unsecured_debt: float = 0.0
    short_term_debt: float = 0.0  # < 1 year
    long_term_debt: float = 0.0

    # Cost of debt
    weighted_avg_interest_rate: Optional[float] = None
    highest_interest_rate: Optional[float] = None
    lowest_interest_rate: Optional[float] = None

    # Maturity profile
    maturing_within_1_year: float = 0.0
    maturing_1_to_3_years: float = 0.0
    maturing_3_to_5_years: float = 0.0
    maturing_beyond_5_years: float = 0.0

    # Lender concentration
    top_lender: Optional[str] = None
    top_lender_exposure: float = 0.0  # ₹ Cr
    number_of_lenders: int = 0

    # Covenants & restrictions (from Material Contracts)
    has_financial_covenants: bool = False
    covenant_details: List[str] = field(default_factory=list)
    covenant_breaches_disclosed: bool = False

    # IPO proceeds for debt
    debt_repayment_from_ipo: float = 0.0  # ₹ Cr from Objects of Issue
    debt_repayment_percent_of_fresh_issue: float = 0.0
    post_ipo_debt: float = 0.0  # Net debt after IPO proceeds

    # Key ratios
    debt_to_equity_pre_ipo: float = 0.0
    debt_to_equity_post_ipo: float = 0.0
    interest_coverage_ratio: Optional[float] = None
    debt_to_ebitda: Optional[float] = None

    citations: List[str] = field(default_factory=list)

@dataclass
class CashFlowAnalysis:
    """Enhanced cash flow and liquidity analysis"""
    fiscal_year: str

    # Core cash flows
    cfo: float = 0.0  # Cash Flow from Operations
    cfi: float = 0.0  # Cash Flow from Investing
    cff: float = 0.0  # Cash Flow from Financing
    net_cash_flow: float = 0.0

    # Free Cash Flow metrics
    capex: float = 0.0
    fcf: float = 0.0  # CFO - Capex
    fcf_margin: float = 0.0  # FCF / Revenue
    fcf_yield_at_floor: Optional[float] = None  # FCF / Market Cap at floor
    fcf_yield_at_cap: Optional[float] = None

    # Cash burn analysis (for loss-making companies)
    is_cash_burning: bool = False
    monthly_cash_burn: float = 0.0  # Average monthly burn
    runway_months: float = 0.0  # Cash / Monthly burn

    # Capex analysis
    capex_to_revenue: float = 0.0  # Capex intensity
    capex_to_depreciation: float = 0.0  # >1 = growth capex, ~1 = maintenance
    maintenance_capex_estimate: float = 0.0  # ≈ Depreciation
    growth_capex_estimate: float = 0.0  # Capex - Depreciation

    # Working capital changes
    wc_change: float = 0.0
    wc_change_to_revenue: float = 0.0

    # Quality indicators
    cfo_to_ebitda: float = 0.0  # Should be > 0.7 for quality earnings
    cfo_to_pat: float = 0.0  # Cash conversion

    # Cash position
    cash_and_equivalents: float = 0.0
    cash_to_current_liabilities: float = 0.0

@dataclass
class WorkingCapitalAnalysis:
    """Detailed working capital cycle analysis"""
    fiscal_year: str

    # Days calculations
    inventory_days: float = 0.0
    receivable_days: float = 0.0
    payable_days: float = 0.0
    cash_conversion_cycle: float = 0.0  # Inv + Recv - Payable

    # Absolute values
    inventory: float = 0.0  # ₹ Cr
    trade_receivables: float = 0.0
    trade_payables: float = 0.0
    net_working_capital: float = 0.0
    nwc_to_revenue: float = 0.0  # Working capital intensity

    # Trend analysis
    receivable_days_change_yoy: float = 0.0
    inventory_days_change_yoy: float = 0.0
    ccc_change_yoy: float = 0.0

    # Red flag checks
    receivable_growth_vs_revenue_growth: float = 0.0  # If >> 0, potential channel stuffing
    is_receivable_days_worsening: bool = False
    is_inventory_piling: bool = False

    # Sector benchmark (to be populated based on sector)
    sector: Optional[str] = None
    sector_avg_receivable_days: Optional[float] = None
    sector_avg_inventory_days: Optional[float] = None
    sector_avg_ccc: Optional[float] = None
    vs_sector_ccc: Optional[float] = None  # CCC - Sector avg

@dataclass
class ContingentLiabilityAnalysis:
    """Categorized contingent liability analysis"""
    total_contingent_liabilities: float = 0.0  # ₹ Cr
    total_as_percent_networth: float = 0.0

    # Category-wise breakdown
    tax_disputes: float = 0.0  # GST, Income Tax, Customs
    tax_disputes_count: int = 0
    legal_civil: float = 0.0
    legal_civil_count: int = 0
    legal_criminal: float = 0.0
    legal_criminal_count: int = 0
    bank_guarantees: float = 0.0
    environmental: float = 0.0
    regulatory_fines: float = 0.0  # SEBI, RBI, sector regulators
    labor_disputes: float = 0.0
    other: float = 0.0

    # Probability-weighted exposure
    high_probability_exposure: float = 0.0  # Likely to crystallize
    medium_probability_exposure: float = 0.0
    low_probability_exposure: float = 0.0

    # Timeline risk
    matters_with_hearing_in_12_months: int = 0
    amount_at_risk_in_12_months: float = 0.0

    # Settlement from IPO proceeds
    amount_earmarked_from_ipo: float = 0.0

    # Detailed items
    items: List['RiskExposure'] = field(default_factory=list)

    citations: List[str] = field(default_factory=list)

@dataclass
class ObjectsOfIssueAnalysis:
    """Detailed use of proceeds analysis with deployment timeline"""
    # Issue structure
    total_issue_size: float = 0.0  # ₹ Cr
    fresh_issue: float = 0.0
    ofs: float = 0.0
    fresh_issue_percent: float = 0.0
    ofs_percent: float = 0.0

    # Use of proceeds breakdown
    capex_amount: float = 0.0
    capex_percent: float = 0.0
    debt_repayment_amount: float = 0.0
    debt_repayment_percent: float = 0.0
    working_capital_amount: float = 0.0
    working_capital_percent: float = 0.0
    acquisition_amount: float = 0.0
    acquisition_percent: float = 0.0
    general_corporate_purposes: float = 0.0
    gcp_percent: float = 0.0  # Should be < 25%
    issue_expenses: float = 0.0
    issue_expenses_percent: float = 0.0

    # Deployment timeline
    deployment_schedule: List[Dict] = field(default_factory=list)  # [{use, fy26, fy27, fy28}]
    full_deployment_expected_by: Optional[str] = None

    # Readiness indicators
    land_acquired_for_capex: bool = False
    approvals_in_place: bool = False
    capex_already_incurred: float = 0.0  # From internal accruals
    orders_placed_for_equipment: bool = False

    # Monitoring
    has_monitoring_agency: bool = False
    monitoring_agency_name: Optional[str] = None

    # Assessment
    is_growth_oriented: bool = False  # Capex > Debt repayment
    is_exit_oriented: bool = False  # OFS > Fresh issue
    is_deleveraging: bool = False  # Debt repayment dominant

    # Red flags
    gcp_exceeds_25_percent: bool = False
    vague_deployment_timeline: bool = False

    citations: List[str] = field(default_factory=list)

@dataclass
class StubPeriodAnalysis:
    """Analysis of stub period (interim) financials"""
    stub_period: str  # e.g., "6 months ended Sep 2025"
    comparable_prior_period: str  # e.g., "6 months ended Sep 2024"

    # Stub period metrics
    stub_revenue: float = 0.0
    stub_ebitda: float = 0.0
    stub_pat: float = 0.0
    stub_ebitda_margin: float = 0.0
    stub_pat_margin: float = 0.0

    # Prior period metrics
    prior_revenue: float = 0.0
    prior_ebitda: float = 0.0
    prior_pat: float = 0.0
    prior_ebitda_margin: float = 0.0
    prior_pat_margin: float = 0.0

    # YoY comparison
    revenue_growth_yoy: float = 0.0
    ebitda_growth_yoy: float = 0.0
    pat_growth_yoy: float = 0.0
    margin_expansion: float = 0.0  # Stub EBITDA margin - Prior EBITDA margin

    # Annualized run-rate
    annualized_revenue: float = 0.0
    annualized_ebitda: float = 0.0
    annualized_pat: float = 0.0

    # vs Full year comparison
    last_full_year_revenue: float = 0.0
    implied_full_year_growth: float = 0.0  # Annualized stub vs last FY

    # Seasonality flag
    is_business_seasonal: bool = False
    seasonality_notes: Optional[str] = None

    # Warning flags
    stub_growth_below_historical_cagr: bool = False
    margin_compression_in_stub: bool = False

    citations: List[str] = field(default_factory=list)
```

---

## 6. Technology Stack

### 6.1 Core Dependencies

```toml
# pyproject.toml or requirements.txt

[tool.poetry.dependencies]
python = "^3.10"

# PDF Processing
PyMuPDF = "^1.23.0"  # fitz
pdfplumber = "^0.10.0"
camelot-py = "^0.11.0"
pytesseract = "^0.3.10"
pdf2image = "^1.16.3"

# Document Processing
unstructured = "^0.11.0"
python-docx = "^1.0.0"

# LLM & NLP
langchain = "^0.1.0"
langgraph = "^0.0.20"
sentence-transformers = "^2.2.0"
transformers = "^4.36.0"
torch = "^2.1.0"
huggingface-hub = "^0.20.0"

# Vector Database
qdrant-client = "^1.7.0"
# Alternative: chromadb = "^0.4.0"

# Database
sqlalchemy = "^2.0.0"
alembic = "^1.13.0"

# NER & Entity Extraction
spacy = "^3.7.0"
# en_core_web_sm model

# Financial Analysis
pandas = "^2.1.0"
numpy = "^1.26.0"
openpyxl = "^3.1.0"

# Report Generation
markdown = "^3.5.0"
weasyprint = "^60.0"  # For PDF generation from HTML
jinja2 = "^3.1.0"  # For templating

# Utilities
pydantic = "^2.5.0"
python-dotenv = "^1.0.0"
tqdm = "^4.66.0"
loguru = "^0.7.0"

# Testing
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"

# Optional: For local LLM serving
# ollama-python = "^0.1.0"
```

### 6.2 System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 16 GB
- Storage: 50 GB free
- Python: 3.10+

**Recommended**:
- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA GPU with 8GB+ VRAM (for local summarizer)
- Storage: 100 GB SSD
- Python: 3.11

### 6.3 External Services

**Required**:
- Hugging Face Account (for API access)
- Hugging Face API Token (free tier sufficient)

**Optional**:
- Ollama (for local LLM serving)
- Docker (for Qdrant if not using embedded mode)

---

## 7. Agent System Design

### 7.1 Agent Architecture

Each agent follows this pattern:
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Base class for all analysis agents
    """
    def __init__(self, llm: ReasoningLLM, vector_store: VectorStore, citation_mgr: CitationManager):
        self.llm = llm
        self.vector_store = vector_store
        self.citation_mgr = citation_mgr
        self.name = self.__class__.__name__

    @abstractmethod
    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        """Perform agent-specific analysis"""
        pass

    def retrieve_context(self, query: str, filters: Dict) -> List[Chunk]:
        """Retrieve relevant chunks from vector store"""
        results = self.vector_store.search(
            query_vector=self._embed_query(query),
            filters=filters,
            top_k=10
        )
        return results

    def _embed_query(self, query: str) -> List[float]:
        """Convert query to embedding"""
        embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
        return embedding_model.encode(query).tolist()

    def cite(self, claim_id: str, chunk: Chunk):
        """Attach citation for downstream audit trail"""
        citation = CitationRecord(
            claim_id=claim_id,
            section=chunk.section,
            page=chunk.page_num,
            paragraph_label=chunk.metadata.get("paragraph"),
            text_snippet=chunk.text[:280]
        )
        self.citation_mgr.attach(claim_id, citation)
```

### 7.2 Individual Agent Specifications

#### 7.2.1 Investment Committee Agent (The Decision Maker)
**Role**: Synthesize all analysis and provide the final verdict (Scorecard)

**System Prompt**:
```python
COMMITTEE_PROMPT = """
You are the Chief Investment Officer (CIO) of a top-tier fund.
Your job is to make the final "Subscribe" or "Avoid" decision based on reports from your analyst team.

You must generate a Weighted Scorecard (0-100):
1. Financial Health (30%): Based on Forensic Agent's report.
2. Valuation Comfort (20%): Based on Valuation Agent's report.
3. Governance Quality (20%): Based on Governance Agent's report. (CRITICAL: If < 5/10, Veto the IPO).
4. Business Moat (15%): Based on Business Analyst's report.
5. Industry Tailwinds (15%): Based on Industry Analyst's report.

Quantitative guardrails (reject the decision if missing):
- Report Base/Bull/Stress diluted EPS, ROE, ROIC from the Projection Engine.
- Show price-band sensitivities (floor, cap) and implied upside/downside vs normalized peer median.
- Cite promoter pledge %, RPT %, litigation/NW %, and other veto metrics.

Final Verdict Options:
- SUBSCRIBE (Listing Gains): Good hype, fair pricing, short term play.
- SUBSCRIBE (Long Term): Great business, good governance, hold for 5 years.
- AVOID: Expensive, bad governance, or weak business.
- AVOID (Toxic): Fraud risk, governance failures.

Structure the output as a formal "Investment Committee Memo".

Context (Analyst Reports):
{analyst_reports}

Scenario Inputs:
{scenario_summary}

Your Verdict:
"""
```

**Implementation**:
```python
class InvestmentCommitteeAgent(BaseAgent):
    """
    Synthesizes all inputs into a final verdict
    """
    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        # Collect outputs from all other agents
        reports = self._collect_reports(state)

        prompt = COMMITTEE_PROMPT.format(analyst_reports=reports)
        analysis = self.llm.analyze(prompt)

        return self._create_analysis_output(analysis)
```

#### 7.2.2 Promoter Due Diligence Agent (NEW)
**Role**: Comprehensive promoter background analysis — "Know the Jockey"

**Focus**:
- Promoter biographies and track record
- Common pursuits and conflicts of interest
- Group companies in same line of business
- Promoter litigation (separate from company litigation)
- Skin-in-the-game analysis (post-IPO holding value)
- Loans/guarantees from company to promoters
- OFS participation and selling pressure

**System Prompt**:
```python
PROMOTER_DUE_DILIGENCE_PROMPT = """
You are a forensic investigator analyzing IPO promoters.
Your job is to assess promoter integrity and alignment with minority shareholders.

CRITICAL: In Indian markets, promoter quality is the #1 predictor of long-term value.

Analyze (with RHP citations for every claim):
1. Promoter Background:
   - Age, qualification, experience (years in this business)
   - Past ventures: Success or failure? Any exits? Any bankruptcies?
   - Other directorships: How many companies? Any in competing businesses?

2. Conflicts of Interest:
   - Common Pursuits: Any group companies in the SAME line of business as the IPO company?
   - If yes, this is a MAJOR RED FLAG (fund diversion risk).
   - Cross-holdings: Complex shareholding structures?

3. Skin-in-the-Game Analysis:
   - Pre-IPO holding %
   - Post-IPO holding % (after OFS)
   - Is the promoter SELLING via OFS? How much (₹ Cr)?
   - Calculate: Post-IPO holding value at CAP price (shares × cap price)
   - If selling >20% of stake OR post-IPO holding <51%, FLAG IT.

4. Financial Interest in the Company:
   - Loans taken from the company (from "Interest of Promoters" section)
   - Guarantees given by the company for promoter's benefit
   - Remuneration trend (last 3 years): Is it excessive?

5. Promoter-Specific Litigation:
   - Criminal cases against promoter (count and amount)
   - Civil cases against promoter
   - Regulatory actions (SEBI, RBI, ROC)
   - Calculate: Total litigation amount as % of post-issue net worth
   - If >5%, FLAG as HIGH RISK.

6. Track Record Signals:
   - Has the promoter mentioned past ventures in the RHP? What was the outcome?
   - Any disqualifications under Companies Act?
   - Any promoter exits/deaths in the last 5 years?

Output Format:
## Promoter Dossier
[For each promoter, create a sub-section with all above details]

## Skin-in-the-Game Scorecard
| Promoter | Pre-IPO % | Post-IPO % | Selling via OFS (₹ Cr) | Post-IPO Value @ Cap (₹ Cr) | Risk Flag |
|----------|-----------|------------|-------------------------|------------------------------|-----------|
| ...      | ...       | ...        | ...                     | ...                          | ...       |

## Conflict of Interest Map
[List all group companies in same business]

## Verdict
- ALIGNED: Promoters retaining >75% stake, no conflicts, clean litigation record
- MIXED: Some concerns but manageable
- EXIT MODE: High OFS, conflicts, litigation
- TOXIC: Criminal cases, fraud history, major conflicts

Promoter Analysis Data:
{promoter_dossiers}

Context:
{context}

Your promoter due diligence:
"""
```

**Implementation**:
```python
class PromoterDueDiligenceAgent(BaseAgent):
    """
    Comprehensive promoter background analysis
    """
    def __init__(self, llm: ReasoningLLM, vector_store: VectorStore,
                 citation_mgr: CitationManager, promoter_extractor: PromoterExtractor):
        super().__init__(llm, vector_store, citation_mgr)
        self.promoter_extractor = promoter_extractor

    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        # Extract promoter dossiers
        promoter_dossiers = self.promoter_extractor.extract_promoters(state)

        # Store in state
        state['promoter_dossiers'] = promoter_dossiers

        # Retrieve additional context
        context = self.retrieve_context(
            query="promoters common pursuits interest benefits litigation directorships",
            filters={}
        )

        # Build prompt
        prompt = PROMOTER_DUE_DILIGENCE_PROMPT.format(
            promoter_dossiers=self._format_dossiers(promoter_dossiers),
            context=self._format_context(context)
        )

        # Generate analysis
        analysis = self.llm.analyze(prompt, system_prompt="You are a forensic due diligence expert.")

        # Extract key findings
        key_findings = self._extract_findings(analysis)
        concerns = self._extract_concerns(analysis)

        return AgentAnalysis(
            agent_name="Promoter Due Diligence",
            analysis=analysis,
            key_findings=key_findings,
            concerns=concerns,
            confidence=0.9,
            sources=[c.page_num for c in context]
        )

    def _format_dossiers(self, dossiers: List[PromoterDossier]) -> str:
        """Format promoter dossiers for LLM"""
        formatted = []
        for dossier in dossiers:
            formatted.append(f"""
Promoter: {dossier.name}
- DIN: {dossier.din}
- Age: {dossier.age}, Experience: {dossier.experience_years} years
- Shareholding: Pre-IPO {dossier.shareholding_pre_ipo}%, Post-IPO {dossier.shareholding_post_ipo}%
- Selling via OFS: ₹{dossier.selling_via_ofs} Cr
- Other Directorships: {len(dossier.other_directorships)} companies
- Group Companies (Same Business): {len(dossier.group_companies_in_same_line)}
- Litigation: {dossier.criminal_cases} criminal, {dossier.civil_cases} civil, Total: ₹{dossier.total_litigation_amount} Cr
- Loans from Company: ₹{dossier.loans_from_company} Cr
            """)
        return "\n\n".join(formatted)
```

#### 7.2.3 Business Analyst Agent
**Role**: Deep dive into business operations and SWOT

**System Prompt**:
```python
BUSINESS_PROMPT = """
You are a Business Analyst. Describe the company's operations in detail.
Do not just summarize; analyze the quality of the business.

Analyze:
1. Revenue Mix: Breakdown by Product, Geography, and Customer Segment.
2. Capacity Utilization: Installed capacity vs Actual production (Trends).
3. Order Book Analysis (if applicable for B2B/EPC/Defense sectors):
   - Total order book and order book-to-revenue ratio (months of revenue cover)
   - Top order concentration risk
   - Execution timeline and government vs private orders
   - Order book growth trend
4. Manufacturing/Service Process: Brief overview of how they deliver value.
5. SWOT Analysis:
   - Strengths (Internal)
   - Weaknesses (Internal)
   - Opportunities (External)
   - Threats (External)
6. Client Concentration: Who are the top customers? (Dependency risk).

Context:
{context}

Your analysis:
"""
```

#### 7.2.3 Industry Analyst Agent
**Role**: Analyze the sector and market opportunity

**System Prompt**:
```python
INDUSTRY_PROMPT = """
You are a Sector Specialist. Analyze the industry landscape.
Ignore the company's marketing fluff; look for hard industry data.

Analyze:
1. Market Size (TAM): Total Addressable Market and CAGR.
2. Key Drivers: What is pushing this industry forward? (Govt policy, consumption, etc.)
3. Competitive Landscape:
   - Who are the listed peers?
   - What is the market share of the top players?
4. Barriers to Entry: Is it easy for new players to enter?

Context:
{context}

Your analysis:
"""
```

#### 7.2.4 Management Agent
**Role**: Evaluate leadership capability ("Bet on the Jockey")

**System Prompt**:
```python
MANAGEMENT_PROMPT = """
You are an HR & Leadership expert. Evaluate the "Jockey" (Management).

Analyze:
1. Key Management Personnel (KMP):
   - MD/CEO: Education, Years of Experience, Past Track Record.
   - CFO: Background and credibility.
2. Remuneration:
   - Are salaries market standard or excessive?
3. Attrition: Any recent high-level exits?

Context:
{context}

Your analysis:
"""
```

#### 7.2.5 Capital Structure Agent (ENHANCED)
**Role**: Analyze shareholding, WACA, pre-IPO investor exits, float, and selling pressure

**Focus**:
- WACA vs IPO price analysis
- Pre-IPO investor IRR calculations
- OFS breakdown and selling pressure
- Float analysis (Day-1, Day-90, Year-1)
- Lock-in expiry calendar

**System Prompt**:
```python
CAP_STRUCTURE_PROMPT = """
You are a cynical analyst investigating the 'Capital Structure' of an IPO.
Your goal is to find out if the Promoters/Investors are cashing out at an inflated price.

Analyze (with structured data provided):
1. Weighted Average Cost of Acquisition (WACA):
   - Compare Promoter's WACA vs IPO Price. (Is the multiple >50x? Flag it).
   - Compare Pre-IPO Investors' WACA vs IPO Price.

2. Pre-IPO Investor Exit Analysis:
   Use the PreIPOInvestor data provided:
   - For each PE/VC investor, report: Entry Date, Entry Price, Exit Multiple (at Cap), IRR %
   - Flag any investor earning >50% IRR in <2 years as "Aggressive Pricing"
   - Identify which investors are selling via OFS

3. Offer For Sale (OFS):
   - Who is selling? (Promoters vs PE Funds vs Others)
   - Promoter OFS amount and % of their holding
   - If Promoters are selling >20% of their holding, flag as "Skin in the Game" risk
   - Total OFS as % of issue size

4. Float & Liquidity Analysis:
   Use FloatAnalysis data:
   - Day-1 free float % (should be >10% for adequate liquidity)
   - Day-90 free float % (post anchor unlock)
   - Retail quota % (higher is more retail-friendly)
   - Flag if Day-1 float <5% (very low liquidity risk)

5. Lock-in Expiry Calendar:
   - When does anchor lock-in expire? (90 days)
   - When does pre-IPO investor lock-in expire? (6 months / 1 year)
   - Calculate cumulative selling pressure at each milestone
   - Identify concentrated unlocking events (>10% of equity in single month)

Output Format:
## Pre-IPO Investor Exit Table
| Investor | Category | Entry Date | Entry Price | Exit Price (Cap) | Multiple | IRR % | Lock-in Expiry | Risk Flag |
|----------|----------|------------|-------------|------------------|----------|-------|----------------|-----------|
| ...      | ...      | ...        | ...         | ...              | ...      | ...   | ...            | ...       |

## Float Evolution
| Milestone | Free Float % | Comment |
|-----------|--------------|---------|
| Day 1     | X%           | ...     |
| Day 90    | Y%           | ...     |
| Month 6   | Z%           | ...     |

## Verdict
- FAIR: Reasonable returns for early investors, adequate float
- AGGRESSIVE: >40% IRR for recent investors, low float
- EXIT VEHICLE: Dominated by OFS, promoter selling, poor float

Pre-IPO Investor Data:
{pre_ipo_investors}

Float Analysis:
{float_analysis}

Context:
{context}

Your analysis:
"""
```

**Implementation**:
```python
class CapitalStructureAgent(BaseAgent):
    """
    Analyzes WACA, Pre-IPO rounds, float, and OFS
    """
    def __init__(self, llm: ReasoningLLM, vector_store: VectorStore,
                 citation_mgr: CitationManager,
                 pre_ipo_analyzer: PreIPOInvestorAnalyzer,
                 float_calculator: FloatCalculator):
        super().__init__(llm, vector_store, citation_mgr)
        self.pre_ipo_analyzer = pre_ipo_analyzer
        self.float_calculator = float_calculator

    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        # Analyze pre-IPO investors
        ipo_details = state['ipo_details']
        pre_ipo_investors = self.pre_ipo_analyzer.analyze_investors(state, ipo_details)

        # Calculate float analysis
        promoter_dossiers = state.get('promoter_dossiers', [])
        float_analysis = self.float_calculator.calculate_float_analysis(
            ipo_details,
            pre_ipo_investors,
            promoter_dossiers
        )

        # Store in state
        state['pre_ipo_investors'] = pre_ipo_investors
        state['float_analysis'] = float_analysis

        # Retrieve relevant sections
        cap_struct_context = self.retrieve_context(
            query="capital structure shareholding pattern weighted average cost of acquisition pre-ipo placement",
            filters={"section": "capital_structure"}
        )

        prompt = CAP_STRUCTURE_PROMPT.format(
            pre_ipo_investors=self._format_investors(pre_ipo_investors),
            float_analysis=self._format_float(float_analysis),
            context=self._format_context(cap_struct_context)
        )
        analysis = self.llm.analyze(prompt)

        return self._create_analysis_output(analysis)

    def _format_investors(self, investors: List[PreIPOInvestor]) -> str:
        """Format pre-IPO investor data for LLM"""
        lines = []
        for inv in investors:
            lines.append(f"""
{inv.name} ({inv.category}):
- Entry: {inv.entry_date} @ ₹{inv.entry_price}
- Holding: {inv.shares_held_pre_ipo} shares ({inv.holding_percent_pre_ipo}%)
- OFS: {inv.shares_selling_via_ofs} shares (₹{inv.ofs_amount} Cr)
- Returns: {inv.implied_return_multiple_at_cap}x (IRR: {inv.implied_irr_at_cap}%)
- Lock-in: {inv.lock_in_period} (expires {inv.lock_in_expiry_date})
            """)
        return "\n".join(lines)

    def _format_float(self, float_analysis: FloatAnalysis) -> str:
        """Format float analysis for LLM"""
        return f"""
Total Shares: {float_analysis.total_shares_post_issue:,}
Day-1 Free Float: {float_analysis.day_1_free_float_percent:.1f}% ({float_analysis.day_1_free_float_shares:,} shares)
Day-90 Free Float: {float_analysis.day_90_free_float_percent:.1f}%
Retail Quota: {float_analysis.retail_quota_percent:.1f}%

Lock-in Calendar:
{self._format_lock_in_calendar(float_analysis.lock_in_calendar)}
        """

    def _format_lock_in_calendar(self, calendar: List[Dict]) -> str:
        """Format lock-in calendar"""
        lines = []
        for event in calendar:
            lines.append(f"- {event['date']}: {event['shares_unlocking']:,} shares ({event['percent_of_float']:.1f}%) from {event['investor']}")
        return "\n".join(lines)
```

#### 7.2.6 Forensic Accountant Agent (ENHANCED)
**Role**: Deep financial analysis with cash flow forensics and working capital deep-dive

**Focus Areas**:
- Revenue quality and sustainability
- Margin trends
- Free Cash Flow and cash burn analysis
- Working capital management with sector benchmarks
- Cash flow vs profit quality
- Related party transactions
- Stub period trend analysis

**System Prompt**:
```python
FORENSIC_PROMPT = """
You are a forensic accountant analyzing an IPO prospectus.
Your goal is to assess the quality of earnings and identify any accounting red flags.

Analyze (using structured financial data provided):
1. Revenue Quality: Is revenue sustainable and high-quality?
   - Revenue growth trend (CAGR)
   - Stub period growth vs historical (any slowdown?)

2. Window Dressing Checks (CRITICAL):
   Use WorkingCapitalAnalysis data:
   - Receivable Days trend: Is it worsening?
   - Receivables growth vs Revenue growth: If Receivables growing faster, flag "Channel Stuffing Risk"
   - Inventory Days trend: Is inventory piling up?

3. Cash Flow Quality (CRITICAL):
   Use CashFlowAnalysis data:
   - CFO / EBITDA ratio: Should be >70%. If <50%, flag "Paper Profits"
   - Free Cash Flow (FCF): Is the company FCF positive or burning cash?
   - If burning cash: Monthly burn rate and runway months
   - Capex Intensity: Capex / Revenue %
   - Growth Capex vs Maintenance Capex

4. Working Capital Efficiency:
   Compare to sector benchmarks:
   - Cash Conversion Cycle vs Sector Average
   - If CCC is >50% worse than sector, flag as "Working Capital Inefficiency"

5. Margin Analysis: Analyze gross, EBITDA, and net margins over time
   - Is margin expanding or contracting?
   - Stub period margin vs historical

6. Related Party Transactions:
   - RPT as % of Revenue
   - If >20%, flag as "High RPT Risk"

7. Accounting Policies: Look for aggressive or unusual policies

8. One-time Items: Identify non-recurring items

Be extremely thorough and skeptical. Question everything.
Use the structured data provided and cite RHP sections for qualitative observations.

Financial Data (Historical):
{financial_data}

Cash Flow Analysis:
{cash_flow_analysis}

Working Capital Analysis:
{working_capital_analysis}

Stub Period Analysis:
{stub_period_analysis}

Context:
{context}

Your forensic analysis:
"""
```

**Implementation**:
```python
class ForensicAccountantAgent(BaseAgent):
    """
    Enhanced forensic analysis with cash flow and working capital deep-dive
    """
    def __init__(self, llm: ReasoningLLM, vector_store: VectorStore,
                 citation_mgr: CitationManager,
                 cash_flow_analyzer: EnhancedCashFlowAnalyzer,
                 wc_analyzer: WorkingCapitalAnalyzer,
                 stub_analyzer: StubPeriodAnalyzer):
        super().__init__(llm, vector_store, citation_mgr)
        self.cash_flow_analyzer = cash_flow_analyzer
        self.wc_analyzer = wc_analyzer
        self.stub_analyzer = stub_analyzer

    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        # Get financial data
        financials = state['financial_data']
        ipo_details = state['ipo_details']
        sector = state.get('sector', 'Unknown')

        # Run enhanced analysis
        cash_flow_analyses = self.cash_flow_analyzer.analyze_cash_flows(financials, ipo_details)
        wc_analyses = self.wc_analyzer.analyze_working_capital(financials, sector)
        stub_analysis = self.stub_analyzer.analyze_stub_period(state, financials)

        # Store in state
        state['cash_flow_analyses'] = cash_flow_analyses
        state['working_capital_analyses'] = wc_analyses
        state['stub_period_analysis'] = stub_analysis

        # Retrieve context
        financial_context = self.retrieve_context(
            query="financial statements accounting policies restated results",
            filters={"section": "financial"}
        )

        prompt = FORENSIC_PROMPT.format(
            financial_data=self._format_financials(financials),
            cash_flow_analysis=self._format_cash_flow(cash_flow_analyses),
            working_capital_analysis=self._format_wc(wc_analyses),
            stub_period_analysis=self._format_stub(stub_analysis),
            context=self._format_context(financial_context)
        )

        analysis = self.llm.analyze(prompt, system_prompt="You are a forensic accountant with 20 years of experience.")

        return self._create_analysis_output(analysis)

    def _format_cash_flow(self, analyses: List[CashFlowAnalysis]) -> str:
        """Format cash flow data"""
        lines = ["## Cash Flow Analysis"]
        for cf in analyses:
            lines.append(f"""
{cf.fiscal_year}:
- CFO: ₹{cf.cfo} Cr, CFI: ₹{cf.cfi} Cr, CFF: ₹{cf.cff} Cr
- Capex: ₹{cf.capex} Cr (Intensity: {cf.capex_to_revenue:.1f}%)
- Free Cash Flow: ₹{cf.fcf} Cr (Margin: {cf.fcf_margin:.1f}%)
- CFO/EBITDA: {cf.cfo_to_ebitda:.1f}% {'✓ GOOD' if cf.cfo_to_ebitda > 70 else '⚠️ CONCERN'}
- Cash Burning: {'Yes, Runway: ' + str(cf.runway_months) + ' months' if cf.is_cash_burning else 'No'}
            """)
        return "\n".join(lines)

    def _format_wc(self, analyses: List[WorkingCapitalAnalysis]) -> str:
        """Format working capital data"""
        lines = ["## Working Capital Analysis"]
        for wc in analyses:
            lines.append(f"""
{wc.fiscal_year}:
- CCC: {wc.cash_conversion_cycle:.0f} days (Sector Avg: {wc.sector_avg_ccc or 'N/A'})
- Receivable Days: {wc.receivable_days:.0f} (Change: {wc.receivable_days_change_yoy:+.0f})
- Inventory Days: {wc.inventory_days:.0f} (Change: {wc.inventory_days_change_yoy:+.0f})
- Receivables vs Revenue Growth: {wc.receivable_growth_vs_revenue_growth:+.1f}% {'⚠️ CHANNEL STUFFING RISK' if wc.receivable_growth_vs_revenue_growth > 10 else '✓'}
            """)
        return "\n".join(lines)

    def _format_stub(self, stub: Optional[StubPeriodAnalysis]) -> str:
        """Format stub period data"""
        if not stub:
            return "No stub period disclosed"

        return f"""
## Stub Period: {stub.stub_period} vs {stub.comparable_prior_period}
- Revenue Growth YoY: {stub.revenue_growth_yoy:.1f}% {'⚠️ SLOWDOWN' if stub.stub_growth_below_historical_cagr else '✓'}
- EBITDA Growth YoY: {stub.ebitda_growth_yoy:.1f}%
- Margin Change: {stub.margin_expansion:+.1f}% {'⚠️ COMPRESSION' if stub.margin_compression_in_stub else '✓'}
- Annualized Revenue: ₹{stub.annualized_revenue} Cr
        """
```

#### 7.2.7 Red Flag Agent
        prompt = CAP_STRUCTURE_PROMPT.format(context=self._format_context(cap_struct_context))
        analysis = self.llm.analyze(prompt)

        return self._create_analysis_output(analysis)
```

#### 7.2.3 Forensic Accountant Agent
**Role**: Deep financial analysis and accounting quality

**Focus Areas**:
- Revenue quality and sustainability
- Margin trends
- Working capital management
- Cash flow vs profit
- Related party transactions
- Accounting policy changes
- One-time items and adjustments
- Deterministic rulebook checks (receivable growth deltas, CFO vs EBITDA gap, auditor remarks)
- Citation enforcement: every ratio must reference the exact RHP table (Financial Information, Auditor Reports)
- Structured outputs that feed Projection Engine assumptions (e.g., sustainable margin bands)

**System Prompt**:
```python
FORENSIC_PROMPT = """
You are a forensic accountant analyzing an IPO prospectus.
Your goal is to assess the quality of earnings and identify any accounting red flags.

Analyze:
1. Revenue Quality: Is revenue sustainable and high-quality?
2. Ratio Divergence (Window Dressing Check):
   - Compare Revenue Growth vs Trade Receivables Growth. (If Receivables > Revenue, flag as Channel Stuffing).
   - Compare EBITDA vs Cash Flow from Operations (CFO). (If EBITDA >> CFO, flag as Paper Profits).
3. Margin Analysis: Analyze gross, EBITDA, and net margins over time
4. Cash Flow: Compare operating cash flow to reported profits
5. Working Capital: Assess working capital trends (Inventory Days, Receivable Days).
6. Related Party Transactions: Identify and evaluate RPTs
7. Accounting Policies: Look for aggressive or unusual policies
8. One-time Items: Identify non-recurring items
9. Red Flags: Any signs of earnings manipulation?

Be extremely thorough and skeptical. Question everything.

Financial Data:
{financial_data}

Context:
{context}

Your forensic analysis:
"""
```

#### 7.2.7 Red Flag Agent (ENHANCED)
**Role**: Identify warning signs with quantified metrics

**Red Flag Categories**:
- Governance issues (using Promoter Dossier data)
- Legal disputes (using quantified litigation data)
- Financial red flags (using forensic analysis)
- Float and liquidity risks
- Concentration risks
- Debt structure concerns
- Cash burn risks

**System Prompt**:
```python
RED_FLAG_PROMPT = """
You are a skeptical analyst looking for red flags in an IPO prospectus.
Your job is to identify warning signs that investors should be aware of, with QUANTIFIED metrics.

IMPORTANT: Ignore generic/boilerplate risks (e.g., "general economic slowdown", "competition exists").
Focus ONLY on company-specific risks with hard numbers.

Categorize red flags as:
- CRITICAL: Deal-breakers that make the IPO highly risky
- MAJOR: Significant concerns that require careful consideration
- MINOR: Issues to be aware of but not necessarily deal-breakers

Analyze (using structured data provided):

1. Promoter Red Flags:
   - Criminal cases against promoters (count and amount)
   - Group companies in same line of business (conflict count)
   - OFS by promoters >20% of holding
   - Post-IPO promoter holding <51%

2. Financial Red Flags:
   - FCF negative for >2 years (cash burn)
   - CFO/EBITDA <50% (paper profits)
   - Receivables growing >20% faster than revenue (channel stuffing)
   - Debt/Equity >2x

3. Litigation Red Flags:
   - Total litigation >10% of net worth
   - Criminal litigation >0
   - Contingent liabilities >15% of net worth

4. Concentration Risk:
   - Top 5 Customers >50% of revenue
   - Single customer >25% of revenue
   - Top 5 suppliers >50% of expenses

5. Liquidity Red Flags:
   - Day-1 free float <5%
   - Pre-IPO investor IRR >60% in <18 months

6. Debt Maturity Red Flags:
   - >50% debt maturing in 12 months
   - Weighted avg interest rate >12%
   - Debt covenants close to breach

7. Working Capital Red Flags:
   - CCC >2x sector average
   - Receivable days >120 (except pharma/capital goods)

8. Use of Proceeds Red Flags:
   - OFS >50% of total issue
   - Debt repayment >60% of fresh issue
   - GCP >25%
   - No monitoring agency for >₹100 Cr issue

Output Format:
## Red Flag Summary Table
| Risk | Category | Severity | Metric/Amount | RHP Reference | Threshold Breached |
|------|----------|----------|---------------|---------------|-------------------|
| ...  | ...      | ...      | ...           | ...           | ...               |

## CRITICAL Red Flags (Auto-Reject Triggers)
[List any critical red flags that should result in AVOID recommendation]

## Verdict
- GREEN: No major red flags
- YELLOW: Some concerns, proceed with caution
- RED: Multiple red flags, high risk

Promoter Data:
{promoter_data}

Financial Metrics:
{financial_metrics}

Litigation Data:
{litigation_data}

Float Analysis:
{float_data}

Debt Structure:
{debt_data}

Context:
{context}

Your red flag analysis:
"""
```

**Implementation**:
```python
class RedFlagAgent(BaseAgent):
    """
    Enhanced red flag detection with quantified metrics
    """
    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        # Gather all structured data
        promoter_data = state.get('promoter_dossiers', [])
        cash_flow = state.get('cash_flow_analyses', [])
        wc_analyses = state.get('working_capital_analyses', [])
        float_analysis = state.get('float_analysis')
        debt_structure = state.get('debt_structure')
        litigation_data = state.get('contingent_liability_analysis')

        # Retrieve risk factors section
        risk_context = self.retrieve_context(
            query="risk factors material risks company specific risks",
            filters={"section": "risk_factors"}
        )

        prompt = RED_FLAG_PROMPT.format(
            promoter_data=self._format_promoter_risks(promoter_data),
            financial_metrics=self._format_financial_risks(cash_flow, wc_analyses),
            litigation_data=self._format_litigation_risks(litigation_data),
            float_data=self._format_float_risks(float_analysis),
            debt_data=self._format_debt_risks(debt_structure),
            context=self._format_context(risk_context)
        )

        analysis = self.llm.analyze(prompt)

        return self._create_analysis_output(analysis)

    def _format_promoter_risks(self, promoters: List[PromoterDossier]) -> str:
        """Format promoter risk metrics"""
        risks = []
        for p in promoters:
            risks.append(f"""
{p.name}:
- Criminal Cases: {p.criminal_cases}
- Total Litigation: ₹{p.total_litigation_amount} Cr
- Conflicts: {len(p.group_companies_in_same_line)} group cos in same business
- OFS: ₹{p.selling_via_ofs} Cr
- Post-IPO Holding: {p.shareholding_post_ipo}%
            """)
        return "\n".join(risks)

    def _format_financial_risks(self, cf_list, wc_list) -> str:
        """Format financial risk metrics"""
        if not cf_list or not wc_list:
            return "No data"

        latest_cf = cf_list[-1]
        latest_wc = wc_list[-1]

        return f"""
- FCF: ₹{latest_cf.fcf} Cr ({'NEGATIVE - Burning Cash' if latest_cf.fcf < 0 else 'Positive'})
- CFO/EBITDA: {latest_cf.cfo_to_ebitda:.1f}% ({'< 50% - Paper Profits Risk' if latest_cf.cfo_to_ebitda < 50 else 'OK'})
- Receivables Growth Delta: {latest_wc.receivable_growth_vs_revenue_growth:+.1f}% ({'CHANNEL STUFFING RISK' if latest_wc.receivable_growth_vs_revenue_growth > 20 else 'OK'})
- CCC: {latest_wc.cash_conversion_cycle:.0f} days (Sector: {latest_wc.sector_avg_ccc or 'N/A'})
        """

    def _format_litigation_risks(self, litigation: Optional[ContingentLiabilityAnalysis]) -> str:
        """Format litigation risk metrics"""
        if not litigation:
            return "No litigation data"

        return f"""
- Total Litigation: ₹{litigation.total_contingent_liabilities} Cr ({litigation.total_as_percent_networth:.1f}% of NW)
- Tax Disputes: ₹{litigation.tax_disputes} Cr ({litigation.tax_disputes_count} cases)
- Criminal Cases: ₹{litigation.legal_criminal} Cr ({litigation.legal_criminal_count} cases)
- Environmental: ₹{litigation.environmental} Cr
- High Probability Exposure: ₹{litigation.high_probability_exposure} Cr
        """

    def _format_float_risks(self, float_analysis: Optional[FloatAnalysis]) -> str:
        """Format float risk metrics"""
        if not float_analysis:
            return "No float data"

        return f"""
- Day-1 Free Float: {float_analysis.day_1_free_float_percent:.1f}% ({'LOW LIQUIDITY' if float_analysis.day_1_free_float_percent < 5 else 'OK'})
- Retail Quota: {float_analysis.retail_quota_percent:.1f}%
        """

    def _format_debt_risks(self, debt: Optional[DebtStructure]) -> str:
        """Format debt risk metrics"""
        if not debt:
            return "No debt data"

        return f"""
- Total Debt: ₹{debt.total_debt} Cr
- Debt/Equity: {debt.debt_to_equity_pre_ipo:.2f}x
- Interest Rate: {debt.weighted_avg_interest_rate or 'N/A'}%
- Maturing <1yr: ₹{debt.maturing_within_1_year} Cr
- Covenants: {'Yes' if debt.has_financial_covenants else 'No'}
        """
```

#### 7.2.8 Governance Agent (ENHANCED)
**Role**: Assess corporate governance with promoter due diligence integration

**System Prompt**:
```python
GOVERNANCE_PROMPT = """
You are a corporate governance expert.
Your task is to evaluate the quality of the company's governance and assign a Governance Score (0-10).

CRITICAL: Governance Score <5 triggers automatic VETO of IPO recommendation.

Analyze (using structured data provided):

1. Promoter Quality & Integrity (Use PromoterDossier data):
   - Promoter background: Education, experience, track record
   - Criminal/regulatory cases: Count and severity
   - Conflicts of interest: Group companies in same business
   - Skin-in-the-game: Post-IPO holding % and OFS participation
   - START AT 10 and DEDUCT:
     * -3 points: Any criminal case against promoter
     * -2 points: >3 group companies in same line of business
     * -2 points: Post-IPO holding <51%
     * -1 point: OFS >20% of promoter stake

2. Board Independence & Quality:
   - % of independent directors (should be ≥50% for listed companies)
   - Audit committee composition
   - Nomination & remuneration committee
   - DEDUCT:
     * -1 point: Independent directors <50%
     * -1 point: Audit committee has executive directors

3. Related Party Transactions (RPTs):
   - RPT revenue as % of total revenue
   - RPT expenses as % of total expenses
   - Identify fund leakage to promoter entities
   - DEDUCT:
     * -2 points: RPT revenue >20%
     * -1 point: RPT revenue 10-20%

4. Executive Compensation:
   - MD/CEO compensation as % of PAT
   - Is it in line with peer companies?
   - DEDUCT:
     * -1 point: Compensation >10% of PAT

5. Auditor Quality & Qualifications:
   - Who are the statutory auditors?
   - Any qualifications in audit report?
   - Any CARO/NCF remarks?
   - DEDUCT:
     * -2 points: Modified audit opinion
     * -1 point: Material CARO observations

6. Promoter Pledge & Encumbrance:
   - Any shares pledged?
   - % of promoter holding pledged
   - DEDUCT:
     * -3 points: Pledge >25%
     * -2 points: Pledge 10-25%
     * -1 point: Pledge >0% but <10%

## Governance Scorecard (0-10)
[Calculate score by starting at 10 and applying deductions]

## Critical Governance Issues
[List any issues that triggered 2+ point deductions]

## Verdict
- EXCELLENT (9-10): Best-in-class governance
- GOOD (7-8): Above-average governance
- ACCEPTABLE (5-6): Manageable concerns
- POOR (<5): VETO - Do not invest

Promoter Dossiers:
{promoter_dossiers}

Board & Management Data:
{board_data}

RPT Data:
{rpt_data}

Context:
{context}

Your governance analysis:
"""
```

**Implementation**:
```python
class GovernanceAgent(BaseAgent):
    """
    Enhanced governance analysis with scoring
    """
    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        # Get promoter dossiers
        promoter_dossiers = state.get('promoter_dossiers', [])

        # Retrieve governance context
        governance_context = self.retrieve_context(
            query="board directors independent audit committee remuneration related party transactions",
            filters={}
        )

        # Extract RPT data from financial statements
        rpt_data = self._extract_rpt_data(state)

        prompt = GOVERNANCE_PROMPT.format(
            promoter_dossiers=self._format_promoter_dossiers(promoter_dossiers),
            board_data="To be extracted from RHP",
            rpt_data=rpt_data,
            context=self._format_context(governance_context)
        )

        analysis = self.llm.analyze(prompt)

        # Extract governance score
        governance_score = self._extract_score(analysis)
        state['governance_score'] = governance_score

        return self._create_analysis_output(analysis)

    def _extract_rpt_data(self, state: AnalysisState) -> str:
        """Extract RPT data from financial statements"""
        # To be implemented
        return "RPT data extraction pending"
```

#### 7.2.9 Legal Agent (ENHANCED)
**Role**: Review legal aspects with categorized contingent liabilities

**System Prompt**:
```python
LEGAL_PROMPT = """
You are a legal expert analyzing an IPO prospectus.

Analyze (using structured ContingentLiabilityAnalysis data):

1. Litigation Summary (by Entity):
   Create a table for each entity (Company, Promoters, Directors, Subsidiaries):
   - Total count of cases
   - Total monetary amount
   - % of post-issue net worth
   - Category breakdown (tax, civil, criminal, regulatory)

2. Contingent Liabilities (by Category):
   Use the categorized data provided:
   - Tax Disputes: Count and amount (typically settle at 10-30%)
   - Bank Guarantees: Low risk if business performing
   - Environmental: High risk for mining/chemicals
   - Regulatory Fines: SEBI, RBI, sector regulators
   - Labor Disputes: Usually manageable

3. Probability-Weighted Exposure:
   - High probability: ₹X Cr (likely to materialize)
   - Medium probability: ₹Y Cr
   - Low probability: ₹Z Cr

4. Timeline Risk:
   - Matters with hearings in next 12 months
   - Amount at risk in near term

5. IPO Proceeds for Settlement:
   - Is any amount from Objects of Issue earmarked for litigation settlement?

6. Material Contracts & IP:
   - Any restrictive covenants?
   - IP ownership clear?

## Litigation Risk Matrix
| Entity | Count | Amount (₹ Cr) | % NW | Next Hearing | Severity |
|--------|-------|---------------|------|--------------|----------|
| ...    | ...   | ...           | ...  | ...          | ...      |

## Contingent Liability Risk Score (0-10)
- 10: Minimal litigation, <2% of NW
- 7-9: Moderate litigation, 2-5% of NW
- 4-6: Material litigation, 5-10% of NW
- 0-3: HIGH RISK, >10% of NW or criminal cases

## Verdict
- CLEAN: Minimal legal risks
- MANAGEABLE: Some risks but within normal range
- CONCERNING: Material litigation
- HIGH RISK: Recommend AVOID

Contingent Liability Analysis:
{contingent_liability_analysis}

Promoter Litigation:
{promoter_litigation}

Context:
{context}

Your legal analysis:
"""
```

**Implementation**:
```python
class LegalAgent(BaseAgent):
    """
    Enhanced legal analysis with categorized contingent liabilities
    """
    def __init__(self, llm: ReasoningLLM, vector_store: VectorStore,
                 citation_mgr: CitationManager,
                 cl_categorizer: ContingentLiabilityCategorizer):
        super().__init__(llm, vector_store, citation_mgr)
        self.cl_categorizer = cl_categorizer

    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        # Get net worth for calculations
        financials = state['financial_data']
        latest_financial = financials[-1] if financials else None
        net_worth = latest_financial.total_equity if latest_financial else 0

        # Get objects analysis for settlement check
        objects_analysis = state.get('objects_of_issue_analysis')

        # Analyze contingent liabilities
        cl_analysis = self.cl_categorizer.analyze_contingent_liabilities(
            state,
            net_worth,
            objects_analysis
        )

        # Store in state
        state['contingent_liability_analysis'] = cl_analysis

        # Get promoter litigation
        promoter_dossiers = state.get('promoter_dossiers', [])

        # Retrieve legal context
        legal_context = self.retrieve_context(
            query="litigation outstanding legal proceedings contingent liabilities material contracts",
            filters={}
        )

        prompt = LEGAL_PROMPT.format(
            contingent_liability_analysis=self._format_cl_analysis(cl_analysis),
            promoter_litigation=self._format_promoter_litigation(promoter_dossiers),
            context=self._format_context(legal_context)
        )

        analysis = self.llm.analyze(prompt)

        return self._create_analysis_output(analysis)

    def _format_cl_analysis(self, cl: ContingentLiabilityAnalysis) -> str:
        """Format contingent liability data"""
        return f"""
Total Contingent Liabilities: ₹{cl.total_contingent_liabilities} Cr ({cl.total_as_percent_networth:.1f}% of NW)

Breakdown by Category:
- Tax Disputes: ₹{cl.tax_disputes} Cr ({cl.tax_disputes_count} cases)
- Civil Litigation: ₹{cl.legal_civil} Cr ({cl.legal_civil_count} cases)
- Criminal Litigation: ₹{cl.legal_criminal} Cr ({cl.legal_criminal_count} cases)
- Bank Guarantees: ₹{cl.bank_guarantees} Cr
- Environmental: ₹{cl.environmental} Cr
- Regulatory: ₹{cl.regulatory_fines} Cr
- Labor: ₹{cl.labor_disputes} Cr

Probability-Weighted:
- High: ₹{cl.high_probability_exposure} Cr
- Medium: ₹{cl.medium_probability_exposure} Cr
- Low: ₹{cl.low_probability_exposure} Cr

Timeline Risk:
- {cl.matters_with_hearing_in_12_months} matters with hearings in next 12 months
- Amount at risk: ₹{cl.amount_at_risk_in_12_months} Cr
        """

    def _format_promoter_litigation(self, promoters: List[PromoterDossier]) -> str:
        """Format promoter-specific litigation"""
        lines = []
        for p in promoters:
            if p.litigation_as_defendant:
                lines.append(f"""
{p.name}:
- Criminal: {p.criminal_cases} cases
- Civil: {p.civil_cases} cases
- Total: ₹{p.total_litigation_amount} Cr
                """)
        return "\n".join(lines) if lines else "No promoter litigation"
```

#### 7.2.10 Utilization Agent (ENHANCED)
**Role**: Analyze use of proceeds with deployment timeline and readiness

**System Prompt**:
```python
UTILIZATION_PROMPT = """
You are an investment analyst checking the 'Objects of the Issue'.

Analyze (using ObjectsOfIssueAnalysis data):

1. Issue Structure:
   - Fresh Issue: ₹X Cr (Y%)
   - Offer for Sale (OFS): ₹A Cr (B%)
   - If OFS >50% of total issue, this is EXIT-ORIENTED (promoters cashing out)

2. Use of Proceeds Breakdown:
   Create a table with % allocation:
   - Capex / Expansion: ₹X Cr (Y%)
   - Debt Repayment: ₹A Cr (B%)
   - Working Capital: ₹C Cr (D%)
   - General Corporate Purposes (GCP): ₹E Cr (F%)
   - Issue Expenses: ₹G Cr (H%)

3. Red Flags:
   - GCP >25%: Vague use, lack of clarity
   - Debt Repayment >60%: Balance sheet repair, not growth
   - OFS > Fresh Issue: Promoter exit

4. Deployment Timeline (if disclosed):
   - FY-wise deployment schedule
   - Expected completion date
   - If vague/missing, flag as "Execution Risk"

5. Readiness Indicators:
   - Land acquired for capex? (Yes/No)
   - Regulatory approvals in place? (Yes/No)
   - Capex already incurred from internal accruals: ₹X Cr
   - Equipment orders placed? (Yes/No)
   - If all "No", flag as "Greenfield Risk - Execution Uncertain"

6. Debt Repayment Context:
   Use DebtStructure data:
   - Pre-IPO Debt: ₹X Cr (Debt/Equity: A.B)
   - Debt being repaid: ₹Y Cr
   - Post-IPO Debt: ₹Z Cr (Debt/Equity: C.D)
   - Interest rate: E%
   - If repaying high-cost debt (>12%), it's value-accretive

7. Monitoring Agency:
   - Appointed? (Mandatory for >₹100 Cr fresh issue)
   - If not appointed and issue >₹100 Cr, flag as governance lapse

## Use of Proceeds Assessment
| Use Case | Amount (₹ Cr) | % of Fresh | Readiness | Timeline | Comment |
|----------|---------------|------------|-----------|----------|---------|
| ...      | ...           | ...        | ...       | ...      | ...     |

## Debt Reduction Impact
- Pre-IPO Debt/Equity: X.X
- Post-IPO Debt/Equity: Y.Y
- Interest Savings: ₹Z Cr/year

## Verdict
- GROWTH-ORIENTED: Capex >60%, OFS <30%, good readiness
- BALANCED: Mix of debt + capex
- DELEVERAGING: Debt repayment dominant (acceptable if high-cost debt)
- EXIT-ORIENTED: OFS >50%, promoters cashing out
- VAGUE: High GCP, poor disclosure, execution risk

Objects of Issue Data:
{objects_analysis}

Debt Structure:
{debt_structure}

Context:
{context}

Your utilization analysis:
"""
```

**Implementation**:
```python
class UtilizationAgent(BaseAgent):
    """
    Enhanced use of proceeds analysis with deployment tracking
    """
    def __init__(self, llm: ReasoningLLM, vector_store: VectorStore,
                 citation_mgr: CitationManager,
                 objects_tracker: ObjectsOfIssueTracker,
                 debt_analyzer: DebtStructureAnalyzer):
        super().__init__(llm, vector_store, citation_mgr)
        self.objects_tracker = objects_tracker
        self.debt_analyzer = debt_analyzer

    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        # Get IPO details
        ipo_details = state['ipo_details']

        # Analyze objects of issue
        objects_analysis = self.objects_tracker.analyze_objects(state, ipo_details)

        # Analyze debt structure
        debt_structure = self.debt_analyzer.analyze_debt(state, objects_analysis)

        # Store in state
        state['objects_of_issue_analysis'] = objects_analysis
        state['debt_structure'] = debt_structure

        # Retrieve context
        objects_context = self.retrieve_context(
            query="objects of the issue use of proceeds deployment schedule monitoring agency",
            filters={}
        )

        prompt = UTILIZATION_PROMPT.format(
            objects_analysis=self._format_objects(objects_analysis),
            debt_structure=self._format_debt(debt_structure),
            context=self._format_context(objects_context)
        )

        analysis = self.llm.analyze(prompt)

        return self._create_analysis_output(analysis)

    def _format_objects(self, obj: ObjectsOfIssueAnalysis) -> str:
        """Format objects of issue data"""
        return f"""
Issue Structure:
- Total Issue: ₹{obj.total_issue_size} Cr
- Fresh Issue: ₹{obj.fresh_issue} Cr ({obj.fresh_issue_percent:.1f}%)
- OFS: ₹{obj.ofs} Cr ({obj.ofs_percent:.1f}%)

Use of Proceeds:
- Capex: ₹{obj.capex_amount} Cr ({obj.capex_percent:.1f}%)
- Debt Repayment: ₹{obj.debt_repayment_amount} Cr ({obj.debt_repayment_percent:.1f}%)
- Working Capital: ₹{obj.working_capital_amount} Cr ({obj.working_capital_percent:.1f}%)
- Acquisitions: ₹{obj.acquisition_amount} Cr ({obj.acquisition_percent:.1f}%)
- GCP: ₹{obj.general_corporate_purposes} Cr ({obj.gcp_percent:.1f}%)
- Issue Expenses: ₹{obj.issue_expenses} Cr ({obj.issue_expenses_percent:.1f}%)

Readiness:
- Land Acquired: {'Yes' if obj.land_acquired_for_capex else 'No'}
- Approvals in Place: {'Yes' if obj.approvals_in_place else 'No'}
- Capex Already Incurred: ₹{obj.capex_already_incurred} Cr

Monitoring:
- Monitoring Agency: {'Yes - ' + obj.monitoring_agency_name if obj.has_monitoring_agency else 'No'}

Red Flags:
- GCP >25%: {'YES ⚠️' if obj.gcp_exceeds_25_percent else 'No'}
- Vague Timeline: {'YES ⚠️' if obj.vague_deployment_timeline else 'No'}

Assessment:
- Growth-Oriented: {'Yes' if obj.is_growth_oriented else 'No'}
- Exit-Oriented: {'Yes ⚠️' if obj.is_exit_oriented else 'No'}
- Deleveraging: {'Yes' if obj.is_deleveraging else 'No'}
        """

    def _format_debt(self, debt: DebtStructure) -> str:
        """Format debt structure data"""
        return f"""
Debt Analysis:
- Pre-IPO Debt: ₹{debt.total_debt} Cr
- Debt Repayment from IPO: ₹{debt.debt_repayment_from_ipo} Cr
- Post-IPO Debt: ₹{debt.post_ipo_debt} Cr

Debt/Equity:
- Pre-IPO: {debt.debt_to_equity_pre_ipo:.2f}x
- Post-IPO: {debt.debt_to_equity_post_ipo:.2f}x

Cost of Debt: {debt.weighted_avg_interest_rate or 'N/A'}%
Maturity Profile:
- <1 year: ₹{debt.maturing_within_1_year} Cr
- 1-3 years: ₹{debt.maturing_1_to_3_years} Cr
- >3 years: ₹{debt.maturing_beyond_5_years} Cr
        """
```

#### 7.2.11 Valuation Agent (Minor Enhancement)
**Role**: Assess corporate governance quality

**Focus**:
- Board composition and independence
- Promoter background and track record
- Related party transactions (RPTs)
- Compensation structures
- Disclosure practices
- Audit committee quality
- Group Company conflicts

**System Prompt**:
```python
GOVERNANCE_PROMPT = """
You are a corporate governance expert.
Your task is to evaluate the quality of the company's governance.

Analyze:
1. Board Independence: Are there enough independent directors?
2. Promoter Background: Any criminal history or questionable track record?
3. Related Party Transactions (RPTs):
   - Calculate RPTs as a % of Revenue.
   - Identify any significant leakage of funds to group companies.
4. Executive Compensation: Is it excessive compared to peers/profits?
5. Auditor Quality: Who are the auditors? Any qualifications?
6. Family Tree & Group Companies:
   - Are there group companies in the same line of business? (Conflict of Interest).
   - Are there complex cross-holdings?
7. Disclose promoter pledge %, encumbrances, and any SEBI enforcement history (Rulebook IDs if triggered).

Be critical of any conflict of interest.
Return findings as bullet points with inline citations [Section-Page].

Context:
{context}

Your governance analysis:
"""
```

#### 7.2.6 Valuation Agent
**Role**: Assess valuation and pricing

**Focus**:
- P/E, P/B, EV/EBITDA vs Peers
- PEG Ratio
- Pre-IPO vs Post-IPO Market Cap
- Basis for Issue Price
- Missing Peers

**System Prompt**:
```python
VALUATION_PROMPT = """
You are a valuation expert.
Your task is to determine if the IPO is priced fairly.

Analyze:
1. Peer Comparison: Compare P/E, P/B, and EV/EBITDA with listed peers.
   - Check for "Missing Peers": Are there competitors mentioned in the Industry Overview that are excluded from the Valuation table?
2. Growth Adjusted Valuation: Calculate PEG ratio. Is the premium justified by growth?
3. Market Cap: Calculate implied market cap at upper price band.
4. Pre-IPO vs Issue Price Gap:
   - Compare the Pre-IPO placement price (if any) with the Issue Price.
   - If the jump is >20% in <6 months, flag as aggressive.
5. Anchor Investor Interest: (If available) Who is buying?
 6. Scenario Sensitivities: Show valuation at Base/Bull/Stress EPS and for both floor and cap prices.

Verdict: Overvalued / Fairly Valued / Undervalued

Use the Valuation Normalization module output and cite every multiple with Section-Page references.

Context:
{context}

Financial Data:
{financial_data}

Your valuation analysis:
"""
```

#### 7.2.7 Utilization Agent
**Role**: Analyze use of proceeds

**Focus**:
- Fresh Issue vs Offer for Sale (OFS)
- Debt Repayment vs Growth Capital
- General Corporate Purposes (GCP)
- Monitoring Agency

**System Prompt**:
```python
UTILIZATION_PROMPT = """
You are an investment analyst checking the 'Objects of the Issue'.

Analyze:
1. Issue Structure: Calculate Fresh Issue % vs OFS %.
   - If OFS is high (>50%), why are promoters selling?
2. Use of Proceeds:
   - How much is for Debt Repayment? (Balance sheet repair vs Growth)
   - How much is for Capex/Expansion?
   - How much is for General Corporate Purposes? (Should be <25%)
3. Monitoring Agency:
   - Is a monitoring agency appointed? (Mandatory for >100Cr).
   - If not, flag as a governance risk for SME/Small issues.

Verdict: Growth-oriented / Exit-oriented / Debt-reduction

Context:
{context}

Your utilization analysis:
"""
```

#### 7.2.8 Legal Agent
**Role**: Review legal and regulatory aspects

**Focus**:
- Pending litigations
- Regulatory compliance
- Material contracts
- Intellectual property
- Contingent liabilities
- Litigation Quantification (Total monetary risk)

**System Prompt**:
```python
LEGAL_PROMPT = """
You are a legal expert analyzing an IPO prospectus.

Analyze:
1. Litigation Summary:
   - Total number of criminal/civil/tax proceedings.
   - Total monetary amount involved (Quantify the risk).
   - Calculate Total Litigation Amount as % of Net Worth. (If >10%, flag as Critical).
2. Contingent Liabilities:
   - Extract the total amount.
   - Calculate as % of Net Worth. (Is it >10%?)
3. Material Contracts: Any restrictive covenants?
4. Regulatory Actions: Any past actions by SEBI/RBI?
 5. Timeline Risk: Highlight matters with hearings/settlement deadlines within 12 months of listing.

Context:
{context}

Output two tables: (a) Litigation Exposure, (b) Contingent Liabilities, each with columns [Entity, Count, Amount (₹ Cr), % of Net Worth, Next Milestone, Section-Page].

Your legal analysis:
"""
```

#### 7.2.9 Self-Critic Agent
**Role**: Validate and challenge other agents' findings

**Purpose**:
- Check for hallucinations
- Verify numerical accuracy
- Ensure page references are valid
- Flag unsupported claims
- Assess overall confidence
- Reject any claim without a CitationRecord entry; request re-run for missing sources

**System Prompt**:
```python
SELF_CRITIC_PROMPT = """
You are a critical reviewer of AI-generated analysis.
Your job is to identify errors, unsupported claims, and hallucinations.

Review the following analyses and:
1. Verify numerical claims against source data
2. Check that page references are valid
3. Flag any unsupported assertions
4. Identify contradictions between agents
5. Assess confidence levels

Be harsh. Any claim without evidence should be flagged.

Analyses to Review:
{all_analyses}

Source Document Metadata:
{metadata}

Your critique:
"""
```

#### 7.2.10 Q&A Agent (On-Demand)
**Role**: Answer specific user questions

**Implementation**:
```python
class QAAgent(BaseAgent):
    """
    Answers specific questions using RAG
    """
    def answer(self, question: str, state: AnalysisState) -> str:
        # Retrieve relevant context
        context_chunks = self.retrieve_context(
            query=question,
            filters={},
            top_k=5
        )

        # Build prompt
        context = "\n\n".join([
            f"[Page {chunk.page_num}] {chunk.text}"
            for chunk in context_chunks
        ])

        prompt = f"""
        Based on the following excerpts from the RHP, answer the question.
        Cite specific page numbers for your answer.
        If the answer cannot be found, say "Insufficient evidence in the document."

        Question: {question}

        Context:
        {context}

        Answer:
        """

        answer = self.llm.generate(prompt)
        return answer
```

---

## 8. Workflow & Processing Pipeline

### 8.1 Detailed Phase Breakdown

#### Phase 1: Ingestion & Preprocessing (15-20 minutes)

**Steps**:
1. **PDF Load** (1-2 min)
   - Validate PDF integrity
   - Extract metadata (page count, file size)

2. **Text Extraction** (5-8 min)
   - Parse all pages with PyMuPDF
   - Identify scanned pages
   - Apply OCR if needed

3. **Table Extraction** (5-7 min)
   - Run unstructured.io detection
   - Extract tables with pdfplumber
   - Classify table types
   - Parse financial statements

4. **Section Mapping** (2-3 min)
   - Analyze document structure
   - Build section hierarchy
   - Extract section boundaries

5. **Entity Extraction** (3-5 min)
   - Run NER on full document
   - Extract companies, people, locations
   - Resolve coreferences

6. **Financial Parsing** (2-3 min)
   - Extract financial metrics from tables
   - Calculate ratios
   - Identify trends

**Output**:
- Structured document data
- Extracted entities
- Financial timeseries
- Section map

#### Phase 2: Chunking & Embedding (10-15 minutes)

**Steps**:
1. **Semantic Chunking** (5-7 min)
   - Split text into 500-1500 token chunks
   - Respect section boundaries
   - Add metadata (section, page, type)

2. **Embedding Generation** (5-8 min)
   - Generate embeddings for all chunks
   - Batch processing (32 chunks at a time)
   - Store in vector database

**Output**:
- ~200-500 chunks with embeddings
- Populated vector database

#### Phase 2.5: Financial Modeling & Valuation Prep (8-10 minutes)

**Steps**:
1. **Historical Normalization** (2-3 min)
    - Align consolidated vs standalone statements
    - Convert all figures to ₹ crore, reconcile restatements
2. **Scenario Builder** (3-4 min)
    - Projection Engine derives Base/Bull/Stress using disclosed drivers (capacity, order book, management guidance)
    - Compute post-issue diluted EPS, ROE, ROIC, net-debt/EBITDA, CCC
3. **Valuation Normalization** (2-3 min)
    - Pull every peer disclosed in the RHP, adjust fiscal years, calculate multiples at floor/cap prices
4. **Rulebook Evaluation** (1 min)
    - Governance & Forensic Rulebook processes structured data to pre-flag breaches for downstream agents

**Output**:
- ProjectionScenario objects with citations
- ValuationSnapshot table with peer-normalized multiples
- Rulebook alerts ready for Governance/Red Flag agents

#### Phase 3: Agent Analysis (30-45 minutes)

**Sequential Execution**:

1. **Business Analyst Agent** (5-7 min)
   - Analyzes revenue mix, capacity, and SWOT
   - Maps out business operations

2. **Industry Analyst Agent** (3-5 min)
   - Extracts TAM, CAGR, and competitive landscape
   - Identifies industry tailwinds

3. **Management Agent** (3-5 min)
   - Profiles MD, CEO, CFO
   - Evaluates experience and remuneration

4. **Capital Structure Agent** (3-5 min)
   - Analyzes WACA and Pre-IPO placements
   - Checks for "Skin in the Game" issues
   - Evaluates OFS selling pressure

5. **Forensic Accountant Agent** (8-10 min)
   - Deep dive into financial statements
   - Analyzes accounting quality
   - Identifies financial red flags

6. **Red Flag Agent** (5-7 min)
   - Scans for warning signs
   - Categorizes risks
   - Assigns severity levels

7. **Governance Agent** (5-7 min)
   - Assesses board composition
   - Evaluates promoter background
   - Reviews related party transactions

8. **Legal Agent** (5-7 min)
   - Reviews legal proceedings
   - Assesses regulatory compliance
   - Identifies contingent liabilities

9. **Valuation Agent** (5-7 min)
   - Compares P/E, P/B with peers
   - Calculates PEG ratio
   - Assesses pricing fairness

10. **Utilization Agent** (3-5 min)
   - Analyzes Objects of the Issue
   - Checks Fresh Issue vs OFS split
   - Evaluates debt repayment vs growth

11. **Investment Committee Agent** (5-7 min)
    - Synthesizes all reports
    - Calculates Weighted Scorecard
    - Generates Final Verdict

**Output**:
- 11 detailed agent analyses
- Final Scorecard and Verdict

#### Phase 4: Critique & Verification (5-10 minutes)

**Steps**:
1. **Self-Critic Review**
   - Validate numerical claims
   - Check page references
   - Identify contradictions
   - Flag hallucinations

2. **Consensus Building**
   - Reconcile agent findings
   - Prioritize key insights
   - Build unified narrative

**Output**:
- Validated findings
- Confidence scores
- Identified gaps

#### Phase 5: Report Generation (5-10 minutes)

**Steps**:
1. **Report Assembly**
   - Combine agent outputs
   - Structure report sections
   - Add executive summary

2. **Markdown Generation**
   - Apply report template
   - Format tables and lists
   - Add table of contents

3. **PDF Conversion** (optional)
   - Convert markdown to HTML
   - Generate styled PDF with WeasyPrint

**Output**:
- Final markdown report (report.md)
- PDF report (report.pdf)
- Summary JSON (metadata)

### 8.2 Total Processing Time

**Estimated Time**:
- 300-page RHP: ~60-90 minutes
- 500-page RHP: ~90-120 minutes

**Breakdown**:
- Ingestion: 25%
- Embedding: 20%
- Analysis: 45%
- Reporting: 10%

---

## 9. Implementation Roadmap

### Phase 1: MVP (Weeks 1-4)

**Goal**: Basic end-to-end pipeline

**Deliverables**:
1. PDF ingestion pipeline
   - Text extraction (PyMuPDF)
   - Basic table extraction
   - Section detection

2. Simple chunking and embedding
   - Fixed-size chunks
   - Basic embedding generation
   - In-memory storage (no vector DB yet)

3. Single agent (Architect)
   - Basic LLM integration
   - Simple prompt template
   - RAG-based analysis

4. Basic markdown report
   - Template-based generation
   - Section summaries
   - Key findings

**Tech Stack (MVP)**:
- PyMuPDF for parsing
- Sentence-transformers for embeddings
- Hugging Face Inference API
- Simple Python scripts
- Markdown output

**Success Criteria**:
- Process one RHP successfully
- Generate readable report
- Extract basic insights

### Phase 2: Enhanced Analysis (Weeks 5-8)

**Goal**: Multi-agent system with vector search

**Deliverables**:
1. Vector database integration
   - Set up Qdrant
   - Implement semantic search
   - Add metadata filtering

2. Multi-agent system
   - Implement 5 core agents
   - LangGraph orchestration
   - State management

3. Enhanced table extraction
   - Financial statement parsing
   - Ratio calculations
   - Trend analysis

4. Improved reporting
   - Layered outputs (TL;DR, detailed)
   - Better formatting
   - Page references

**Tech Stack (Phase 2)**:
- Qdrant vector DB
- LangGraph for orchestration
- Multiple LLM models
- SQLite for structured data

**Success Criteria**:
- All 5 agents working
- High-quality financial analysis
- Comprehensive reports

### Phase 3: Production Polish (Weeks 9-12)

**Goal**: Robust, production-ready system

**Deliverables**:
1. Self-Critic agent
   - Validation logic
   - Error detection
   - Confidence scoring

2. CLI interface
   - Easy invocation
   - Progress tracking
   - Error handling

3. PDF report generation
   - Styled PDF output
   - Professional formatting

4. Error handling and logging
   - Comprehensive error catching
   - Detailed logs
   - Recovery mechanisms

5. Testing suite
   - Unit tests
   - Integration tests
   - Test with multiple RHPs

**Tech Stack (Phase 3)**:
- WeasyPrint for PDF
- Click/Typer for CLI
- Loguru for logging
- Pytest for testing

**Success Criteria**:
- Handle edge cases gracefully
- Professional-quality reports
- Reliable execution

### Phase 4: Optional Enhancements (Weeks 13+)

**Potential Features**:
1. Interactive Q&A
   - Command-line query interface
   - Real-time answers

2. Comparative analysis
   - Compare multiple IPOs
   - Industry benchmarking

3. Local LLM integration
   - Reduce API costs
   - Faster processing

4. Web interface
   - Simple Flask/FastAPI app
   - Upload and analyze
   - View reports online

---

## 10. Testing Strategy

### 10.1 Unit Tests

**Components to Test**:
```python
# tests/test_pdf_processor.py
def test_pdf_extraction():
    """Test PDF text extraction"""
    processor = PDFProcessor("test_data/sample.pdf")
    pages = processor.extract_with_pymupdf()
    assert len(pages) > 0
    assert all(page.text for page in pages)

# tests/test_table_extractor.py
def test_table_extraction():
    """Test table extraction"""
    extractor = TableExtractor()
    tables = extractor.extract_tables("test_data/sample.pdf", (10, 20))
    assert len(tables) > 0

# tests/test_financial_parser.py
def test_ratio_calculation():
    """Test financial ratio calculations"""
    parser = FinancialParser()
    financials = FinancialData(
        fiscal_year="2023",
        revenue=1000,
        pat=100,
        total_equity=500
    )
    ratios = parser.calculate_ratios(financials)
    assert ratios['roe'] == 0.2  # 100/500

# tests/test_embedding.py
def test_embedding_generation():
    """Test embedding generation"""
    generator = EmbeddingGenerator()
    texts = ["This is a test", "Another test"]
    embeddings = generator.generate(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
```

### 10.2 Integration Tests

```python
# tests/test_workflow.py
def test_end_to_end_workflow():
    """Test complete workflow"""
    workflow = RHPAnalysisWorkflow()
    result = workflow.run("test_data/sample_rhp.pdf")

    # Check all phases completed
    assert result['current_phase'] == 'completed'
    assert result['final_report'] is not None

    # Check agent outputs
    assert result['architect_analysis'] is not None
    assert result['forensic_analysis'] is not None

    # Check report file generated
    import os
    assert os.path.exists(f"outputs/{result['document_id']}/report.md")
```

### 10.3 Test Data

**Required Test Files**:
1. Small RHP (50 pages) - Quick tests
2. Medium RHP (200 pages) - Standard tests
3. Large RHP (500 pages) - Performance tests
4. Scanned RHP - OCR tests
5. Complex tables - Table extraction tests

### 10.4 Quality Metrics

**Automated Checks**:
```python
class QualityChecker:
    """
    Validates analysis quality
    """
    def check_report(self, report: Report) -> Dict[str, bool]:
        checks = {
            'has_executive_summary': len(report.tldr) > 100,
            'has_investment_thesis': len(report.investment_thesis) > 0,
            'has_risks': len(report.key_risks) > 0,
            'has_page_references': self._check_citations(report),
            'numeric_consistency': self._check_numbers(report),
            'no_hallucinations': self._check_hallucinations(report)
        }
        return checks
```

---

## 11. Performance Considerations

### 11.1 Optimization Strategies

**1. Parallel Processing**:
```python
import concurrent.futures

class ParallelProcessor:
    """
    Parallel execution for independent tasks
    """
    def process_pages_parallel(self, pages: List[Page]) -> List[Page]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            processed = list(executor.map(self.process_page, pages))
        return processed
```

**2. Caching**:
```python
from functools import lru_cache

class CachedEmbedding:
    """
    Cache embeddings to avoid recomputation
    """
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
```

**3. Batch Processing**:
```python
# Process chunks in batches
BATCH_SIZE = 32
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    embeddings = embedding_model.encode([c.text for c in batch])
```

**4. GPU Utilization**:
```python
# Use GPU for local models
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name, device=device)
```

### 11.2 Resource Management

**Memory Management**:
```python
class MemoryEfficientProcessor:
    """
    Process large documents without loading everything in memory
    """
    def process_in_chunks(self, pdf_path: str, chunk_size: int = 10):
        with fitz.open(pdf_path) as doc:
            for i in range(0, len(doc), chunk_size):
                # Process 10 pages at a time
                pages = [doc[j] for j in range(i, min(i+chunk_size, len(doc)))]
                yield self.process_pages(pages)
                # Pages go out of scope and get garbage collected
```

**API Rate Limiting**:
```python
import time
from functools import wraps

def rate_limit(calls_per_minute: int = 60):
    """Decorator to rate limit API calls"""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

class RateLimitedLLM:
    @rate_limit(calls_per_minute=30)
    def generate(self, prompt: str) -> str:
        return self.client.text_generation(prompt)
```

### 11.3 Performance Targets

**Processing Time Goals**:
- 300-page RHP: < 60 minutes
- 500-page RHP: < 90 minutes
- Text extraction: < 5 seconds/page
- Embedding: < 1 second/chunk
- Agent analysis: < 10 minutes/agent

**Resource Usage**:
- Peak memory: < 8 GB
- Disk space: < 2 GB per RHP
- Network bandwidth: Depends on API usage

---

## 12. Error Handling & Logging

### 12.1 Error Handling Strategy

```python
from loguru import logger
from typing import Optional

class RobustProcessor:
    """
    Implements comprehensive error handling
    """
    def process_with_fallback(self, pdf_path: str) -> Optional[Dict]:
        try:
            # Primary processing
            result = self.primary_processing(pdf_path)
            return result

        except PDFProcessingError as e:
            logger.warning(f"Primary processing failed: {e}")
            try:
                # Fallback to alternative method
                result = self.fallback_processing(pdf_path)
                return result
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return None

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return None
```

### 12.2 Logging Configuration

```python
from loguru import logger
import sys

# Configure logging
logger.remove()  # Remove default handler

# Console logging
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# File logging
logger.add(
    "logs/rhp_analyzer_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# Error-specific log
logger.add(
    "logs/errors.log",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)
```

### 12.3 Error Categories

```python
class RHPAnalyzerError(Exception):
    """Base exception for RHP Analyzer"""
    pass

class PDFProcessingError(RHPAnalyzerError):
    """PDF parsing failed"""
    pass

class TableExtractionError(RHPAnalyzerError):
    """Table extraction failed"""
    pass

class EmbeddingError(RHPAnalyzerError):
    """Embedding generation failed"""
    pass

class LLMError(RHPAnalyzerError):
    """LLM API call failed"""
    pass

class ValidationError(RHPAnalyzerError):
    """Data validation failed"""
    pass
```

### 12.4 Progress Tracking

```python
from tqdm import tqdm

class ProgressTracker:
    """
    Track processing progress
    """
    def track_workflow(self, phases: List[str]):
        with tqdm(total=len(phases), desc="Overall Progress") as pbar:
            for phase in phases:
                logger.info(f"Starting phase: {phase}")
                self.execute_phase(phase)
                pbar.update(1)

    def track_page_processing(self, total_pages: int):
        for page_num in tqdm(range(total_pages), desc="Processing pages"):
            self.process_page(page_num)
```

---

## 13. Future Enhancements

### 13.1 Short-term Enhancements

1. **Interactive Q&A Mode**
   ```python
   # CLI mode for asking questions
   $ python rhp_analyzer.py query --document report_123
   > What is the company's revenue growth?
   > What are the main risks?
   ```

2. **Comparative Analysis**
   - Compare multiple IPOs side-by-side
   - Industry peer comparison
   - Historical trend analysis

3. **Custom Report Templates**
   - User-defined report sections
   - Configurable detail levels
   - Export formats (Word, PowerPoint)

### 13.2 Long-term Enhancements

1. **Web Interface**
   - Upload RHP via browser
   - View analysis in dashboard
   - Interactive drill-down

2. **Real-time Market Data Integration**
   - Current market cap
   - Peer valuations
   - Industry trends

3. **Multi-language Support**
   - Handle Hindi sections in RHPs
   - Regional language support

4. **Collaborative Features**
   - Share analysis with team
   - Comments and annotations
   - Version control

### 13.3 Advanced Features

1. **Fine-tuned Models**
   - Fine-tune LLM on Indian IPO data
   - Custom NER for financial entities
   - Sector-specific embeddings

2. **Automated Monitoring**
   - Track IPO performance post-listing
   - Alert on material changes
   - Quarterly update analysis

3. **Portfolio Integration**
   - Link to portfolio tracking
   - Investment decision support
   - Risk aggregation

---

## Appendices

### Appendix A: Sample Report Structure

```markdown
# [Company Name] IPO Initiation Note
**Verdict: [SUBSCRIBE / AVOID]** | **Score: [XX]/100**

## 1. Investment Verdict (The Committee View)
### Scorecard
| Parameter | Score (0-10) | Weight | Weighted Score |
| :--- | :---: | :---: | :---: |
| Financial Health | [X] | 30% | [Y] |
| Valuation Comfort | [X] | 20% | [Y] |
| Governance Quality | [X] | 20% | [Y] |
| Business Moat | [X] | 15% | [Y] |
| Industry Tailwinds | [X] | 15% | [Y] |
| **TOTAL** | | | **[XX]** |

### Scenario Dashboard
| Scenario | FY26E EPS (₹) | ROE % | ROIC % | Net Debt/EBITDA | Implied Upside @ Floor | Implied Upside @ Cap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base |  |  |  |  |  |  |
| Bull |  |  |  |  |  |  |
| Stress |  |  |  |  |  |  |

### The Thesis
**The Hook (Why Buy):**
*   [Point 1]
*   [Point 2]

**The Drag (Why Skip):**
*   [Point 1]
*   [Point 2]

**Final Recommendation:** [Detailed Verdict]

---

## 2. IPO At a Glance
| Metric | Value | Metric | Value |
| :--- | :--- | :--- | :--- |
| **Price Band** | ₹[X] - ₹[Y] | **Market Cap** | ₹[Z] Cr |
| **Open Date** | [Date] | **P/E Ratio** | [X]x |
| **Lot Size** | [X] Shares | **Fresh Issue** | ₹[A] Cr |
| **Registrar** | [Name] | **OFS** | ₹[B] Cr |

### Objects of the Issue (₹ Cr)
| Use Case | Amount | % of Fresh Issue | Deployment Timeline | Monitoring Agency |
| :--- | :---: | :---: | :--- | :--- |
| Capex |  |  |  |  |
| Debt Repayment |  |  |  |  |
| Working Capital |  |  |  |  |
| General Corporate Purposes |  |  |  |  |

---

## 3. Industry & Market Opportunity
*   **TAM:** [Size] (CAGR: [X]%)
*   **Key Drivers:** [Drivers]
*   **Competitive Landscape:** [Comparison]

## 4. Business Deep Dive
### Revenue Mix
[Chart/Table]

### SWOT Analysis
*   **Strengths:** ...
*   **Weaknesses:** ...
*   **Opportunities:** ...
*   **Threats:** ...

## 5. Financial Analysis (The Numbers)
### Condensed P&L (Restated)
| Particulars (₹ Cr) | FY23 | FY24 | FY25 |
| :--- | :--- | :--- | :--- |
| Revenue | ... | ... | ... |
| EBITDA | ... | ... | ... |
| PAT | ... | ... | ... |

### Key Ratios
*   **ROE:** ...
*   **ROCE:** ...
*   **Debt/Equity:** ...

### Forensic Checks
*   **Cash Conversion Cycle:** [Analysis]
*   **CFO vs EBITDA:** [Analysis]

## 6. Governance & Management
*   **Promoter Background:** [Details]
*   **Management Quality:** [Details]
*   **Related Party Transactions:** [Analysis]

## 7. Valuation & Peer Comparison
| Company | P/E | P/B | EV/EBITDA | RoNW |
| :--- | :--- | :--- | :--- | :--- |
| **[Target]** | **[X]** | **[X]** | **[X]** | **[X]** |
| Peer A | [Y] | [Y] | [Y] | [Y] |

**Valuation Verdict:** [Undervalued/Overvalued]

### Valuation Ladder
| Price Point | Implied Market Cap (₹ Cr) | P/E | EV/EBITDA | Premium/(Discount) vs Peer Median | Comment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Floor |  |  |  |  |  |
| Cap |  |  |  |  |  |

## 8. Risk Factors (Critical)
| Risk | Category | Severity | Amount/Metric | RHP Reference |
| :--- | :--- | :---: | :---: | :--- |
|  |  |  |  |  |

## 9. Legal & Litigation
| Entity | Count | Amount (₹ Cr) | % of Net Worth | Next Milestone | Section-Page |
| :--- | :---: | :---: | :---: | :--- | :--- |
| Company |  |  |  |  |  |
| Promoters |  |  |  |  |  |
| Directors |  |  |  |  |  |
| Subsidiaries |  |  |  |  |  |

Contingent Liabilities Summary: ₹[A] Cr ( [Y]% of Net Worth )

---
*Generated by RHP Analyzer on [Date]*

## Citations
1. [Financial Information, p. XXX]
2. [Objects of the Issue, p. YYY]
3. ... (auto-generated from Citation Manager)
```

### Appendix B: Configuration File

```yaml
# config.yaml

# Model configuration
models:
  context_model: "Qwen/Qwen2.5-32B-Instruct"
  reasoning_model: "meta-llama/Llama-3.3-70B-Instruct"
  local_model: "llama3.2:8b-instruct-fp16"
  embedding_model: "nomic-ai/nomic-embed-text-v1.5"

# API configuration
huggingface:
  token: ${HF_TOKEN}
  timeout: 300
  max_retries: 3

# Vector database
vector_db:
  type: "qdrant"
  path: "./qdrant_storage"
  collection_name: "rhp_chunks"
  embedding_dim: 1024

# Processing configuration
processing:
  chunk_size: 1000
  chunk_overlap: 100
  batch_size: 32
  max_workers: 4

# Agent configuration
agents:
  enabled:
    - architect
    - forensic
    - red_flag
    - governance
    - legal
    - self_critic

  architect:
    temperature: 0.3
    max_tokens: 4096

  forensic:
    temperature: 0.2
    max_tokens: 4096

  # ... more agent configs

# Report configuration
report:
  output_format: ["markdown", "pdf"]
  include_sections:
    - executive_summary
    - investment_thesis
    - business_overview
    - financial_analysis
    - risk_assessment
    - governance
    - legal_review
    - red_flags

  tldr_length: 250  # words
  detail_level: "comprehensive"  # or "concise"

# Logging
logging:
  level: "INFO"
  console: true
  file: "logs/rhp_analyzer.log"
  rotation: "1 day"
  retention: "30 days"
```

### Appendix C: CLI Usage

```bash
# Basic usage
python rhp_analyzer.py analyze path/to/rhp.pdf

# With custom config
python rhp_analyzer.py analyze path/to/rhp.pdf --config config.yaml

# Specify output format
python rhp_analyzer.py analyze path/to/rhp.pdf --format pdf

# Query mode
python rhp_analyzer.py query --document doc_123
> What is the revenue CAGR?

# Batch processing
python rhp_analyzer.py batch path/to/rhp_folder/

# Generate summary only
python rhp_analyzer.py analyze path/to/rhp.pdf --summary-only

# List processed documents
python rhp_analyzer.py list

# View specific report
python rhp_analyzer.py view doc_123
```
