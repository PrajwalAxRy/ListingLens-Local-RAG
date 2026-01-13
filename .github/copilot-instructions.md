# RHP Analyzer - Copilot Agent Instructions

## Project Overview

You are assisting with **RHP Analyzer**, a local AI-powered system for analyzing Indian IPO Red Herring Prospectus (RHP) documents. The system generates comprehensive investment analysis reports using a multi-agent architecture with LLMs.

**Key Context:**
- Analyzes 300-500 page RHP PDF documents
- Runs locally (single user, single RHP at a time)
- Uses Hugging Face Inference API (unlimited budget)
- Outputs analyst-grade markdown/PDF reports
- Self-contained with no external financial APIs

---

## Architecture Quick Reference

```
User CLI → LangGraph Orchestration → Ingestion Tier → Vector/SQL Storage → Multi-Agent Analysis → Report Generation
```

### Core Components
| Layer | Technologies | Purpose |
|-------|-------------|---------|
| PDF Processing | PyMuPDF, pdfplumber, camelot-py, Tesseract | Extract text, tables, sections |
| Storage | Qdrant (vectors), SQLite (structured), Local FS | Persist data and embeddings |
| Embeddings | nomic-embed-text-v1.5 / gte-large-en-v1.5 | Semantic search (768-1024 dim) |
| LLMs | Qwen2.5-32B (context), Llama-3.3-70B (reasoning), Llama-3.2-8B (local summarizer) |
| Orchestration | LangGraph state machine | Workflow coordination |
| Reports | Jinja2 templates, WeasyPrint | Markdown/PDF output |

---

## Coding Standards & Conventions

### Python Standards
- **Python Version**: 3.10+
- **Type Hints**: Required for all function signatures
- **Dataclasses**: Use `@dataclass` for all data models (see Section 5 of blueprint)
- **Async**: Not required (single-threaded processing)
- **Logging**: Use `loguru` for all logging
- **Error Handling**: Custom exceptions inherit from `RHPAnalyzerError`

### Naming Conventions
```python
# Classes: PascalCase, descriptive
class PDFProcessor, TableExtractor, ForensicAccountantAgent

# Functions/Methods: snake_case, verb-first
def extract_tables(), calculate_ratios(), analyze_governance()

# Constants: UPPER_SNAKE_CASE
CHUNK_SIZE = 1000
SECTOR_BENCHMARKS = {...}

# Files: snake_case.py
pdf_processor.py, forensic_agent.py, working_capital_analyzer.py
```

### Directory Structure
```
src/
├── ingestion/          # PDF parsing, table extraction, section mapping
├── storage/            # Qdrant, SQLite, file storage wrappers
├── agents/             # All analysis agents (one file per agent)
├── models/             # Pydantic/dataclass data models
├── financial/          # Financial parsing, ratios, projections
├── report/             # Template rendering, markdown/PDF generation
├── orchestration/      # LangGraph workflow definitions
└── utils/              # Shared utilities, logging, config
```

---
### Extra instructions
Note: make sure to activate python venv at the beginning of the code generation if any code is to be executed.

## Agent Implementation Guidelines

### Base Agent Pattern
All agents MUST follow this structure:

```python
from abc import ABC, abstractmethod
from typing import List, Dict
from models.state import AnalysisState
from models.outputs import AgentAnalysis

class BaseAgent(ABC):
    def __init__(self, llm, vector_store, citation_mgr):
        self.llm = llm
        self.vector_store = vector_store
        self.citation_mgr = citation_mgr
        self.name = self.__class__.__name__

    @abstractmethod
    def analyze(self, state: AnalysisState) -> AgentAnalysis:
        """Perform agent-specific analysis - MUST be implemented"""
        pass

    def retrieve_context(self, query: str, filters: Dict, top_k: int = 10) -> List[Chunk]:
        """RAG retrieval from vector store"""
        pass

    def cite(self, claim_id: str, chunk: Chunk):
        """Attach citation for audit trail - ALWAYS use for numerical claims"""
        pass
```

### Agent Roster (11 Agents)
| Agent | Responsibility |
|-------|---------------|
| **BusinessAnalystAgent** | Revenue mix, SWOT, capacity utilization, order book analysis |
| **IndustryAnalystAgent** | TAM, CAGR, competitive landscape, barriers to entry |
| **ManagementAgent** | KMP profiles, remuneration, track record, attrition |
| **CapitalStructureAgent** | WACA, pre-IPO investor exits, float analysis, lock-in calendar |
| **ForensicAccountantAgent** | CFO vs EBITDA, working capital forensics, window dressing detection |
| **RedFlagAgent** | Quantified risks with severity ratings (CRITICAL/MAJOR/MINOR) |
| **GovernanceAgent** | Governance score (0-10), promoter conflicts, RPT analysis |
| **LegalAgent** | Litigation quantification, contingent liabilities, timeline risks |
| **ValuationAgent** | Peer multiples, PEG ratio, premium/discount vs median |
| **UtilizationAgent** | Objects of Issue analysis, deployment timeline, readiness flags |
| **InvestmentCommitteeAgent** | Final scorecard (0-100), SUBSCRIBE/AVOID/VETO verdict |

### Agent Output Requirements
Every agent output MUST include:
```python
@dataclass
class AgentAnalysis:
    agent_name: str
    analysis: str           # Full analysis text
    key_findings: List[str] # Bullet points
    concerns: List[str]     # Issues identified
    confidence: float       # 0.0 to 1.0
    sources: List[int]      # Page references
```

---

## Financial Analysis Rules

### Critical Red Flags (Auto-Veto Triggers)
```python
# These conditions automatically trigger AVOID recommendation:
VETO_CONDITIONS = {
    "governance_score < 5": "Governance failure",
    "litigation_to_networth > 10%": "Excessive litigation risk",
    "promoter_pledge > 25%": "High pledge risk",
    "criminal_cases > 0": "Criminal litigation against promoters",
}
```

### Forensic Checks (Always Apply)
| Check | Red Flag Threshold | Signal |
|-------|-------------------|--------|
| Channel Stuffing | Receivables growth > Revenue growth + 10 ppts | Window dressing |
| Paper Profits | CFO/EBITDA < 50% | Earnings quality issue |
| Working Capital Stress | Receivable days ↑ >10 days YoY | Collection problems |
| Cash Burn | Negative FCF for 2+ years | Funding risk |
| Inventory Piling | Inventory days ↑ >15 days YoY | Demand concerns |

### Investment Scorecard (Weighted 0-100)
| Category | Weight | Scoring Basis |
|----------|--------|---------------|
| Financial Health | 30% | ROE bands (>20%=10, 15-20=8, 10-15=6, <10=3), CFO quality |
| Valuation Comfort | 20% | Peer premium/discount, PEG ratio |
| Governance Quality | 20% | Start at 10, deduct for breaches |
| Business Moat | 15% | TAM, barriers, customer concentration |
| Industry Tailwinds | 15% | Market growth, policy support |

---

## RHP-Specific Domain Knowledge

### Key Sections to Parse
| Section | Data to Extract |
|---------|-----------------|
| Summary of Prospectus | Issue size, price band, lot size, dates |
| Risk Factors | Company-specific risks (IGNORE generic boilerplate) |
| Business Overview | Revenue mix by product/geography/customer, capacity |
| Financial Statements | P&L, Balance Sheet, Cash Flow (3-5 years restated) |
| Our Promoters | DIN, age, experience, qualifications, conflicts |
| Capital Structure | WACA, pre-IPO placements, OFS breakdown, history |
| Objects of the Issue | Use of proceeds, deployment timeline, monitoring agency |
| Indebtedness | Debt structure, covenants, maturity profile |
| Outstanding Litigation | Company, promoter, director cases with amounts |
| Related Party Transactions | RPT as % of revenue and expenses |
| Basis for Issue Price | Peer comparison table (watch for missing peers) |

### Indian IPO Context
```python
# Currency formats
CRORE = 10_000_000   # ₹1 Cr = ₹10 million
LAKH = 100_000       # ₹1 Lakh = ₹100 thousand

# Fiscal year
# FY25 = April 2024 - March 2025

# Regulatory bodies
REGULATORS = ["SEBI", "RBI", "ROC", "NCLT", "NCLAT"]

# Lock-in periods
LOCK_INS = {
    "promoters": "3 years",
    "anchor_investors": "90 days",
    "pre_ipo_investors": "6-12 months",
}

# Quota allocation (book-built issues)
QUOTAS = {
    "QIB": "50%",    # Qualified Institutional Buyers
    "NII": "15%",    # Non-Institutional Investors
    "Retail": "35%", # Retail Individual Investors
}
```

---

## Common Implementation Patterns

### Table Extraction Strategy
```python
# Try multiple strategies in order of reliability:
TABLE_STRATEGIES = ['pdfplumber', 'camelot', 'unstructured']

# Table types to classify:
TABLE_TYPES = [
    'financial_statement',   # P&L, Balance Sheet, Cash Flow
    'shareholding_pattern',  # Pre/post IPO holding
    'peer_comparison',       # Basis for issue price
    'litigation_summary',    # Outstanding cases
    'rpt_table',            # Related party transactions
    'indebtedness',         # Debt structure
    'objects_of_issue',     # Use of proceeds
]
```

### Chunking Parameters
```python
CHUNK_CONFIG = {
    "chunk_size": 1000,      # tokens (range: 500-1500)
    "chunk_overlap": 100,    # tokens
    "batch_size": 32,        # for embedding generation
    "embedding_dim": 1024,   # nomic-embed-text-v1.5
}
```

### LLM Configuration
```python
LLM_CONFIG = {
    "context_model": "Qwen/Qwen2.5-32B-Instruct",      # Large context
    "reasoning_model": "meta-llama/Llama-3.3-70B-Instruct",  # Analysis
    "local_summarizer": "llama3.2:8b-instruct-fp16",  # Via Ollama

    # Temperature by task
    "extraction_temp": 0.1,   # Factual extraction
    "analysis_temp": 0.2,     # Reasoning tasks
    "synthesis_temp": 0.3,    # Report generation
}
```

---

## Citation Requirements

### Mandatory Citations
Every numerical claim or qualitative assertion MUST have a citation:

```python
@dataclass
class CitationRecord:
    claim_id: str
    section: str        # RHP section name
    page: int           # Page number
    paragraph_label: Optional[str]
    text_snippet: str   # Max 280 chars of source text
```

### Citation Format in Reports
```markdown
Revenue grew 45% CAGR over FY22-FY24 [Financial Information, p. 156]
Promoter holds 15 other directorships [Other Directorships of Promoters, p. 203]
```

### Self-Critic Enforcement
- Self-Critic agent REJECTS outputs lacking citations
- Claims without page references are flagged as potential hallucinations
- Report footnotes auto-generated from CitationManager

---

## Report Output Structure

### Standard Report Format
```markdown
# [Company Name] IPO Initiation Note
**Verdict: [SUBSCRIBE/AVOID]** | **Score: [XX]/100**

## 1. Investment Verdict
### Scorecard Table (weighted scores)
### Scenario Dashboard (Base/Bull/Stress projections)
### The Thesis (Why Buy / Why Skip bullets)

## 2. IPO At a Glance
### Key Metrics Table
### Objects of Issue Breakdown

## 3. Industry & Market Opportunity
## 4. Business Deep Dive
## 5. Financial Analysis
## 6. Governance & Management
## 7. Valuation & Peer Comparison
## 8. Risk Factors (Critical)
## 9. Legal & Litigation

## Citations
[Auto-generated numbered footnotes]
```

### Output Files
```
outputs/{document_id}/
├── report.md              # Primary markdown report
├── report.pdf             # Styled PDF via WeasyPrint
└── analysis_summary.json  # Structured metadata
```

---

## Error Handling

### Exception Hierarchy
```python
class RHPAnalyzerError(Exception):
    """Base exception for all RHP Analyzer errors"""
    pass

class PDFProcessingError(RHPAnalyzerError): pass
class TableExtractionError(RHPAnalyzerError): pass
class EmbeddingError(RHPAnalyzerError): pass
class LLMError(RHPAnalyzerError): pass
class ValidationError(RHPAnalyzerError): pass
class CitationError(RHPAnalyzerError): pass
```

### Fallback Strategies
```python
FALLBACK_CHAINS = {
    "pdf_parsing": ["pymupdf", "pdfplumber", "tesseract_ocr"],
    "table_extraction": ["pdfplumber", "camelot", "unstructured"],
    "llm_calls": "retry_3x_exponential_backoff",
}
```

---

## Testing Expectations

### Test Categories
- **Unit Tests**: All parsers, calculators, individual agents
- **Integration Tests**: Full workflow from PDF to report
- **Stress Tests**: 500-page RHP processing

### Test Data
```python
TEST_RHPS = {
    "quick": "50_page_sample.pdf",     # <5 min
    "standard": "200_page_sample.pdf", # ~30 min
    "stress": "500_page_sample.pdf",   # ~90 min
}
```

### Quality Metrics
| Metric | Target |
|--------|--------|
| Processing time (300 pages) | < 60 minutes |
| Peak memory usage | < 8 GB |
| Text extraction accuracy | > 95% |
| Table extraction accuracy | > 85% |

---

## Key Data Models Reference

```python
# Analysis state (LangGraph)
class AnalysisState(TypedDict):
    document_id: str
    pdf_path: str
    pages: List[Dict]
    sections: Dict[str, Dict]
    tables: List[Dict]
    entities: Dict[str, List[str]]
    financial_data: Dict[str, Dict]
    chunks: List[Dict]
    # Agent outputs (one per agent)
    architect_analysis: Optional[str]
    forensic_analysis: Optional[str]
    # ... etc
    final_report: Optional[str]
    errors: List[str]
    warnings: List[str]

# Financial data per year
@dataclass
class FinancialData:
    fiscal_year: str
    revenue: float
    ebitda: float
    ebitda_margin: float
    pat: float
    total_assets: float
    total_equity: float
    total_debt: float
    roe: float
    roce: float
    debt_equity_ratio: float
    cash_conversion_cycle: float

# Investment scorecard
@dataclass
class Scorecard:
    financial_health_score: float   # 0-10, weight 30%
    valuation_comfort_score: float  # 0-10, weight 20%
    governance_score: float         # 0-10, weight 20%
    business_moat_score: float      # 0-10, weight 15%
    industry_tailwind_score: float  # 0-10, weight 15%
    total_score: float              # 0-100
    verdict: str                    # "Subscribe" | "Avoid"
    veto_flag: bool                 # True if governance < 5
```

---

## When Generating Code

### Always Do:
1. ✅ Use type hints for all function parameters and return values
2. ✅ Include docstrings with purpose, params, and returns
3. ✅ Handle Indian number formats (crore, lakh)
4. ✅ Include page references when extracting RHP data
5. ✅ Use dataclasses for structured data
6. ✅ Log at appropriate levels (DEBUG/INFO/WARNING/ERROR)
7. ✅ Raise specific exceptions from the hierarchy
8. ✅ Add citations for every numerical claim

### Never Do:
1. ❌ Skip type hints
2. ❌ Make unsourced numerical claims
3. ❌ Ignore edge cases (scanned PDFs, missing tables)
4. ❌ Use generic exception handling
5. ❌ Hardcode paths or credentials
6. ❌ Process multiple RHPs concurrently (not supported)

---

## Blueprint Reference

For complete implementation specifications, component details, and data model definitions, see:
- [blueprint.md](../blueprint.md) - Full technical design document
- [milestones.md](../milestones.md) - Implementation roadmap and phases
