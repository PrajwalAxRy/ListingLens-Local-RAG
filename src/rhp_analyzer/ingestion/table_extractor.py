"""
Table Extraction Module for RHP Analyzer.

This module provides multi-strategy table extraction from PDF documents,
with special handling for financial statements common in Indian RHP documents.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available. Table extraction will be limited.")

try:
    import camelot

    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("camelot-py not available. Fallback table extraction disabled.")


class TableType(Enum):
    """Classification of table types found in RHP documents."""

    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    SHAREHOLDING_PATTERN = "shareholding_pattern"
    OBJECT_OF_ISSUE = "object_of_issue"
    RELATED_PARTY = "related_party"
    KEY_METRICS = "key_metrics"
    PEER_COMPARISON = "peer_comparison"
    INDEBTEDNESS = "indebtedness"
    LITIGATION_SUMMARY = "litigation_summary"
    CONTINGENT_LIABILITIES = "contingent_liabilities"
    OTHER = "other"


@dataclass
class Table:
    """Represents an extracted table from a PDF document."""

    table_id: str
    page_num: int
    rows: list[list[str]]
    headers: list[str] = field(default_factory=list)
    table_type: TableType = TableType.OTHER
    confidence: float = 0.0
    extraction_method: str = "pdfplumber"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        """Return the number of data rows (excluding header)."""
        return len(self.rows)

    @property
    def col_count(self) -> int:
        """Return the number of columns."""
        if self.headers:
            return len(self.headers)
        return len(self.rows[0]) if self.rows else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert table to dictionary representation."""
        return {
            "table_id": self.table_id,
            "page_num": self.page_num,
            "headers": self.headers,
            "rows": self.rows,
            "table_type": self.table_type.value,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "metadata": self.metadata,
        }


@dataclass
class FinancialData:
    """Parsed financial data from a financial statement table."""

    fiscal_year: str
    metric_name: str
    value: float | None
    unit: str = "crore"  # crore, lakh, or actual
    raw_value: str = ""

    def to_actual(self) -> float | None:
        """Convert value to actual amount based on unit."""
        if self.value is None:
            return None
        if self.unit == "crore":
            return self.value * 10_000_000
        elif self.unit == "lakh":
            return self.value * 100_000
        return self.value


class TableClassifier:
    """Classifies tables based on their content and structure."""

    # Keywords for identifying table types
    TABLE_TYPE_KEYWORDS = {
        TableType.INCOME_STATEMENT: [
            "revenue",
            "income",
            "profit",
            "loss",
            "ebitda",
            "expenses",
            "statement of profit",
            "profit and loss",
            "p&l",
            "operating income",
            "restated statement",
            "restated consolidated",
            "restated standalone",
        ],
        TableType.BALANCE_SHEET: [
            "assets",
            "liabilities",
            "equity",
            "balance sheet",
            "net worth",
            "total assets",
            "current assets",
            "non-current",
            "shareholders",
            "restated assets",
            "statement of assets",
        ],
        TableType.CASH_FLOW: [
            "cash flow",
            "operating activities",
            "investing activities",
            "financing activities",
            "cash and cash equivalents",
            "net cash",
        ],
        TableType.SHAREHOLDING_PATTERN: [
            "shareholding",
            "shareholder",
            "promoter",
            "holding pattern",
            "pre-issue",
            "post-issue",
            "category of shareholders",
            "public holding",
        ],
        TableType.OBJECT_OF_ISSUE: [
            "object of the issue",
            "use of proceeds",
            "objects of the offer",
            "net proceeds",
            "deployment",
            "objects of the issue",
        ],
        TableType.RELATED_PARTY: [
            "related party",
            "related parties",
            "rpt",
            "transactions with",
            "key managerial",
            "promoter group",
        ],
        TableType.KEY_METRICS: [
            "key ratios",
            "financial ratios",
            "key performance",
            "operating metrics",
            "kpi",
            "roce",
            "roe",
        ],
        TableType.PEER_COMPARISON: [
            "peer comparison",
            "comparable companies",
            "basis for issue price",
            "peer group",
            "industry peers",
            "listed peers",
        ],
        TableType.INDEBTEDNESS: [
            "indebtedness",
            "borrowings",
            "outstanding debt",
            "loan",
            "secured",
            "unsecured",
            "debt structure",
        ],
        TableType.LITIGATION_SUMMARY: [
            "litigation",
            "outstanding legal",
            "legal proceedings",
            "pending cases",
            "material litigations",
        ],
        TableType.CONTINGENT_LIABILITIES: [
            "contingent liabilities",
            "contingent",
            "guarantees",
            "commitments",
            "claims against",
        ],
    }

    def classify(self, table: Table) -> tuple[TableType, float]:
        """
        Classify a table and return the type with confidence score.

        Args:
            table: The Table object to classify

        Returns:
            Tuple of (TableType, confidence_score)
        """
        # Combine headers and first few rows for analysis
        text_to_analyze = " ".join(table.headers).lower()

        # Add first 5 rows to analysis
        for row in table.rows[:5]:
            text_to_analyze += " " + " ".join(str(cell) for cell in row if cell).lower()

        # Score each table type
        scores: dict[TableType, int] = dict.fromkeys(TableType, 0)

        for table_type, keywords in self.TABLE_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_to_analyze:
                    scores[table_type] += 1

        # Find the best match
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]

        # Calculate confidence (normalize by number of keywords checked)
        total_keywords = len(self.TABLE_TYPE_KEYWORDS.get(best_type, []))
        if total_keywords > 0 and max_score > 0:
            confidence = min(max_score / (total_keywords * 0.3), 1.0)  # 30% of keywords = 100% confidence
        else:
            best_type = TableType.OTHER
            confidence = 0.0

        return best_type, confidence


class FinancialTableParser:
    """
    Parses financial statement tables and extracts structured data.
    Handles Indian accounting formats (lakhs, crores).
    """

    # Patterns for Indian number formats
    CRORE_PATTERN = re.compile(r"(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)\s*(?:cr\.?|crore)", re.IGNORECASE)
    LAKH_PATTERN = re.compile(r"(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)\s*(?:lakh|lac|l)", re.IGNORECASE)
    NUMBER_PATTERN = re.compile(r"^[₹$]?\s*[\(\-]?\s*[\d,]+\.?\d*\s*[\)]?$")
    PARENTHESES_NEGATIVE = re.compile(r"\(\s*([\d,]+\.?\d*)\s*\)")

    # Fiscal year patterns
    FY_PATTERNS = [
        re.compile(r"FY\s*\'?(\d{2,4})", re.IGNORECASE),  # FY24, FY'24, FY2024
        re.compile(r"(\d{4})\s*-\s*(\d{2,4})"),  # 2023-24, 2023-2024
        re.compile(r"(?:March|Mar)\s*(\d{4})", re.IGNORECASE),  # March 2024
        re.compile(r"(\d{4})"),  # Just a year
    ]

    def parse_number(self, value: str) -> float | None:
        """
        Parse a number from various Indian accounting formats.

        Args:
            value: String representation of the number

        Returns:
            Parsed float value or None if unparseable
        """
        if not value or not isinstance(value, str):
            return None

        value = value.strip()

        if not value or value.lower() in ["nil", "na", "n/a", "-", "—", "–"]:
            return 0.0

        # Check for parentheses (negative numbers) - handles spaces like "( 500 )"
        is_negative = bool(self.PARENTHESES_NEGATIVE.search(value)) or value.strip().startswith("-")

        # Remove parentheses if present
        match = self.PARENTHESES_NEGATIVE.search(value)
        if match:
            value = match.group(1)

        # Remove Rs./Rs prefix before other processing
        value = re.sub(r"^Rs\.?\s*", "", value, flags=re.IGNORECASE)

        # Remove currency symbols and whitespace
        value = re.sub(r"[₹$£€\s]", "", value)

        # Remove commas
        value = value.replace(",", "")

        # Remove leading minus sign (we already tracked negativity)
        value = value.lstrip("-")

        try:
            result = float(value)
            return -result if is_negative else result
        except ValueError:
            return None

    def detect_unit(self, table: Table) -> str:
        """
        Detect the unit used in a financial table (crore, lakh, or actual).

        Args:
            table: The Table to analyze

        Returns:
            Unit string: 'crore', 'lakh', or 'actual'
        """
        # Check headers and first few rows for unit indicators
        text_to_check = " ".join(table.headers).lower()
        for row in table.rows[:3]:
            text_to_check += " " + " ".join(str(cell) for cell in row if cell).lower()

        # Check for crore indicators (cr, cr., crore, crores)
        if any(term in text_to_check for term in ["in crore", "cr.", "crores", "crore", "₹ crore", "rs. crore"]):
            return "crore"
        # Also check for standalone "cr" pattern
        if " cr" in text_to_check or text_to_check.endswith("cr"):
            return "crore"

        # Check for lakh indicators
        if any(term in text_to_check for term in ["in lakh", "lakhs", "lakh", "lacs", "lac", "₹ lakh"]):
            return "lakh"

        if any(term in text_to_check for term in ["in million", "mn", "million"]):
            return "crore"  # Treat millions as crores for simplicity (1 cr = 10 mn)

        return "actual"

    def extract_fiscal_years(self, table: Table) -> list[str]:
        """
        Extract fiscal year labels from table headers.

        Args:
            table: The Table to analyze

        Returns:
            List of fiscal year strings found
        """
        fiscal_years = []

        for header in table.headers:
            for pattern in self.FY_PATTERNS:
                match = pattern.search(str(header))
                if match:
                    fiscal_years.append(header.strip())
                    break

        return fiscal_years

    def parse_financial_statement(self, table: Table) -> list[FinancialData]:
        """
        Parse a financial statement table into structured data.

        Args:
            table: The Table to parse

        Returns:
            List of FinancialData objects
        """
        results = []
        unit = self.detect_unit(table)
        fiscal_years = self.extract_fiscal_years(table)

        # Find which columns contain fiscal year data
        year_columns: dict[int, str] = {}
        for i, header in enumerate(table.headers):
            for fy in fiscal_years:
                if fy in str(header):
                    year_columns[i] = fy
                    break

        # Parse each row
        for row in table.rows:
            if not row or len(row) == 0:
                continue

            # First non-empty cell is typically the metric name
            metric_name = None
            for cell in row:
                if cell and str(cell).strip():
                    metric_name = str(cell).strip()
                    break

            if not metric_name:
                continue

            # Skip rows that appear to be headers or subtotals
            if metric_name.lower() in ["particulars", "description", "total", "sub-total"]:
                continue

            # Extract values for each fiscal year column
            for col_idx, fy in year_columns.items():
                if col_idx < len(row):
                    raw_value = str(row[col_idx]) if row[col_idx] else ""
                    parsed_value = self.parse_number(raw_value)

                    results.append(
                        FinancialData(
                            fiscal_year=fy,
                            metric_name=metric_name,
                            value=parsed_value,
                            unit=unit,
                            raw_value=raw_value,
                        )
                    )

        return results

    def calculate_growth_rates(self, data: list[FinancialData]) -> dict[str, dict[str, float]]:
        """
        Calculate year-over-year growth rates for financial metrics.

        Args:
            data: List of FinancialData from parsing

        Returns:
            Dict mapping metric names to dict of year -> growth rate
        """
        # Group by metric name
        by_metric: dict[str, list[FinancialData]] = {}
        for item in data:
            if item.metric_name not in by_metric:
                by_metric[item.metric_name] = []
            by_metric[item.metric_name].append(item)

        growth_rates: dict[str, dict[str, float]] = {}

        for metric, values in by_metric.items():
            # Sort by fiscal year
            sorted_values = sorted(values, key=lambda x: x.fiscal_year)

            growth_rates[metric] = {}
            for i in range(1, len(sorted_values)):
                prev = sorted_values[i - 1]
                curr = sorted_values[i]

                if prev.value and prev.value != 0 and curr.value is not None:
                    growth = ((curr.value - prev.value) / abs(prev.value)) * 100
                    growth_rates[metric][curr.fiscal_year] = round(growth, 2)

        return growth_rates


class TableExtractor:
    """
    Multi-strategy table extraction from PDF documents.

    Uses pdfplumber as the primary extraction method with camelot as fallback
    for complex tables. Provides table classification and financial parsing
    capabilities.
    """

    def __init__(self):
        """Initialize the table extractor with classifier and parser."""
        self.classifier = TableClassifier()
        self.financial_parser = FinancialTableParser()
        self.strategies = ["pdfplumber", "camelot"]
        self._table_counter = 0

    def _generate_table_id(self, page_num: int) -> str:
        """Generate a unique table ID."""
        self._table_counter += 1
        return f"table_{page_num:03d}_{self._table_counter:03d}"

    def extract_tables(
        self,
        pdf_path: str,
        page_range: tuple[int, int] | None = None,
        classify: bool = True,
    ) -> list[Table]:
        """
        Extract tables from a PDF document.

        Args:
            pdf_path: Path to the PDF file
            page_range: Optional tuple of (start_page, end_page) (1-indexed)
            classify: Whether to classify table types

        Returns:
            List of Table objects
        """
        if not PDFPLUMBER_AVAILABLE:
            logger.error("pdfplumber is required for table extraction")
            return []

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return []

        tables: list[Table] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Determine page range
                start_page = (page_range[0] - 1) if page_range else 0
                end_page = page_range[1] if page_range else len(pdf.pages)

                logger.info(f"Extracting tables from pages {start_page + 1} to {end_page}")

                for page_num in range(start_page, min(end_page, len(pdf.pages))):
                    page = pdf.pages[page_num]
                    page_tables = self._extract_from_page_pdfplumber(page, page_num + 1)

                    # If pdfplumber found few/no tables, try camelot
                    if len(page_tables) == 0 and CAMELOT_AVAILABLE:
                        camelot_tables = self._extract_with_camelot(str(pdf_path), page_num + 1)
                        page_tables.extend(camelot_tables)

                    tables.extend(page_tables)

            # Classify tables if requested
            if classify:
                for table in tables:
                    table_type, confidence = self.classifier.classify(table)
                    table.table_type = table_type
                    table.confidence = confidence

            logger.info(f"Extracted {len(tables)} tables from PDF")
            return tables

        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")
            return []

    def _extract_from_page_pdfplumber(self, page, page_num: int) -> list[Table]:
        """
        Extract tables from a single page using pdfplumber.

        Args:
            page: pdfplumber page object
            page_num: 1-indexed page number

        Returns:
            List of Table objects
        """
        tables = []

        try:
            extracted_tables = page.extract_tables()

            for raw_table in extracted_tables:
                if not raw_table or len(raw_table) < 2:
                    continue

                # Clean the table data
                cleaned_rows = []
                for row in raw_table:
                    cleaned_row = [(cell.strip() if cell else "") for cell in row]
                    cleaned_rows.append(cleaned_row)

                # First row is typically the header
                headers = cleaned_rows[0] if cleaned_rows else []
                data_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []

                # Skip tables that are too small or empty
                if len(data_rows) < 1 or all(not any(row) for row in data_rows):
                    continue

                table = Table(
                    table_id=self._generate_table_id(page_num),
                    page_num=page_num,
                    headers=headers,
                    rows=data_rows,
                    extraction_method="pdfplumber",
                    metadata={"raw_row_count": len(raw_table)},
                )
                tables.append(table)

        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num}: {e}")

        return tables

    def _extract_with_camelot(self, pdf_path: str, page_num: int) -> list[Table]:
        """
        Extract tables using camelot as fallback.

        Args:
            pdf_path: Path to the PDF file
            page_num: 1-indexed page number

        Returns:
            List of Table objects
        """
        if not CAMELOT_AVAILABLE:
            return []

        tables = []

        try:
            # Try lattice method first (for tables with visible borders)
            camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor="lattice")

            # If no tables found, try stream method
            if len(camelot_tables) == 0:
                camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor="stream")

            for ct in camelot_tables:
                df = ct.df
                if df.empty or len(df) < 2:
                    continue

                headers = df.iloc[0].tolist()
                rows = df.iloc[1:].values.tolist()

                # Clean the data
                headers = [str(h).strip() if h else "" for h in headers]
                cleaned_rows = []
                for row in rows:
                    cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                    cleaned_rows.append(cleaned_row)

                table = Table(
                    table_id=self._generate_table_id(page_num),
                    page_num=page_num,
                    headers=headers,
                    rows=cleaned_rows,
                    extraction_method="camelot",
                    confidence=ct.accuracy / 100 if hasattr(ct, "accuracy") else 0.5,
                    metadata={"camelot_accuracy": getattr(ct, "accuracy", None)},
                )
                tables.append(table)

        except Exception as e:
            logger.debug(f"Camelot extraction failed for page {page_num}: {e}")

        return tables

    def classify_table(self, table: Table) -> tuple[TableType, float]:
        """
        Classify a table's type using heuristics.

        Args:
            table: Table object to classify

        Returns:
            Tuple of (TableType, confidence_score)
        """
        return self.classifier.classify(table)

    def parse_financial_statement(self, table: Table) -> list[FinancialData]:
        """
        Parse a financial statement table into structured data.

        Args:
            table: Table object to parse

        Returns:
            List of FinancialData objects
        """
        return self.financial_parser.parse_financial_statement(table)

    def get_tables_by_type(
        self,
        tables: list[Table],
        table_type: TableType,
        min_confidence: float = 0.3,
    ) -> list[Table]:
        """
        Filter tables by type and minimum confidence.

        Args:
            tables: List of Table objects
            table_type: The TableType to filter for
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of Table objects
        """
        return [t for t in tables if t.table_type == table_type and t.confidence >= min_confidence]

    def extract_financial_tables(
        self,
        pdf_path: str,
        page_range: tuple[int, int] | None = None,
    ) -> dict[str, list[Table]]:
        """
        Extract and categorize financial tables from a PDF.

        Args:
            pdf_path: Path to the PDF file
            page_range: Optional tuple of (start_page, end_page)

        Returns:
            Dict mapping table type names to lists of tables
        """
        all_tables = self.extract_tables(pdf_path, page_range, classify=True)

        financial_types = [
            TableType.INCOME_STATEMENT,
            TableType.BALANCE_SHEET,
            TableType.CASH_FLOW,
            TableType.KEY_METRICS,
        ]

        result: dict[str, list[Table]] = {}

        for ft in financial_types:
            matching = self.get_tables_by_type(all_tables, ft)
            if matching:
                result[ft.value] = matching

        # Add other tables
        other_tables = [t for t in all_tables if t.table_type not in financial_types]
        if other_tables:
            result["other"] = other_tables

        return result
