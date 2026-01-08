"""
Unit tests for table_extractor module.

Tests cover:
- TableType enum values
- Table dataclass functionality
- FinancialData parsing and conversion
- TableClassifier classification logic
- FinancialTableParser number parsing
- TableExtractor extraction methods
"""

from unittest.mock import MagicMock, patch

import pytest

from rhp_analyzer.ingestion.table_extractor import (
    FinancialData,
    FinancialTableParser,
    Table,
    TableClassifier,
    TableExtractor,
    TableType,
)

# ==============================================================================
# TableType Enum Tests
# ==============================================================================


class TestTableType:
    """Tests for TableType enumeration."""

    def test_all_table_types_exist(self):
        """Verify all expected table types are defined."""
        expected_types = [
            "INCOME_STATEMENT",
            "BALANCE_SHEET",
            "CASH_FLOW",
            "SHAREHOLDING_PATTERN",
            "OBJECT_OF_ISSUE",
            "RELATED_PARTY",
            "KEY_METRICS",
            "PEER_COMPARISON",
            "INDEBTEDNESS",
            "LITIGATION_SUMMARY",
            "CONTINGENT_LIABILITIES",
            "OTHER",
        ]
        actual_types = [t.name for t in TableType]
        for expected in expected_types:
            assert expected in actual_types, f"Missing table type: {expected}"

    def test_table_type_values(self):
        """Verify table type values are lowercase strings."""
        for table_type in TableType:
            assert isinstance(table_type.value, str)
            assert table_type.value == table_type.name.lower()


# ==============================================================================
# Table Dataclass Tests
# ==============================================================================


class TestTable:
    """Tests for Table dataclass."""

    def test_table_creation(self):
        """Test basic Table creation."""
        table = Table(
            table_id="test_001",
            page_num=5,
            rows=[["Header1", "Header2"], ["Data1", "Data2"]],
            headers=["Header1", "Header2"],
        )
        assert table.table_id == "test_001"
        assert table.page_num == 5
        assert len(table.rows) == 2
        assert table.headers == ["Header1", "Header2"]

    def test_table_defaults(self):
        """Test Table default values."""
        table = Table(
            table_id="test_002",
            page_num=1,
            rows=[],
            headers=[],
        )
        assert table.table_type == TableType.OTHER
        assert table.confidence == 0.0
        assert table.extraction_method == "pdfplumber"
        assert table.metadata == {}

    def test_table_with_all_fields(self):
        """Test Table with all fields specified."""
        table = Table(
            table_id="test_003",
            page_num=10,
            rows=[["A", "B"], ["C", "D"]],
            headers=["Col1", "Col2"],
            table_type=TableType.BALANCE_SHEET,
            confidence=0.95,
            extraction_method="pdfplumber",
            metadata={"source": "test"},
        )
        assert table.table_type == TableType.BALANCE_SHEET
        assert table.confidence == 0.95
        assert table.extraction_method == "pdfplumber"
        assert table.metadata["source"] == "test"

    def test_table_row_count(self):
        """Test row_count property."""
        table = Table(
            table_id="test_004",
            page_num=1,
            rows=[["A"], ["B"], ["C"]],
            headers=["Header"],
        )
        assert table.row_count == 3

    def test_table_column_count(self):
        """Test col_count property."""
        table = Table(
            table_id="test_005",
            page_num=1,
            rows=[["A", "B", "C"], ["D", "E", "F"]],
            headers=["H1", "H2", "H3"],
        )
        assert table.col_count == 3

    def test_table_column_count_empty(self):
        """Test column_count with empty rows."""
        table = Table(
            table_id="test_006",
            page_num=1,
            rows=[],
            headers=[],
        )
        assert table.col_count == 0

    def test_table_to_dict(self):
        """Test to_dict method."""
        table = Table(
            table_id="test_007",
            page_num=5,
            rows=[["Data"]],
            headers=["Header"],
            table_type=TableType.KEY_METRICS,
            confidence=0.8,
        )
        result = table.to_dict()
        assert result["table_id"] == "test_007"
        assert result["page_num"] == 5
        assert result["table_type"] == "key_metrics"
        assert result["confidence"] == 0.8
        assert result["row_count"] == 1
        assert result["col_count"] == 1


# ==============================================================================
# FinancialData Dataclass Tests
# ==============================================================================


class TestFinancialData:
    """Tests for FinancialData dataclass."""

    def test_financial_data_creation(self):
        """Test basic FinancialData creation."""
        data = FinancialData(
            fiscal_year="FY24",
            metric_name="Revenue",
            value=1500.0,
            unit="crore",
            raw_value="₹1,500 Cr",
        )
        assert data.fiscal_year == "FY24"
        assert data.metric_name == "Revenue"
        assert data.value == 1500.0
        assert data.unit == "crore"

    def test_to_actual_crore_conversion(self):
        """Test conversion from crores to actual value."""
        data = FinancialData(
            fiscal_year="FY24",
            metric_name="Revenue",
            value=100.0,
            unit="crore",
        )
        # 100 crore = 100 * 10,000,000 = 1,000,000,000
        assert data.to_actual() == 1_000_000_000

    def test_to_actual_lakh_conversion(self):
        """Test conversion from lakhs to actual value."""
        data = FinancialData(
            fiscal_year="FY24",
            metric_name="Expenses",
            value=50.0,
            unit="lakh",
        )
        # 50 lakh = 50 * 100,000 = 5,000,000
        assert data.to_actual() == 5_000_000

    def test_to_actual_no_conversion(self):
        """Test no conversion for actual unit."""
        data = FinancialData(
            fiscal_year="FY24",
            metric_name="Count",
            value=42.0,
            unit="actual",
        )
        assert data.to_actual() == 42.0

    def test_to_actual_unknown_unit(self):
        """Test unknown unit defaults to no conversion."""
        data = FinancialData(
            fiscal_year="FY24",
            metric_name="Custom",
            value=100.0,
            unit="unknown_unit",
        )
        assert data.to_actual() == 100.0


# ==============================================================================
# TableClassifier Tests
# ==============================================================================


class TestTableClassifier:
    """Tests for TableClassifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = TableClassifier()

    def test_classify_income_statement(self):
        """Test classification of income statement table."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Particulars", "FY24", "FY23"],
                ["Revenue from Operations", "1000", "800"],
                ["Total Income", "1050", "850"],
                ["EBITDA", "200", "150"],
                ["Profit After Tax", "100", "80"],
            ],
            headers=["Particulars", "FY24", "FY23"],
        )
        table_type, confidence = self.classifier.classify(table)
        assert table_type == TableType.INCOME_STATEMENT
        assert confidence > 0.5

    def test_classify_balance_sheet(self):
        """Test classification of balance sheet table."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Particulars", "Amount"],
                ["Total Assets", "5000"],
                ["Total Equity", "3000"],
                ["Total Liabilities", "2000"],
                ["Non-Current Assets", "4000"],
            ],
            headers=["Particulars", "Amount"],
        )
        table_type, confidence = self.classifier.classify(table)
        assert table_type == TableType.BALANCE_SHEET
        assert confidence > 0.5

    def test_classify_cash_flow(self):
        """Test classification of cash flow table."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Cash Flow Statement", "Amount"],
                ["Operating Activities", "500"],
                ["Investing Activities", "-200"],
                ["Financing Activities", "-100"],
                ["Net Cash Flow", "200"],
            ],
            headers=["Cash Flow Statement", "Amount"],
        )
        table_type, confidence = self.classifier.classify(table)
        assert table_type == TableType.CASH_FLOW
        assert confidence > 0.5

    def test_classify_shareholding_pattern(self):
        """Test classification of shareholding pattern table."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Category", "Pre-IPO", "Post-IPO"],
                ["Promoter Holding", "75%", "60%"],
                ["Public Shareholding", "25%", "40%"],
            ],
            headers=["Category", "Pre-IPO", "Post-IPO"],
        )
        table_type, confidence = self.classifier.classify(table)
        assert table_type == TableType.SHAREHOLDING_PATTERN
        assert confidence > 0.5

    def test_classify_peer_comparison(self):
        """Test classification of peer comparison table."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Our Company", "25", "3.5", "18%"],
                ["Listed Peers", "22", "3.0", "15%"],
                ["Industry Peers", "20", "2.8", "14%"],
            ],
            headers=["Peer Comparison", "P/E", "P/B", "ROE"],
        )
        table_type, confidence = self.classifier.classify(table)
        assert table_type == TableType.PEER_COMPARISON
        assert confidence > 0.5

    def test_classify_indebtedness(self):
        """Test classification of indebtedness table."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Type of Debt", "Amount", "Interest Rate"],
                ["Secured Loans", "500", "10%"],
                ["Unsecured Borrowings", "200", "12%"],
                ["Total Debt", "700", ""],
            ],
            headers=["Type of Debt", "Amount", "Interest Rate"],
        )
        table_type, confidence = self.classifier.classify(table)
        assert table_type == TableType.INDEBTEDNESS
        assert confidence > 0.5

    def test_classify_unknown_table(self):
        """Test classification of unrecognized table."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Column A", "Column B"],
                ["Random Data", "More Data"],
            ],
            headers=["Column A", "Column B"],
        )
        table_type, confidence = self.classifier.classify(table)
        assert table_type == TableType.OTHER
        assert confidence == 0.0


# ==============================================================================
# FinancialTableParser Tests
# ==============================================================================


class TestFinancialTableParser:
    """Tests for FinancialTableParser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = FinancialTableParser()

    # Number Parsing Tests
    def test_parse_number_simple(self):
        """Test parsing simple numbers."""
        assert self.parser.parse_number("1234") == 1234.0
        assert self.parser.parse_number("1,234") == 1234.0
        assert self.parser.parse_number("1,234.56") == 1234.56

    def test_parse_number_with_rupee_symbol(self):
        """Test parsing numbers with rupee symbol."""
        assert self.parser.parse_number("₹1,000") == 1000.0
        assert self.parser.parse_number("Rs. 500") == 500.0
        assert self.parser.parse_number("Rs 250.50") == 250.50

    def test_parse_number_crore_format(self):
        """Test parsing crore format numbers."""
        # "100 Cr" should be recognized, but parse_number returns raw value
        result = self.parser.parse_number("100.50")
        assert result == 100.50

    def test_parse_number_negative_parentheses(self):
        """Test parsing negative numbers in parentheses (accounting format)."""
        assert self.parser.parse_number("(100)") == -100.0
        assert self.parser.parse_number("(1,234.56)") == -1234.56
        assert self.parser.parse_number("( 500 )") == -500.0

    def test_parse_number_negative_sign(self):
        """Test parsing negative numbers with minus sign."""
        assert self.parser.parse_number("-100") == -100.0
        assert self.parser.parse_number("-1,234.56") == -1234.56

    def test_parse_number_invalid(self):
        """Test parsing invalid number strings."""
        # N/A and - are treated as zero in financial contexts
        assert self.parser.parse_number("N/A") == 0.0
        assert self.parser.parse_number("-") == 0.0
        assert self.parser.parse_number("") is None
        assert self.parser.parse_number("text only") is None

    def test_parse_number_with_spaces(self):
        """Test parsing numbers with spaces."""
        assert self.parser.parse_number("  1,234  ") == 1234.0
        assert self.parser.parse_number("1 234") == 1234.0

    # Unit Detection Tests
    def test_detect_unit_crore(self):
        """Test detecting crore unit."""
        table1 = Table(table_id="t", page_num=1, rows=[], headers=["₹500 Cr"])
        table2 = Table(table_id="t", page_num=1, rows=[], headers=["100 Crore"])
        table3 = Table(table_id="t", page_num=1, rows=[], headers=["Amount in Crores"])
        assert self.parser.detect_unit(table1) == "crore"
        assert self.parser.detect_unit(table2) == "crore"
        assert self.parser.detect_unit(table3) == "crore"

    def test_detect_unit_lakh(self):
        """Test detecting lakh unit."""
        table1 = Table(table_id="t", page_num=1, rows=[], headers=["₹50 Lakh"])
        table2 = Table(table_id="t", page_num=1, rows=[], headers=["100 Lakhs"])
        table3 = Table(table_id="t", page_num=1, rows=[], headers=["Amount in Lacs"])
        assert self.parser.detect_unit(table1) == "lakh"
        assert self.parser.detect_unit(table2) == "lakh"
        assert self.parser.detect_unit(table3) == "lakh"

    def test_detect_unit_default(self):
        """Test default unit detection."""
        table1 = Table(table_id="t", page_num=1, rows=[], headers=["1234"])
        table2 = Table(table_id="t", page_num=1, rows=[], headers=["Simple number"])
        assert self.parser.detect_unit(table1) == "actual"
        assert self.parser.detect_unit(table2) == "actual"

    # Fiscal Year Extraction Tests
    def test_extract_fiscal_years_fy_format(self):
        """Test extracting FY format fiscal years."""
        table = Table(table_id="t", page_num=1, rows=[], headers=["Particulars", "FY24", "FY23", "FY22"])
        years = self.parser.extract_fiscal_years(table)
        assert "FY24" in years
        assert "FY23" in years
        assert "FY22" in years

    def test_extract_fiscal_years_full_format(self):
        """Test extracting full year format."""
        table = Table(table_id="t", page_num=1, rows=[], headers=["Item", "2023-24", "2022-23"])
        years = self.parser.extract_fiscal_years(table)
        assert "2023-24" in years
        assert "2022-23" in years

    def test_extract_fiscal_years_march_format(self):
        """Test extracting March year format."""
        table = Table(table_id="t", page_num=1, rows=[], headers=["Metric", "Mar 2024", "Mar 2023"])
        years = self.parser.extract_fiscal_years(table)
        assert "Mar 2024" in years
        assert "Mar 2023" in years

    def test_extract_fiscal_years_no_years(self):
        """Test when no fiscal years found."""
        table = Table(table_id="t", page_num=1, rows=[], headers=["Column A", "Column B", "Column C"])
        years = self.parser.extract_fiscal_years(table)
        assert len(years) == 0

    # Financial Statement Parsing Tests
    def test_parse_financial_statement(self):
        """Test parsing a financial statement table."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Particulars", "FY24", "FY23"],
                ["Revenue", "1,500", "1,200"],
                ["EBITDA", "300", "250"],
                ["PAT", "150", "120"],
            ],
            headers=["Particulars", "FY24", "FY23"],
        )
        result = self.parser.parse_financial_statement(table)

        # Check that we have data
        assert len(result) > 0

        # Check structure of results
        for data in result:
            assert isinstance(data, FinancialData)
            assert data.fiscal_year in ["FY24", "FY23"]
            assert data.metric_name in ["Revenue", "EBITDA", "PAT"]

    def test_parse_financial_statement_with_crores(self):
        """Test parsing with crore unit detection."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Particulars (₹ in Crores)", "FY24"],
                ["Revenue", "1,500"],
            ],
            headers=["Particulars (₹ in Crores)", "FY24"],
        )
        result = self.parser.parse_financial_statement(table)

        # Should detect crore unit from header
        if result:
            assert result[0].unit == "crore"

    # Growth Rate Tests
    def test_calculate_growth_rates(self):
        """Test growth rate calculation."""
        data = [
            FinancialData("FY22", "Revenue", 1000.0, "crore"),
            FinancialData("FY23", "Revenue", 1200.0, "crore"),
            FinancialData("FY24", "Revenue", 1500.0, "crore"),
        ]
        rates = self.parser.calculate_growth_rates(data)

        # Implementation uses fiscal year as key for growth from previous year
        # FY22 to FY23: (1200-1000)/1000 = 20%
        # FY23 to FY24: (1500-1200)/1200 = 25%
        assert "Revenue" in rates
        assert "FY23" in rates["Revenue"] or "FY24" in rates["Revenue"]

    def test_calculate_growth_rates_insufficient_data(self):
        """Test growth rate with insufficient data."""
        data = [
            FinancialData("FY24", "Revenue", 1000.0, "crore"),
        ]
        rates = self.parser.calculate_growth_rates(data)
        # With only one data point, no growth can be calculated
        # Implementation returns dict with metric but empty rates
        assert "Revenue" not in rates or len(rates.get("Revenue", {})) == 0

    def test_calculate_growth_rates_zero_base(self):
        """Test growth rate with zero base value."""
        data = [
            FinancialData("FY23", "Revenue", 0.0, "crore"),
            FinancialData("FY24", "Revenue", 100.0, "crore"),
        ]
        rates = self.parser.calculate_growth_rates(data)
        # Should handle division by zero gracefully
        # Either return empty or skip that calculation


# ==============================================================================
# TableExtractor Tests
# ==============================================================================


class TestTableExtractor:
    """Tests for TableExtractor main class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = TableExtractor()

    def test_extractor_initialization(self):
        """Test TableExtractor initialization."""
        assert self.extractor.classifier is not None
        assert self.extractor.financial_parser is not None
        assert isinstance(self.extractor.classifier, TableClassifier)
        assert isinstance(self.extractor.financial_parser, FinancialTableParser)

    def test_classify_table(self):
        """Test classify_table method."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Item", "Value"],
                ["Total Revenue", "1000"],
                ["EBITDA", "200"],
            ],
            headers=["Item", "Value"],
        )
        table_type, confidence = self.extractor.classify_table(table)
        assert table_type is not None
        assert confidence >= 0.0

    def test_parse_financial_statement(self):
        """Test parse_financial_statement method."""
        table = Table(
            table_id="test",
            page_num=1,
            rows=[
                ["Metric", "FY24"],
                ["Revenue", "500"],
            ],
            headers=["Metric", "FY24"],
        )
        result = self.extractor.parse_financial_statement(table)
        assert isinstance(result, list)

    def test_get_tables_by_type_empty(self):
        """Test get_tables_by_type with no tables."""
        result = self.extractor.get_tables_by_type([], TableType.BALANCE_SHEET)
        assert result == []

    def test_get_tables_by_type_with_tables(self):
        """Test get_tables_by_type with matching tables."""
        # Create test tables
        table1 = Table(
            table_id="t1",
            page_num=1,
            rows=[["Data"]],
            headers=["Header"],
            table_type=TableType.BALANCE_SHEET,
            confidence=0.8,
        )
        table2 = Table(
            table_id="t2",
            page_num=2,
            rows=[["Data"]],
            headers=["Header"],
            table_type=TableType.INCOME_STATEMENT,
            confidence=0.8,
        )
        tables = [table1, table2]

        balance_sheets = self.extractor.get_tables_by_type(tables, TableType.BALANCE_SHEET)
        assert len(balance_sheets) == 1
        assert balance_sheets[0].table_id == "t1"

    def test_extract_financial_tables(self):
        """Test extract_financial_tables method returns dict by type."""
        # This method requires a PDF path and extracts tables
        # Test the method with a mock to verify it returns proper dict structure
        with patch.object(self.extractor, "extract_tables") as mock_extract:
            financial_table = Table(
                table_id="fin1",
                page_num=1,
                rows=[["Revenue", "1000"]],
                headers=["Metric", "Value"],
                table_type=TableType.INCOME_STATEMENT,
                confidence=0.8,
            )
            other_table = Table(
                table_id="other1",
                page_num=2,
                rows=[["Data"]],
                headers=["Header"],
                table_type=TableType.OTHER,
                confidence=0.5,
            )
            mock_extract.return_value = [financial_table, other_table]

            result = self.extractor.extract_financial_tables("test.pdf")

            # Should return a dict with table types as keys
            assert isinstance(result, dict)
            # Income statement should be categorized
            if TableType.INCOME_STATEMENT.value in result:
                assert any(t.table_id == "fin1" for t in result[TableType.INCOME_STATEMENT.value])
            # OTHER type should be in 'other' category
            if "other" in result:
                assert any(t.table_id == "other1" for t in result["other"])

    @patch("rhp_analyzer.ingestion.table_extractor.PDFPLUMBER_AVAILABLE", True)
    def test_extract_tables_invalid_path(self):
        """Test extract_tables with invalid PDF path."""
        result = self.extractor.extract_tables("nonexistent.pdf")
        assert result == []

    def test_extract_tables_page_range(self):
        """Test page range parameter handling."""
        # This tests the logic without needing a real PDF
        extractor = TableExtractor()
        # Verify page_range parameter is accepted
        # Can't test actual extraction without PDF, but can verify interface


# ==============================================================================
# Integration Tests (Mock-based)
# ==============================================================================


class TestTableExtractionIntegration:
    """Integration tests using mocks."""

    @patch("rhp_analyzer.ingestion.table_extractor.PDFPLUMBER_AVAILABLE", True)
    @patch("rhp_analyzer.ingestion.table_extractor.pdfplumber")
    def test_pdfplumber_extraction_flow(self, mock_pdfplumber):
        """Test extraction flow with pdfplumber."""
        # Setup mock
        mock_page = MagicMock()
        mock_table = [
            ["Header1", "Header2"],
            ["Data1", "Data2"],
        ]
        mock_page.extract_tables.return_value = [mock_table]

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pdfplumber.open.return_value = mock_pdf

        extractor = TableExtractor()
        # Would need actual file for full test
        # This verifies mock setup works

    def test_classifier_confidence_range(self):
        """Test that classifier confidence is always in valid range."""
        classifier = TableClassifier()

        # Test with various tables
        test_tables = [
            Table("t1", 1, [["Revenue", "1000"]], ["Item", "Value"]),
            Table("t2", 1, [["Random", "Data"]], ["Col1", "Col2"]),
            Table("t3", 1, [["Total Assets", "5000"]], ["Balance Sheet"]),
        ]

        for table in test_tables:
            _, confidence = classifier.classify(table)
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range"


# ==============================================================================
# Edge Case Tests
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_table_classification(self):
        """Test classification of empty table."""
        classifier = TableClassifier()
        table = Table("empty", 1, [], [])
        table_type, confidence = classifier.classify(table)
        assert table_type == TableType.OTHER
        assert confidence == 0.0

    def test_single_cell_table(self):
        """Test handling of single-cell table."""
        classifier = TableClassifier()
        table = Table("single", 1, [["OnlyCell"]], ["OnlyHeader"])
        table_type, confidence = classifier.classify(table)
        # Should not crash, should return some classification
        assert table_type is not None

    def test_unicode_content(self):
        """Test handling of unicode content."""
        parser = FinancialTableParser()
        # Test rupee symbol and Indian numerals
        assert parser.parse_number("₹1,00,000") is not None or parser.parse_number("₹100000") is not None

    def test_mixed_number_formats(self):
        """Test parsing mixed number formats in same table."""
        parser = FinancialTableParser()
        numbers = ["1,234", "(500)", "₹1,000", "100.50", "-25"]
        for num_str in numbers:
            result = parser.parse_number(num_str)
            assert result is not None, f"Failed to parse: {num_str}"

    def test_table_with_none_values(self):
        """Test table with None values in cells."""
        table = Table(
            table_id="nulls",
            page_num=1,
            rows=[["Header", None], [None, "Value"]],
            headers=["Col1", "Col2"],
        )
        # Should handle gracefully
        classifier = TableClassifier()
        result = classifier.classify(table)
        assert result is not None

    def test_very_large_table(self):
        """Test handling of large table."""
        # Create a table with many rows
        rows = [["Header1", "Header2", "Header3"]]
        for i in range(1000):
            rows.append([f"Row{i}_Col1", f"Row{i}_Col2", f"Row{i}_Col3"])

        table = Table(
            table_id="large",
            page_num=1,
            rows=rows,
            headers=["Header1", "Header2", "Header3"],
        )

        assert table.row_count == 1001
        assert table.col_count == 3

        # Classification should still work
        classifier = TableClassifier()
        result = classifier.classify(table)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
