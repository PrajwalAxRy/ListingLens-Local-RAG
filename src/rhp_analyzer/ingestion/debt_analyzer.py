"""
Debt Structure Analyzer Module

Subtask 2.5A.4: Analyzes comprehensive debt structure from RHP indebtedness section.

Extracts and analyzes:
- Secured vs unsecured debt breakdown
- Short-term vs long-term classification
- Maturity waterfall (0-1yr, 1-3yr, 3-5yr, 5yr+)
- Interest rate analysis (weighted average, range)
- Financial covenants from material contracts
- Post-IPO debt calculation after repayment
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DebtItem:
    """Represents a single debt instrument."""

    lender: str = ""
    facility_type: str = ""  # Term loan, Working capital, NCD, etc.
    amount: float = 0.0  # In crores
    interest_rate: Optional[float] = None  # Percentage
    is_secured: bool = True
    is_short_term: bool = False  # True if maturity < 1 year
    maturity_date: Optional[str] = None
    security_details: str = ""
    covenants: list[str] = field(default_factory=list)
    outstanding_as_of: Optional[str] = None
    page_reference: Optional[int] = None


@dataclass
class MaturityProfile:
    """Debt maturity breakdown by time bucket."""

    within_1_year: float = 0.0
    between_1_and_3_years: float = 0.0
    between_3_and_5_years: float = 0.0
    beyond_5_years: float = 0.0

    @property
    def total(self) -> float:
        """Total debt from maturity profile."""
        return self.within_1_year + self.between_1_and_3_years + self.between_3_and_5_years + self.beyond_5_years

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "0-1yr": self.within_1_year,
            "1-3yr": self.between_1_and_3_years,
            "3-5yr": self.between_3_and_5_years,
            "5yr+": self.beyond_5_years,
            "total": self.total,
        }


@dataclass
class Covenant:
    """Financial covenant from loan agreements."""

    covenant_type: str = ""  # DSCR, D/E, Current Ratio, etc.
    description: str = ""
    threshold: Optional[str] = None  # e.g., ">=1.5x", "<=2.0x"
    current_value: Optional[float] = None
    is_breached: bool = False
    lender: str = ""
    source: str = ""  # Section/page reference


@dataclass
class DebtStructure:
    """Comprehensive debt structure analysis result."""

    # Total debt breakdown
    total_debt: float = 0.0
    secured_debt: float = 0.0
    unsecured_debt: float = 0.0
    short_term_debt: float = 0.0
    long_term_debt: float = 0.0

    # Cost of debt
    weighted_avg_interest_rate: Optional[float] = None
    highest_interest_rate: Optional[float] = None
    lowest_interest_rate: Optional[float] = None

    # Maturity profile
    maturity_profile: MaturityProfile = field(default_factory=MaturityProfile)

    # Lender concentration
    top_lender: Optional[str] = None
    top_lender_exposure: float = 0.0
    number_of_lenders: int = 0

    # Covenants
    has_financial_covenants: bool = False
    covenants: list[Covenant] = field(default_factory=list)
    covenant_breaches_disclosed: bool = False

    # IPO proceeds for debt
    debt_repayment_from_ipo: float = 0.0
    debt_repayment_percent_of_fresh_issue: float = 0.0
    post_ipo_debt: float = 0.0

    # Key ratios
    debt_to_equity_pre_ipo: float = 0.0
    debt_to_equity_post_ipo: float = 0.0
    interest_coverage_ratio: Optional[float] = None
    debt_to_ebitda: Optional[float] = None

    # Individual debt items
    debt_items: list[DebtItem] = field(default_factory=list)

    # Metadata
    as_of_date: Optional[str] = None
    citations: list[str] = field(default_factory=list)

    def get_concentration_risk(self) -> str:
        """Assess lender concentration risk."""
        if self.total_debt == 0:
            return "N/A"

        concentration = (self.top_lender_exposure / self.total_debt) * 100

        if concentration > 50:
            return "HIGH"
        elif concentration > 30:
            return "MEDIUM"
        else:
            return "LOW"

    def get_maturity_risk(self) -> str:
        """Assess near-term maturity risk."""
        if self.total_debt == 0:
            return "N/A"

        near_term_pct = (self.maturity_profile.within_1_year / self.total_debt) * 100

        if near_term_pct > 50:
            return "HIGH"
        elif near_term_pct > 30:
            return "MEDIUM"
        else:
            return "LOW"


class DebtStructureAnalyzer:
    """
    Analyzes comprehensive debt structure from RHP indebtedness section.

    Implements Subtask 2.5A.4 from the milestone plan:
    - Parse "Indebtedness" section tables
    - Extract secured vs unsecured debt
    - Parse short-term vs long-term split
    - Extract interest rates (weighted average, range)
    - Build debt maturity waterfall
    - Extract financial covenants from material contracts
    - Calculate post-IPO debt after repayment
    """

    # Common lender patterns for extraction
    LENDER_PATTERNS = [
        r"State Bank of India|SBI",
        r"HDFC Bank|HDFC Ltd",
        r"ICICI Bank|ICICI",
        r"Axis Bank",
        r"Punjab National Bank|PNB",
        r"Bank of Baroda|BOB",
        r"Canara Bank",
        r"Union Bank",
        r"Kotak Mahindra Bank|Kotak",
        r"Yes Bank",
        r"IndusInd Bank",
        r"Federal Bank",
        r"RBL Bank",
        r"IDFC First Bank|IDFC",
        r"Bandhan Bank",
        r"L&T Finance",
        r"Tata Capital",
        r"Bajaj Finance",
        r"Sundaram Finance",
        r"Cholamandalam",
    ]

    # Facility type patterns
    FACILITY_PATTERNS = {
        "term_loan": r"term\s*loan|TL|long\s*term\s*loan",
        "working_capital": r"working\s*capital|WC|cash\s*credit|CC|overdraft|OD",
        "ncd": r"non[- ]?convertible\s*debenture|NCD",
        "ecb": r"external\s*commercial\s*borrowing|ECB",
        "vehicle_loan": r"vehicle\s*loan|equipment\s*loan",
        "unsecured_loan": r"unsecured\s*loan|inter[- ]?corporate",
    }

    # Covenant type patterns
    COVENANT_PATTERNS = {
        "dscr": r"debt\s*service\s*coverage|DSCR",
        "debt_equity": r"debt[/-]?equity|D/E|DE\s*ratio",
        "current_ratio": r"current\s*ratio|CR",
        "interest_coverage": r"interest\s*coverage|ICR",
        "total_debt_ebitda": r"(total\s*)?debt[/-]?EBITDA",
        "fixed_asset_coverage": r"fixed\s*asset\s*coverage|FAC",
        "tangible_net_worth": r"tangible\s*net\s*worth|TNW|minimum\s*net\s*worth",
    }

    def __init__(self):
        """Initialize the debt structure analyzer."""
        self.debt_items: list[DebtItem] = []
        self.covenants: list[Covenant] = []

    def analyze_debt(
        self,
        indebtedness_data: dict[str, Any],
        financial_data: Optional[dict[str, Any]] = None,
        ipo_details: Optional[dict[str, Any]] = None,
    ) -> DebtStructure:
        """
        Analyze comprehensive debt structure from RHP data.

        Args:
            indebtedness_data: Extracted data from indebtedness section
            financial_data: Latest financial data (equity, EBITDA)
            ipo_details: IPO details including objects of issue

        Returns:
            DebtStructure: Comprehensive debt structure analysis
        """
        result = DebtStructure()

        # Parse debt items
        debt_items = indebtedness_data.get("debt_items", [])
        if debt_items:
            result.debt_items = self._parse_debt_items(debt_items)

        # Calculate totals from debt items
        result = self._calculate_debt_totals(result)

        # Calculate interest rates
        result = self._calculate_interest_rates(result)

        # Build maturity profile
        maturity_data = indebtedness_data.get("maturity_profile", {})
        result.maturity_profile = self._parse_maturity_profile(maturity_data)

        # Analyze lender concentration
        result = self._analyze_lender_concentration(result)

        # Extract covenants
        covenant_data = indebtedness_data.get("covenants", [])
        if covenant_data:
            result.covenants = self._parse_covenants(covenant_data)
            result.has_financial_covenants = len(result.covenants) > 0
            result.covenant_breaches_disclosed = any(c.is_breached for c in result.covenants)

        # Calculate IPO impact
        if ipo_details:
            result = self._calculate_ipo_impact(result, ipo_details, financial_data)

        # Calculate key ratios
        if financial_data:
            result = self._calculate_ratios(result, financial_data)

        # Set metadata
        result.as_of_date = indebtedness_data.get("as_of_date")
        result.citations = indebtedness_data.get("citations", [])

        return result

    def _parse_debt_items(self, items: list[dict[str, Any]]) -> list[DebtItem]:
        """Parse individual debt instruments from extracted data."""
        debt_items = []

        for item in items:
            debt_item = DebtItem(
                lender=item.get("lender", ""),
                facility_type=self._classify_facility_type(item.get("facility_type", "")),
                amount=self._parse_amount(item.get("amount", 0)),
                interest_rate=self._parse_rate(item.get("interest_rate")),
                is_secured=item.get("is_secured", True),
                is_short_term=item.get("is_short_term", False),
                maturity_date=item.get("maturity_date"),
                security_details=item.get("security_details", ""),
                covenants=item.get("covenants", []),
                outstanding_as_of=item.get("as_of_date"),
                page_reference=item.get("page_reference"),
            )
            debt_items.append(debt_item)

        return debt_items

    def _classify_facility_type(self, facility_text: str) -> str:
        """Classify facility type based on description."""
        if not facility_text:
            return "Other"

        text_lower = facility_text.lower()

        for facility_type, pattern in self.FACILITY_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return facility_type.replace("_", " ").title()

        return facility_text

    def _parse_amount(self, amount: Any) -> float:
        """Parse amount value, handling various formats."""
        if amount is None:
            return 0.0

        if isinstance(amount, (int, float)):
            return float(amount)

        if isinstance(amount, str):
            # Remove commas and currency symbols
            cleaned = re.sub(r"[₹,\s]", "", amount)

            # Handle crore/lakh notation
            if "crore" in amount.lower() or "cr" in amount.lower():
                match = re.search(r"([\d.]+)", cleaned)
                if match:
                    return float(match.group(1))
            elif "lakh" in amount.lower():
                match = re.search(r"([\d.]+)", cleaned)
                if match:
                    return float(match.group(1)) / 100  # Convert to crores

            try:
                return float(cleaned)
            except ValueError:
                return 0.0

        return 0.0

    def _parse_rate(self, rate: Any) -> Optional[float]:
        """Parse interest rate value."""
        if rate is None:
            return None

        if isinstance(rate, (int, float)):
            return float(rate)

        if isinstance(rate, str):
            # Extract numeric value from rate string
            match = re.search(r"([\d.]+)", rate)
            if match:
                return float(match.group(1))

        return None

    def _calculate_debt_totals(self, result: DebtStructure) -> DebtStructure:
        """Calculate debt totals from individual items."""
        for item in result.debt_items:
            result.total_debt += item.amount

            if item.is_secured:
                result.secured_debt += item.amount
            else:
                result.unsecured_debt += item.amount

            if item.is_short_term:
                result.short_term_debt += item.amount
            else:
                result.long_term_debt += item.amount

        return result

    def _calculate_interest_rates(self, result: DebtStructure) -> DebtStructure:
        """Calculate weighted average and range of interest rates."""
        rates = []
        weighted_sum = 0.0
        total_amount = 0.0

        for item in result.debt_items:
            if item.interest_rate is not None and item.amount > 0:
                rates.append(item.interest_rate)
                weighted_sum += item.interest_rate * item.amount
                total_amount += item.amount

        if rates:
            result.highest_interest_rate = max(rates)
            result.lowest_interest_rate = min(rates)

            if total_amount > 0:
                result.weighted_avg_interest_rate = weighted_sum / total_amount

        return result

    def _parse_maturity_profile(self, maturity_data: dict[str, Any]) -> MaturityProfile:
        """Parse debt maturity profile."""
        profile = MaturityProfile()

        if not maturity_data:
            # Try to build from debt items
            return self._build_maturity_from_items()

        profile.within_1_year = self._parse_amount(maturity_data.get("0-1yr", 0))
        profile.between_1_and_3_years = self._parse_amount(maturity_data.get("1-3yr", 0))
        profile.between_3_and_5_years = self._parse_amount(maturity_data.get("3-5yr", 0))
        profile.beyond_5_years = self._parse_amount(maturity_data.get("5yr+", 0))

        return profile

    def _build_maturity_from_items(self) -> MaturityProfile:
        """Build maturity profile from individual debt items based on maturity dates."""
        profile = MaturityProfile()

        for item in self.debt_items:
            if item.is_short_term:
                profile.within_1_year += item.amount
            else:
                # Default long-term to 1-3 year bucket if no specific date
                profile.between_1_and_3_years += item.amount

        return profile

    def _analyze_lender_concentration(self, result: DebtStructure) -> DebtStructure:
        """Analyze lender concentration risk."""
        lender_totals: dict[str, float] = {}

        for item in result.debt_items:
            lender = item.lender or "Unknown"
            lender_totals[lender] = lender_totals.get(lender, 0) + item.amount

        result.number_of_lenders = len(lender_totals)

        if lender_totals:
            top_lender = max(lender_totals, key=lender_totals.get)
            result.top_lender = top_lender
            result.top_lender_exposure = lender_totals[top_lender]

        return result

    def _parse_covenants(self, covenant_data: list[dict[str, Any]]) -> list[Covenant]:
        """Parse financial covenants from extracted data."""
        covenants = []

        for item in covenant_data:
            covenant = Covenant(
                covenant_type=self._classify_covenant_type(item.get("type", "")),
                description=item.get("description", ""),
                threshold=item.get("threshold"),
                current_value=item.get("current_value"),
                is_breached=item.get("is_breached", False),
                lender=item.get("lender", ""),
                source=item.get("source", ""),
            )
            covenants.append(covenant)

        return covenants

    def _classify_covenant_type(self, covenant_text: str) -> str:
        """Classify covenant type based on description."""
        if not covenant_text:
            return "Other"

        text_lower = covenant_text.lower()

        for covenant_type, pattern in self.COVENANT_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return covenant_type.upper()

        return covenant_text

    def _calculate_ipo_impact(
        self, result: DebtStructure, ipo_details: dict[str, Any], financial_data: Optional[dict[str, Any]]
    ) -> DebtStructure:
        """Calculate IPO impact on debt structure."""
        # Get debt repayment from objects of issue
        debt_repayment = self._parse_amount(ipo_details.get("debt_repayment", 0))
        fresh_issue = self._parse_amount(ipo_details.get("fresh_issue", 0))

        result.debt_repayment_from_ipo = debt_repayment

        if fresh_issue > 0:
            result.debt_repayment_percent_of_fresh_issue = (debt_repayment / fresh_issue) * 100

        result.post_ipo_debt = max(0, result.total_debt - debt_repayment)

        # Calculate post-IPO D/E if financial data available
        if financial_data:
            equity = self._parse_amount(financial_data.get("total_equity", 0))
            fresh_issue_equity = fresh_issue  # Fresh issue adds to equity

            if equity > 0:
                result.debt_to_equity_pre_ipo = result.total_debt / equity

                post_ipo_equity = equity + fresh_issue_equity
                if post_ipo_equity > 0:
                    result.debt_to_equity_post_ipo = result.post_ipo_debt / post_ipo_equity

        return result

    def _calculate_ratios(self, result: DebtStructure, financial_data: dict[str, Any]) -> DebtStructure:
        """Calculate debt-related financial ratios."""
        equity = self._parse_amount(financial_data.get("total_equity", 0))
        ebitda = self._parse_amount(financial_data.get("ebitda", 0))
        interest_expense = self._parse_amount(financial_data.get("interest_expense", 0))

        if equity > 0 and result.debt_to_equity_pre_ipo == 0:
            result.debt_to_equity_pre_ipo = result.total_debt / equity

        if ebitda > 0:
            result.debt_to_ebitda = result.total_debt / ebitda

            if interest_expense > 0:
                result.interest_coverage_ratio = ebitda / interest_expense

        return result

    def parse_indebtedness_table(self, table_data: list[list[str]]) -> list[DebtItem]:
        """
        Parse indebtedness table from PDF extraction.

        Args:
            table_data: Raw table data as list of rows

        Returns:
            List of DebtItem objects
        """
        if not table_data or len(table_data) < 2:
            return []

        debt_items = []
        headers = [h.lower().strip() for h in table_data[0]]

        # Map headers to expected fields
        header_map = self._map_headers(headers)

        for row in table_data[1:]:
            if len(row) < 2:
                continue

            item = self._parse_table_row(row, header_map)
            if item and item.amount > 0:
                debt_items.append(item)

        return debt_items

    def _map_headers(self, headers: list[str]) -> dict[str, int]:
        """Map table headers to field indices."""
        header_map = {}

        for i, header in enumerate(headers):
            header_lower = header.lower()

            if any(w in header_lower for w in ["lender", "bank", "institution", "party"]):
                header_map["lender"] = i
            elif any(w in header_lower for w in ["amount", "outstanding", "balance"]):
                header_map["amount"] = i
            elif any(w in header_lower for w in ["rate", "interest", "roi"]):
                header_map["rate"] = i
            elif any(w in header_lower for w in ["facility", "type", "nature"]):
                header_map["facility"] = i
            elif any(w in header_lower for w in ["secured", "security"]):
                header_map["secured"] = i
            elif any(w in header_lower for w in ["maturity", "repayment", "due"]):
                header_map["maturity"] = i

        return header_map

    def _parse_table_row(self, row: list[str], header_map: dict[str, int]) -> Optional[DebtItem]:
        """Parse a single row from indebtedness table."""
        try:
            item = DebtItem()

            if "lender" in header_map and header_map["lender"] < len(row):
                item.lender = row[header_map["lender"]].strip()

            if "amount" in header_map and header_map["amount"] < len(row):
                item.amount = self._parse_amount(row[header_map["amount"]])

            if "rate" in header_map and header_map["rate"] < len(row):
                item.interest_rate = self._parse_rate(row[header_map["rate"]])

            if "facility" in header_map and header_map["facility"] < len(row):
                item.facility_type = self._classify_facility_type(row[header_map["facility"]])

            if "secured" in header_map and header_map["secured"] < len(row):
                secured_text = row[header_map["secured"]].lower()
                item.is_secured = "unsecured" not in secured_text

            if "maturity" in header_map and header_map["maturity"] < len(row):
                item.maturity_date = row[header_map["maturity"]].strip()

            return item

        except Exception:
            return None

    def extract_covenants_from_text(self, text: str) -> list[Covenant]:
        """
        Extract financial covenants from material contract text.

        Args:
            text: Text from material contracts section

        Returns:
            List of Covenant objects
        """
        covenants = []

        for covenant_type, pattern in self.COVENANT_PATTERNS.items():
            matches = re.finditer(
                rf"({pattern})[^.]*?(>=?|<=?|at\s*least|not\s*exceed)[^.]*?([\d.]+)", text, re.IGNORECASE
            )

            for match in matches:
                covenant = Covenant(
                    covenant_type=covenant_type.upper(),
                    description=match.group(0),
                    threshold=f"{match.group(2)} {match.group(3)}",
                )
                covenants.append(covenant)

        return covenants

    def get_debt_summary(self, result: DebtStructure) -> dict[str, Any]:
        """
        Generate a summary of debt analysis for reporting.

        Args:
            result: DebtStructure analysis result

        Returns:
            Dictionary with summary metrics
        """
        summary = {
            "total_debt_cr": result.total_debt,
            "secured_percent": (result.secured_debt / result.total_debt * 100) if result.total_debt > 0 else 0,
            "short_term_percent": (result.short_term_debt / result.total_debt * 100) if result.total_debt > 0 else 0,
            "weighted_avg_rate": result.weighted_avg_interest_rate,
            "rate_range": f"{result.lowest_interest_rate or 0:.1f}% - {result.highest_interest_rate or 0:.1f}%",
            "maturity_risk": result.get_maturity_risk(),
            "concentration_risk": result.get_concentration_risk(),
            "top_lender": result.top_lender,
            "number_of_lenders": result.number_of_lenders,
            "has_covenants": result.has_financial_covenants,
            "covenant_count": len(result.covenants),
            "any_breach": result.covenant_breaches_disclosed,
            "debt_repayment_from_ipo_cr": result.debt_repayment_from_ipo,
            "debt_repayment_percent": result.debt_repayment_percent_of_fresh_issue,
            "post_ipo_debt_cr": result.post_ipo_debt,
            "de_ratio_pre": result.debt_to_equity_pre_ipo,
            "de_ratio_post": result.debt_to_equity_post_ipo,
            "debt_to_ebitda": result.debt_to_ebitda,
            "interest_coverage": result.interest_coverage_ratio,
        }

        return summary

    def assess_debt_risk(self, result: DebtStructure) -> dict[str, str]:
        """
        Assess overall debt risk based on various factors.

        Args:
            result: DebtStructure analysis result

        Returns:
            Dictionary with risk assessments
        """
        risks = {}

        # Leverage risk
        if result.debt_to_equity_pre_ipo > 2.0:
            risks["leverage"] = "HIGH - D/E ratio exceeds 2x"
        elif result.debt_to_equity_pre_ipo > 1.0:
            risks["leverage"] = "MEDIUM - D/E ratio between 1-2x"
        else:
            risks["leverage"] = "LOW - D/E ratio below 1x"

        # Interest cost risk
        if result.weighted_avg_interest_rate and result.weighted_avg_interest_rate > 12:
            risks["interest_cost"] = "HIGH - Weighted average rate exceeds 12%"
        elif result.weighted_avg_interest_rate and result.weighted_avg_interest_rate > 9:
            risks["interest_cost"] = "MEDIUM - Weighted average rate 9-12%"
        else:
            risks["interest_cost"] = "LOW - Weighted average rate below 9%"

        # Refinancing risk
        maturity_risk = result.get_maturity_risk()
        risks[
            "refinancing"
        ] = f"{maturity_risk} - {result.maturity_profile.within_1_year:.1f} Cr maturing within 1 year"

        # Concentration risk
        conc_risk = result.get_concentration_risk()
        if result.top_lender:
            risks[
                "concentration"
            ] = f"{conc_risk} - Top lender {result.top_lender} has {result.top_lender_exposure:.1f} Cr exposure"
        else:
            risks["concentration"] = conc_risk

        # Covenant risk
        if result.covenant_breaches_disclosed:
            risks["covenant"] = "HIGH - Covenant breaches disclosed"
        elif result.has_financial_covenants:
            risks["covenant"] = "MEDIUM - Financial covenants in place"
        else:
            risks["covenant"] = "LOW - No restrictive covenants disclosed"

        # Interest coverage risk
        if result.interest_coverage_ratio:
            if result.interest_coverage_ratio < 1.5:
                risks[
                    "coverage"
                ] = f"HIGH - Interest coverage ratio {result.interest_coverage_ratio:.2f}x is below 1.5x"
            elif result.interest_coverage_ratio < 3.0:
                risks[
                    "coverage"
                ] = f"MEDIUM - Interest coverage ratio {result.interest_coverage_ratio:.2f}x is below 3x"
            else:
                risks["coverage"] = f"LOW - Interest coverage ratio {result.interest_coverage_ratio:.2f}x is healthy"

        return risks
