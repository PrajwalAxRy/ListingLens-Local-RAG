"""Order book analyzer module for RHP Analyzer."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from loguru import logger

APPLICABLE_SECTORS = {
    "epc",
    "defense",
    "infrastructure",
    "capital goods",
    "engineering",
    "construction",
    "it services",
    "industrial services",
}


@dataclass
class OrderBookAnalysis:
    """Structured order-book insights for transparency."""

    applicable: bool = False
    total_order_book: float = 0.0
    order_book_as_of_date: Optional[str] = None
    order_book_to_ltm_revenue: Optional[float] = None
    top_5_orders_value: float = 0.0
    top_5_orders_concentration: Optional[float] = None
    largest_single_order: float = 0.0
    largest_single_order_percent: Optional[float] = None
    executable_in_12_months: float = 0.0
    executable_in_12_months_percent: Optional[float] = None
    order_book_1yr_ago: Optional[float] = None
    order_book_growth_yoy: Optional[float] = None
    government_orders_percent: Optional[float] = None
    private_orders_percent: Optional[float] = None
    repeat_customer_orders_percent: Optional[float] = None


class OrderBookAnalyzer:
    """Analyze order-book disclosures for applicable sectors."""

    def __init__(self) -> None:
        self.logger = logger.bind(module="order_book_analyzer")

    def analyze_order_book(self, state: Dict, sector: str) -> OrderBookAnalysis:
        """Return an analysis of the disclosed order book."""
        if not self._is_applicable_sector(sector):
            return OrderBookAnalysis(applicable=False)

        analysis = OrderBookAnalysis(applicable=True)
        data = self._extract_order_book_data(state)
        if not data:
            self.logger.warning("Order book data missing for applicable sector")
            return analysis

        analysis.total_order_book = float(data.get("total_order_book", 0.0))
        analysis.order_book_as_of_date = data.get("as_of_date")
        ltm_revenue = self._get_ltm_revenue(state)
        analysis.order_book_to_ltm_revenue = self._safe_divide(analysis.total_order_book, ltm_revenue)

        top_orders: List[float] = data.get("top_orders", [])
        analysis.top_5_orders_value = sum(top_orders[:5])
        if analysis.total_order_book:
            analysis.top_5_orders_concentration = (
                analysis.top_5_orders_value / analysis.total_order_book
            ) * 100

        largest_order = data.get("largest_order") or (max(top_orders) if top_orders else 0.0)
        analysis.largest_single_order = largest_order
        if analysis.total_order_book and largest_order:
            analysis.largest_single_order_percent = (largest_order / analysis.total_order_book) * 100

        analysis.executable_in_12_months = float(data.get("executable_in_12_months", 0.0))
        if analysis.total_order_book and analysis.executable_in_12_months:
            analysis.executable_in_12_months_percent = (
                analysis.executable_in_12_months / analysis.total_order_book
            ) * 100

        analysis.order_book_1yr_ago = data.get("order_book_1yr_ago")
        if analysis.order_book_1yr_ago:
            analysis.order_book_growth_yoy = (
                (analysis.total_order_book / analysis.order_book_1yr_ago) - 1
            ) * 100

        analysis.government_orders_percent = data.get("government_orders_percent")
        analysis.private_orders_percent = data.get("private_orders_percent")
        analysis.repeat_customer_orders_percent = data.get("repeat_customer_orders_percent")

        self.logger.info(
            "Order book coverage {:.1f}x LTM revenue", analysis.order_book_to_ltm_revenue or 0.0
        )
        return analysis

    def _is_applicable_sector(self, sector: str) -> bool:
        """Return True if order-book intel is expected for the sector."""
        if not sector:
            return False
        return sector.lower() in APPLICABLE_SECTORS

    def _extract_order_book_data(self, state: Dict) -> Optional[Dict]:
        """Fetch normalized order-book payload from state."""
        if "order_book_data" in state:
            return state["order_book_data"]
        sections = state.get("sections", {})
        return sections.get("Order Book", {}).get("structured_data")

    def _get_ltm_revenue(self, state: Dict) -> Optional[float]:
        """Best-effort fetch of latest revenue for ratio math."""
        financials = state.get("financial_data", {})
        return financials.get("ltm_revenue") or financials.get("latest_revenue")

    def _safe_divide(self, numerator: float, denominator: Optional[float]) -> Optional[float]:
        """Return quotient when safe to divide."""
        if not denominator:
            return None
        return numerator / denominator
