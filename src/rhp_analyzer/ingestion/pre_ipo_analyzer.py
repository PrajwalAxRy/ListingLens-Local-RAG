"""Pre-IPO investor analyzer module.

This module structures historical capital structure disclosures into investor-level
insights such as implied returns, holding periods, and lock-in schedules.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from loguru import logger

RUPEES_PER_CRORE = 10_000_000


@dataclass
class PreIPOInvestor:
    """Structured representation of a pre-IPO investor."""
    name: str
    category: str
    entry_date: Optional[str] = None
    entry_price: Optional[float] = None
    shares_acquired: int = 0
    investment_amount: float = 0.0  # ₹ Cr
    shares_held_pre_ipo: int = 0
    holding_percent_pre_ipo: float = 0.0
    shares_selling_via_ofs: int = 0
    ofs_amount: float = 0.0  # ₹ Cr
    implied_return_multiple_at_floor: Optional[float] = None
    implied_return_multiple_at_cap: Optional[float] = None
    implied_irr_at_floor: Optional[float] = None
    implied_irr_at_cap: Optional[float] = None
    holding_period_months: Optional[int] = None
    lock_in_period: Optional[str] = None
    lock_in_expiry_date: Optional[str] = None
    shares_locked: int = 0
    shares_free_post_listing: int = 0


class PreIPOInvestorAnalyzer:
    """Analyze pre-IPO investor entries and exits."""

    def __init__(self, citation_manager: Optional[object] = None) -> None:
        self.citation_manager = citation_manager
        self.logger = logger.bind(module="pre_ipo_analyzer")

    def analyze_investors(self, state: Dict) -> List[PreIPOInvestor]:
        """Return structured investor summaries using state data."""
        capital_history = self._parse_capital_structure_history(state)
        if not capital_history:
            self.logger.warning("No capital structure history found in state")
            return []

        ofs_details = self._parse_ofs_details(state)
        lock_in_schedule = self._parse_lock_in_schedule(state)
        ipo_details = state.get("ipo_details", {})
        floor_price, cap_price = self._get_price_band(ipo_details)
        listing_date = ipo_details.get("listing_date")
        total_shares_pre_ipo = self._get_total_pre_ipo_shares(state)

        investors: List[PreIPOInvestor] = []
        for investor_name, entries in capital_history.items():
            sorted_entries = sorted(entries, key=lambda item: item["date"] or "")
            total_shares = sum(item["shares"] for item in sorted_entries)
            total_investment = sum(item["shares"] * item["price"] for item in sorted_entries)
            avg_entry_price = self._safe_divide(total_investment, total_shares)
            entry_date = sorted_entries[0]["date"] if sorted_entries else None
            holding_period_months = self._calculate_holding_period_months(entry_date, listing_date)

            ofs_info = ofs_details.get(investor_name, {})
            lock_info = lock_in_schedule.get(investor_name, {})
            shares_selling = ofs_info.get("shares", 0)
            shares_locked = lock_info.get("shares_locked", 0)
            shares_free = max(total_shares - shares_locked, 0)
            shares_held_percent = self._safe_divide(total_shares * 100, total_shares_pre_ipo) or 0.0

            investor = PreIPOInvestor(
                name=investor_name,
                category=self._classify_investor(investor_name),
                entry_date=entry_date,
                entry_price=avg_entry_price,
                shares_acquired=total_shares,
                investment_amount=self._rupees_to_crore(total_investment),
                shares_held_pre_ipo=total_shares,
                holding_percent_pre_ipo=shares_held_percent,
                shares_selling_via_ofs=shares_selling,
                ofs_amount=self._rupees_to_crore(shares_selling * (cap_price or floor_price or 0.0)),
                implied_return_multiple_at_floor=self._calculate_return_multiple(floor_price, avg_entry_price),
                implied_return_multiple_at_cap=self._calculate_return_multiple(cap_price, avg_entry_price),
                implied_irr_at_floor=self._calculate_irr(avg_entry_price, floor_price, holding_period_months),
                implied_irr_at_cap=self._calculate_irr(avg_entry_price, cap_price, holding_period_months),
                holding_period_months=holding_period_months,
                lock_in_period=lock_info.get("period"),
                lock_in_expiry_date=lock_info.get("expiry"),
                shares_locked=shares_locked,
                shares_free_post_listing=shares_free,
            )
            investors.append(investor)

        self.logger.info("Generated {} pre-IPO investor profiles", len(investors))
        return investors

    def _parse_capital_structure_history(self, state: Dict) -> Dict[str, List[Dict[str, float]]]:
        """Group capital structure entries by investor."""
        history = state.get("capital_structure_history", [])
        grouped: Dict[str, List[Dict[str, float]]] = {}
        for row in history:
            investor = row.get("investor")
            if not investor:
                continue
            grouped.setdefault(investor, []).append(
                {
                    "date": row.get("date"),
                    "price": float(row.get("price", 0.0)),
                    "shares": int(row.get("shares", 0)),
                }
            )
        return grouped

    def _parse_ofs_details(self, state: Dict) -> Dict[str, Dict[str, float]]:
        """Map OFS participation by investor."""
        ofs_entries = state.get("ofs_details", [])
        details: Dict[str, Dict[str, float]] = {}
        for row in ofs_entries:
            investor = row.get("investor")
            if not investor:
                continue
            details[investor] = {
                "shares": int(row.get("shares", 0)),
                "price": float(row.get("price", 0.0)),
            }
        return details

    def _parse_lock_in_schedule(self, state: Dict) -> Dict[str, Dict[str, Optional[str]]]:
        """Return lock-in metadata keyed by investor."""
        schedule = state.get("lock_in_schedule", [])
        details: Dict[str, Dict[str, Optional[str]]] = {}
        for row in schedule:
            investor = row.get("investor")
            if not investor:
                continue
            details[investor] = {
                "period": row.get("period"),
                "expiry": row.get("expiry"),
                "shares_locked": int(row.get("shares_locked", 0)),
            }
        return details

    def _calculate_holding_period_months(self, entry_date: Optional[str], listing_date: Optional[str]) -> Optional[int]:
        """Return months between entry and listing dates."""
        if not entry_date or not listing_date:
            return None
        start = self._parse_date(entry_date)
        end = self._parse_date(listing_date)
        if not start or not end or end <= start:
            return None
        months = (end.year - start.year) * 12 + (end.month - start.month)
        if end.day < start.day:
            months -= 1
        return max(months, 0)

    def _calculate_return_multiple(self, exit_price: Optional[float], entry_price: Optional[float]) -> Optional[float]:
        """Return simple price multiple."""
        if not exit_price or not entry_price:
            return None
        if entry_price <= 0:
            return None
        return exit_price / entry_price

    def _calculate_irr(self, entry_price: Optional[float], exit_price: Optional[float], holding_period_months: Optional[int]) -> Optional[float]:
        """Return annualized IRR % based on entry/exit prices."""
        if not entry_price or not exit_price or not holding_period_months:
            return None
        if entry_price <= 0 or holding_period_months <= 0:
            return None
        years = holding_period_months / 12
        if years <= 0:
            return None
        irr = (exit_price / entry_price) ** (1 / years) - 1
        return irr * 100

    def _classify_investor(self, name: str) -> str:
        """Heuristic classification based on investor name."""
        lowered = name.lower()
        if any(keyword in lowered for keyword in {"fund", "capital", "partners", "ventures"}):
            return "PE/VC"
        if any(keyword in lowered for keyword in {"trust", "esop"}):
            return "ESOP Trust"
        if "promoter" in lowered:
            return "Promoter"
        return "Other"

    def _get_price_band(self, ipo_details: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Extract floor and cap prices."""
        floor = ipo_details.get("price_band_floor") or ipo_details.get("floor_price")
        cap = ipo_details.get("price_band_cap") or ipo_details.get("cap_price")
        return floor, cap

    def _get_total_pre_ipo_shares(self, state: Dict) -> int:
        """Return pre-issue share count if available."""
        share_capital = state.get("share_capital", {})
        return int(share_capital.get("pre_issue_shares", 0))

    def _parse_date(self, value: str) -> Optional[datetime]:
        """Try to parse YYYY-MM-DD dates."""
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except (TypeError, ValueError):
            return None

    def _rupees_to_crore(self, value: float) -> float:
        """Convert rupees to crore with 2 decimal precision."""
        return round(value / RUPEES_PER_CRORE, 4) if value else 0.0

    def _safe_divide(self, numerator: float, denominator: float) -> Optional[float]:
        """Avoid ZeroDivisionError."""
        if not denominator:
            return None
        return numerator / denominator
