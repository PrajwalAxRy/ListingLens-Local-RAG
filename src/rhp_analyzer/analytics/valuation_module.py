"""Valuation normalization and peer comparison module.

Extracts and normalizes peers from RHP disclosures to calculate consistent
multiples at floor/cap price points.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Optional

from loguru import logger

from .models import PeerComparable, ValuationSummary


@dataclass
class IssuerMetrics:
    """Issuer's latest financials and IPO pricing for valuation."""

    name: str
    pat_cr: float
    net_worth_cr: float
    ebitda_cr: float
    net_debt_cr: float
    pat_cagr_3yr: Optional[float] = None
    shares_post_issue: int = 0
    price_floor: float = 0.0
    price_cap: float = 0.0


class ValuationNormalization:
    """Normalize peer valuations and calculate issuer multiples."""

    def __init__(self) -> None:
        self.logger = logger.bind(module="valuation_module")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def normalize_peers(
        self,
        raw_peers: list[dict],
        issuer: IssuerMetrics,
        *,
        industry_peers: Optional[list[str]] = None,
    ) -> ValuationSummary:
        """Normalize peer comparables and compute issuer valuation metrics.

        Args:
            raw_peers: List of peer dicts extracted from RHP 'Basis for Issue Price'
                       table. Keys may include: name, market_cap, pat, net_worth,
                       ebitda, net_debt, pe, pb, ev_ebitda, fiscal_year, citation.
            issuer: Issuer's metrics and pricing.
            industry_peers: Peer names mentioned in Industry Overview for
                            missing-peer flagging.

        Returns:
            ValuationSummary with normalized peers, medians, issuer metrics, and
            premium/discount calculations.
        """
        peers: list[PeerComparable] = []
        for raw in raw_peers:
            peer = self._parse_peer(raw)
            if peer:
                peers.append(peer)

        peer_medians = self._compute_medians(peers)
        issuer_floor = self._compute_issuer_metrics(issuer, issuer.price_floor)
        issuer_cap = self._compute_issuer_metrics(issuer, issuer.price_cap)
        premium_discount = self._compute_premium_discount(issuer_cap, peer_medians)

        missing = self._identify_missing_peers(peers, industry_peers)

        summary = ValuationSummary(
            peer_multiples=peers,
            peer_medians=peer_medians,
            issuer_floor_metrics=issuer_floor,
            issuer_cap_metrics=issuer_cap,
            premium_discount_vs_peers=premium_discount,
            missing_peers=missing,
        )

        self.logger.info(
            "Valuation summary: {} peers, issuer P/E floor={:.1f} cap={:.1f}",
            len(peers),
            issuer_floor.get("pe", 0),
            issuer_cap.get("pe", 0),
        )
        return summary

    def compute_peg(
        self,
        pe: float,
        growth_rate: Optional[float],
    ) -> Optional[float]:
        """Compute PEG ratio (P/E to growth).

        Args:
            pe: Price to earnings ratio.
            growth_rate: PAT CAGR in percentage (e.g. 15 for 15%).

        Returns:
            PEG ratio or None if growth unavailable.
        """
        if growth_rate is None or growth_rate <= 0:
            return None
        return round(pe / growth_rate, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_peer(self, raw: dict) -> Optional[PeerComparable]:
        """Parse raw peer dict into PeerComparable dataclass."""
        name = raw.get("name")
        if not name:
            return None

        market_cap = self._safe_float(raw.get("market_cap"))
        pat = self._safe_float(raw.get("pat"))
        net_worth = self._safe_float(raw.get("net_worth"))
        ebitda = self._safe_float(raw.get("ebitda"))
        net_debt = self._safe_float(raw.get("net_debt")) or 0.0

        # Calculate EV if not provided
        ev = self._safe_float(raw.get("enterprise_value"))
        if ev is None and market_cap is not None:
            ev = market_cap + (net_debt or 0.0)

        # Calculate multiples if not provided
        pe = self._safe_float(raw.get("pe"))
        if pe is None and market_cap and pat and pat > 0:
            pe = round(market_cap / pat, 2)

        pb = self._safe_float(raw.get("pb"))
        if pb is None and market_cap and net_worth and net_worth > 0:
            pb = round(market_cap / net_worth, 2)

        ev_ebitda = self._safe_float(raw.get("ev_ebitda"))
        if ev_ebitda is None and ev and ebitda and ebitda > 0:
            ev_ebitda = round(ev / ebitda, 2)

        peg = self._safe_float(raw.get("peg"))

        return PeerComparable(
            name=name,
            fiscal_year=raw.get("fiscal_year"),
            market_cap_cr=market_cap,
            pat_cr=pat,
            net_worth_cr=net_worth,
            ebitda_cr=ebitda,
            net_debt_cr=net_debt,
            enterprise_value_cr=ev,
            pe=pe,
            pb=pb,
            ev_ebitda=ev_ebitda,
            peg=peg,
            citation=raw.get("citation"),
        )

    def _compute_medians(self, peers: list[PeerComparable]) -> dict[str, float]:
        """Compute median multiples from peers."""
        pe_vals = [p.pe for p in peers if p.pe is not None and p.pe > 0]
        pb_vals = [p.pb for p in peers if p.pb is not None and p.pb > 0]
        ev_vals = [p.ev_ebitda for p in peers if p.ev_ebitda is not None and p.ev_ebitda > 0]

        return {
            "pe": round(median(pe_vals), 2) if pe_vals else 0.0,
            "pb": round(median(pb_vals), 2) if pb_vals else 0.0,
            "ev_ebitda": round(median(ev_vals), 2) if ev_vals else 0.0,
        }

    def _compute_issuer_metrics(
        self,
        issuer: IssuerMetrics,
        price: float,
    ) -> dict[str, float]:
        """Calculate issuer multiples at given price."""
        if issuer.shares_post_issue <= 0 or price <= 0:
            return {"pe": 0.0, "pb": 0.0, "ev_ebitda": 0.0, "peg": 0.0, "market_cap": 0.0}

        # Market cap in crores
        market_cap = (price * issuer.shares_post_issue) / 1e7

        pe = round(market_cap / issuer.pat_cr, 2) if issuer.pat_cr > 0 else 0.0
        pb = round(market_cap / issuer.net_worth_cr, 2) if issuer.net_worth_cr > 0 else 0.0

        ev = market_cap + issuer.net_debt_cr
        ev_ebitda = round(ev / issuer.ebitda_cr, 2) if issuer.ebitda_cr > 0 else 0.0

        peg = self.compute_peg(pe, issuer.pat_cagr_3yr) or 0.0

        return {
            "market_cap": round(market_cap, 2),
            "pe": pe,
            "pb": pb,
            "ev_ebitda": ev_ebitda,
            "peg": peg,
        }

    def _compute_premium_discount(
        self,
        issuer_metrics: dict[str, float],
        peer_medians: dict[str, float],
    ) -> dict[str, float]:
        """Calculate premium/discount vs peer medians."""
        result: dict[str, float] = {}
        for key in ["pe", "pb", "ev_ebitda"]:
            issuer_val = issuer_metrics.get(key, 0.0)
            peer_val = peer_medians.get(key, 0.0)
            if peer_val > 0:
                pct = round((issuer_val / peer_val - 1) * 100, 2)
                result[key] = pct
            else:
                result[key] = 0.0
        return result

    def _identify_missing_peers(
        self,
        peers: list[PeerComparable],
        industry_peers: Optional[list[str]],
    ) -> list[str]:
        """Identify peers mentioned in Industry Overview but excluded from pricing."""
        if not industry_peers:
            return []

        peer_names_lower = {p.name.lower() for p in peers}
        missing: list[str] = []
        for name in industry_peers:
            if name.lower() not in peer_names_lower:
                missing.append(name)
        return missing

    def _safe_float(self, value: Optional[object]) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(",", "").strip())
        except ValueError:
            return None


__all__ = [
    "IssuerMetrics",
    "ValuationNormalization",
]
