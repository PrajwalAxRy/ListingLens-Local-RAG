"""Historical financial normalization utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Optional

from loguru import logger

from .models import FinancialRecord


class HistoricalNormalizer:
    """Normalize raw financial disclosures into structured crore-denominated data."""

    UNIT_FACTORS = {
        "cr": 1.0,
        "crore": 1.0,
        "crores": 1.0,
        "crs": 1.0,
        "cro": 1.0,
        "lakh": 0.01,
        "lakhs": 0.01,
        "lac": 0.01,
        "lacs": 0.01,
        "million": 0.1,
        "millions": 0.1,
        "mn": 0.1,
        "billion": 100.0,
        "billions": 100.0,
    }

    def __init__(self) -> None:
        self.logger = logger.bind(module="historical_normalizer")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def normalize(
        self,
        records: Sequence[dict],
        *,
        prefer_consolidated: bool = True,
        default_statement: str = "consolidated",
    ) -> list[FinancialRecord]:
        """Normalize a list of raw financial rows.

        Args:
            records: Raw disclosures extracted from tables. Each record should contain
                ``fiscal_year``, ``values`` (dict of metrics), optional ``unit`` and
                ``statement_type``.
            prefer_consolidated: Whether to pick consolidated numbers when multiple
                statement types exist for the same fiscal year.
            default_statement: Statement label to fall back on when missing.

        Returns:
            List of :class:`FinancialRecord` entries sorted chronologically.
        """

        grouped: dict[str, dict[str, FinancialRecord]] = {}
        for entry in records:
            try:
                normalized = self._normalize_entry(entry, default_statement)
            except ValueError as exc:  # pragma: no cover - logged path
                self.logger.warning("Skipping record due to error: {}", exc)
                continue

            grouped.setdefault(normalized.fiscal_year, {})[normalized.statement_type] = normalized

        normalized_list: list[FinancialRecord] = []
        for fiscal_year in sorted(grouped, key=self._year_sort_key):
            options = grouped[fiscal_year]
            record = self._pick_preferred_record(options, prefer_consolidated)
            normalized_list.append(record)

        return normalized_list

    def calculate_cagr(
        self,
        records: Sequence[FinancialRecord],
        metric: str,
        *,
        periods: Optional[int] = None,
    ) -> Optional[float]:
        """Calculate CAGR for a metric across the provided records.

        Args:
            records: Chronologically ordered normalized records.
            metric: Attribute name on :class:`FinancialRecord` (e.g. ``"revenue"``).
            periods: Optional number of periods to consider from the tail of the
                series. Defaults to the full available window.
        """

        numeric_series: list[tuple[str, float]] = []
        for record in records:
            value = getattr(record, metric, None)
            if isinstance(value, (int, float)) and value > 0:
                numeric_series.append((record.fiscal_year, float(value)))

        if len(numeric_series) < 2:
            return None

        if periods is not None and periods >= 2:
            numeric_series = numeric_series[-periods:]

        start_value = numeric_series[0][1]
        end_value = numeric_series[-1][1]
        if start_value <= 0 or end_value <= 0:
            return None

        years = len(numeric_series) - 1
        if years <= 0:
            return None

        cagr = (end_value / start_value) ** (1 / years) - 1
        return round(cagr * 100, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_entry(self, entry: dict, default_statement: str) -> FinancialRecord:
        fiscal_year = entry.get("fiscal_year")
        if not fiscal_year:
            raise ValueError("Missing fiscal_year in financial record")

        unit = (entry.get("unit") or "cr").strip().lower()
        factor = self.UNIT_FACTORS.get(unit)
        if factor is None:
            raise ValueError(f"Unsupported unit '{unit}' for fiscal year {fiscal_year}")

        values = entry.get("values") or {}
        statement_type = (entry.get("statement_type") or default_statement).lower()
        source = entry.get("source")

        normalized_values = {key: self._to_crore(value, factor) for key, value in values.items()}

        record = FinancialRecord(
            fiscal_year=fiscal_year,
            revenue=normalized_values.get("revenue"),
            ebitda=normalized_values.get("ebitda"),
            pat=normalized_values.get("pat"),
            total_assets=normalized_values.get("total_assets"),
            total_equity=normalized_values.get("total_equity"),
            total_debt=normalized_values.get("total_debt"),
            cfo=normalized_values.get("cfo"),
            ccc_days=normalized_values.get("cash_conversion_cycle"),
            statement_type=statement_type,
            source=source,
        )
        self.logger.debug("Normalized record: {}", asdict(record))
        return record

    def _pick_preferred_record(
        self,
        options: dict[str, FinancialRecord],
        prefer_consolidated: bool,
    ) -> FinancialRecord:
        if prefer_consolidated:
            for key in ("consolidated", "standalone"):
                if key in options:
                    return options[key]
        else:
            for key in ("standalone", "consolidated"):
                if key in options:
                    return options[key]
        return next(iter(options.values()))

    def _to_crore(self, value: Optional[object], factor: float) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return round(float(value) * factor, 4)

        string_value = str(value).strip()
        if not string_value:
            return None

        negative = string_value.startswith("(") and string_value.endswith(")")
        sanitized = string_value.replace(",", "").replace("₹", "")
        sanitized = sanitized.replace("Rs.", "").replace("INR", "").strip("() ")
        if not sanitized:
            return None

        try:
            number = float(sanitized)
        except ValueError as exc:  # pragma: no cover - logged path
            self.logger.warning("Unable to parse numeric value '%s': %s", value, exc)
            return None

        if negative:
            number *= -1
        return round(number * factor, 4)

    def _year_sort_key(self, fiscal_year: str) -> tuple[int, str]:
        digits = "".join(ch for ch in fiscal_year if ch.isdigit())
        try:
            return int(digits[-2:]), fiscal_year
        except ValueError:
            return 0, fiscal_year


__all__ = ["HistoricalNormalizer"]
