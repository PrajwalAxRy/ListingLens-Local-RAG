"""Citation and Audit Trail Manager for sourced claims.

Enforces machine-readable citations for all numerical or qualitative claims,
provides validation hooks for the Self-Critic agent, and auto-generates
footnotes for reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class CitationRecord:
    """Audit trail record for a sourced claim."""

    claim_id: str
    section: str
    page: int
    paragraph_label: Optional[str] = None
    text_snippet: str = ""
    table_id: Optional[str] = None


class CitationManager:
    """Manage audit trail for all sourced claims in an RHP analysis.

    Provides:
    - Citation storage per document
    - Lookup by claim ID
    - Validation (check if citation exists)
    - Report footnote generation
    """

    def __init__(self) -> None:
        self._store: dict[str, CitationRecord] = {}
        self.logger = logger.bind(module="citation_manager")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def attach(
        self,
        claim_id: str,
        section: str,
        page: int,
        text_snippet: str = "",
        *,
        paragraph_label: Optional[str] = None,
        table_id: Optional[str] = None,
    ) -> CitationRecord:
        """Attach a citation to a claim.

        Args:
            claim_id: Unique identifier for the claim (e.g., "revenue_cagr_fy24").
            section: RHP section name (e.g., "Financial Information").
            page: Page number(s) in the RHP.
            text_snippet: Short snippet from source (max ~280 chars).
            paragraph_label: Optional paragraph/table reference.
            table_id: Optional table identifier if claim is from a table.

        Returns:
            The created CitationRecord.
        """
        citation = CitationRecord(
            claim_id=claim_id,
            section=section,
            page=page,
            paragraph_label=paragraph_label,
            text_snippet=text_snippet[:280] if text_snippet else "",
            table_id=table_id,
        )
        self._store[claim_id] = citation
        self.logger.debug(
            "Attached citation for claim '{}': {} p.{}",
            claim_id,
            section,
            page,
        )
        return citation

    def add_citation(
        self,
        claim_id: str,
        section: str,
        page: int,
        snippet: str = "",
    ) -> CitationRecord:
        """Alias for attach() with simplified signature.

        Args:
            claim_id: Unique identifier for the claim.
            section: RHP section name.
            page: Page number.
            snippet: Text snippet from source.

        Returns:
            The created CitationRecord.
        """
        return self.attach(claim_id, section, page, snippet)

    def get(self, claim_id: str) -> Optional[CitationRecord]:
        """Retrieve a citation by claim ID.

        Args:
            claim_id: The claim identifier.

        Returns:
            CitationRecord if found, None otherwise.
        """
        return self._store.get(claim_id)

    def validate_claim(self, claim_id: str) -> bool:
        """Check if a citation exists for a claim.

        Used by Self-Critic agent to reject outputs lacking citations.

        Args:
            claim_id: The claim identifier.

        Returns:
            True if citation exists, False otherwise.
        """
        exists = claim_id in self._store
        if not exists:
            self.logger.warning("No citation found for claim: {}", claim_id)
        return exists

    def validate_claims(self, claim_ids: list[str]) -> dict[str, bool]:
        """Validate multiple claims at once.

        Args:
            claim_ids: List of claim identifiers.

        Returns:
            Dict mapping claim_id to validation result.
        """
        return {cid: self.validate_claim(cid) for cid in claim_ids}

    def get_uncited_claims(self, expected_claims: list[str]) -> list[str]:
        """Get list of claims that are missing citations.

        Args:
            expected_claims: List of expected claim IDs.

        Returns:
            List of claim IDs without citations.
        """
        return [cid for cid in expected_claims if cid not in self._store]

    def generate_footnotes(self, *, format_style: str = "markdown") -> str:
        """Auto-generate footnotes for report from stored citations.

        Args:
            format_style: Output format ("markdown" or "text").

        Returns:
            Formatted footnotes as a string.
        """
        if not self._store:
            return ""

        lines: list[str] = []
        sorted_citations = sorted(
            self._store.items(),
            key=lambda x: (x[1].section, x[1].page),
        )

        for i, (claim_id, citation) in enumerate(sorted_citations, start=1):
            if format_style == "markdown":
                footnote = self._format_markdown_footnote(i, citation)
            else:
                footnote = self._format_text_footnote(i, citation)
            lines.append(footnote)

        return "\n".join(lines)

    def export_citations(self) -> list[dict]:
        """Export all citations as a list of dicts for serialization.

        Returns:
            List of citation dicts.
        """
        return [
            {
                "claim_id": c.claim_id,
                "section": c.section,
                "page": c.page,
                "paragraph_label": c.paragraph_label,
                "text_snippet": c.text_snippet,
                "table_id": c.table_id,
            }
            for c in self._store.values()
        ]

    def import_citations(self, citations: list[dict]) -> int:
        """Import citations from a list of dicts.

        Args:
            citations: List of citation dicts.

        Returns:
            Number of citations imported.
        """
        count = 0
        for cit in citations:
            claim_id = cit.get("claim_id")
            if claim_id:
                record = CitationRecord(
                    claim_id=claim_id,
                    section=cit.get("section", ""),
                    page=cit.get("page", 0),
                    paragraph_label=cit.get("paragraph_label"),
                    text_snippet=cit.get("text_snippet", ""),
                    table_id=cit.get("table_id"),
                )
                self._store[claim_id] = record
                count += 1
        return count

    def clear(self) -> None:
        """Clear all stored citations."""
        self._store.clear()
        self.logger.debug("Cleared all citations")

    @property
    def count(self) -> int:
        """Number of stored citations."""
        return len(self._store)

    def __len__(self) -> int:
        """Number of stored citations."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _format_markdown_footnote(
        self,
        index: int,
        citation: CitationRecord,
    ) -> str:
        """Format a single citation as markdown footnote."""
        base = f"[^{index}]: [{citation.section}, p. {citation.page}]"
        if citation.text_snippet:
            base += f' — "{citation.text_snippet}"'
        return base

    def _format_text_footnote(
        self,
        index: int,
        citation: CitationRecord,
    ) -> str:
        """Format a single citation as plain text footnote."""
        base = f"[{index}] {citation.section}, Page {citation.page}"
        if citation.text_snippet:
            base += f': "{citation.text_snippet}"'
        return base


__all__ = ["CitationManager", "CitationRecord"]
