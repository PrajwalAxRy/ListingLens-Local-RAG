"""
Financial Parser Module for RHP Analysis.

This module extracts and computes financial metrics from parsed tables,
calculates key financial ratios, and detects window dressing signals.

Subtask 2.5A.0: Financial Parser Module
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import re


@dataclass
class FinancialMetrics:
    """Financial metrics for a single fiscal year."""
    
    fiscal_year: str
    
    # Core financial metrics (in crores)
    revenue: float = 0.0
    cost_of_goods_sold: float = 0.0
    gross_profit: float = 0.0
    ebitda: float = 0.0
    depreciation: float = 0.0
    ebit: float = 0.0
    interest_expense: float = 0.0
    pbt: float = 0.0  # Profit Before Tax
    tax_expense: float = 0.0
    pat: float = 0.0  # Profit After Tax
    
    # Balance sheet items
    total_assets: float = 0.0
    total_equity: float = 0.0
    total_debt: float = 0.0
    current_assets: float = 0.0
    current_liabilities: float = 0.0
    cash_and_equivalents: float = 0.0
    inventory: float = 0.0
    trade_receivables: float = 0.0
    trade_payables: float = 0.0
    net_fixed_assets: float = 0.0
    
    # Cash flow items
    cfo: float = 0.0  # Cash Flow from Operations
    cfi: float = 0.0  # Cash Flow from Investing
    cff: float = 0.0  # Cash Flow from Financing
    capex: float = 0.0
    working_capital_change: float = 0.0
    
    # Calculated ratios (populated by calculate_ratios)
    roe: Optional[float] = None
    roce: Optional[float] = None
    debt_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    gross_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    pat_margin: Optional[float] = None
    asset_turnover: Optional[float] = None
    interest_coverage: Optional[float] = None
    receivable_days: Optional[float] = None
    inventory_days: Optional[float] = None
    payable_days: Optional[float] = None
    cash_conversion_cycle: Optional[float] = None
    
    # Metadata
    is_consolidated: bool = True
    restated: bool = False
    audited: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fiscal_year': self.fiscal_year,
            'revenue': self.revenue,
            'ebitda': self.ebitda,
            'pat': self.pat,
            'total_assets': self.total_assets,
            'total_equity': self.total_equity,
            'total_debt': self.total_debt,
            'roe': self.roe,
            'roce': self.roce,
            'debt_equity': self.debt_equity,
            'ebitda_margin': self.ebitda_margin,
            'pat_margin': self.pat_margin,
            'receivable_days': self.receivable_days,
            'inventory_days': self.inventory_days,
            'cash_conversion_cycle': self.cash_conversion_cycle,
            'cfo': self.cfo,
            'is_consolidated': self.is_consolidated,
        }


@dataclass
class NewAgeMetrics:
    """Metrics for loss-making/startup IPOs."""
    
    fiscal_year: str
    
    # Contribution and unit economics
    contribution_margin: Optional[float] = None  # (Revenue - Variable Costs) / Revenue
    gross_margin: Optional[float] = None
    
    # Customer metrics
    cac: Optional[float] = None  # Customer Acquisition Cost
    ltv: Optional[float] = None  # Customer Lifetime Value
    cac_ltv_ratio: Optional[float] = None  # LTV / CAC (should be > 3x)
    
    # Cash metrics
    burn_rate: Optional[float] = None  # Monthly cash consumption
    runway_months: Optional[int] = None  # Months of cash remaining
    
    # Growth metrics
    revenue_growth_rate: Optional[float] = None
    user_growth_rate: Optional[float] = None
    
    # Unit economics
    revenue_per_user: Optional[float] = None
    gmv: Optional[float] = None  # Gross Merchandise Value
    take_rate: Optional[float] = None  # Revenue / GMV for marketplaces
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fiscal_year': self.fiscal_year,
            'contribution_margin': self.contribution_margin,
            'burn_rate': self.burn_rate,
            'runway_months': self.runway_months,
            'cac_ltv_ratio': self.cac_ltv_ratio,
            'revenue_per_user': self.revenue_per_user,
        }


@dataclass
class DivergenceWarning:
    """Represents a window dressing or divergence warning."""
    
    fiscal_year: str
    warning_type: str  # 'channel_stuffing', 'paper_profits', 'inventory_piling'
    severity: str  # 'CRITICAL', 'MAJOR', 'MINOR'
    description: str
    metric_name: str
    actual_value: float
    threshold_value: float
    citation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fiscal_year': self.fiscal_year,
            'warning_type': self.warning_type,
            'severity': self.severity,
            'description': self.description,
            'metric_name': self.metric_name,
            'actual_value': self.actual_value,
            'threshold_value': self.threshold_value,
            'citation': self.citation,
        }


@dataclass
class TrendAnalysis:
    """Analysis of financial trends over time."""
    
    revenue_cagr: Optional[float] = None
    ebitda_cagr: Optional[float] = None
    pat_cagr: Optional[float] = None
    
    margin_trend: str = "stable"  # 'expanding', 'contracting', 'stable'
    ebitda_margin_change: Optional[float] = None  # First year to last year
    
    roe_trend: str = "stable"
    roe_change: Optional[float] = None
    
    debt_trend: str = "stable"  # 'increasing', 'decreasing', 'stable'
    debt_equity_change: Optional[float] = None
    
    working_capital_trend: str = "stable"
    ccc_change: Optional[float] = None
    
    anomalies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'revenue_cagr': self.revenue_cagr,
            'ebitda_cagr': self.ebitda_cagr,
            'pat_cagr': self.pat_cagr,
            'margin_trend': self.margin_trend,
            'ebitda_margin_change': self.ebitda_margin_change,
            'roe_trend': self.roe_trend,
            'debt_trend': self.debt_trend,
            'working_capital_trend': self.working_capital_trend,
            'anomalies': self.anomalies,
        }


class FinancialParser:
    """
    Extracts and computes financial metrics from parsed tables.
    
    This parser handles:
    - Parsing P&L, Balance Sheet, and Cash Flow statements
    - Calculating key financial ratios
    - Detecting window dressing signals
    - Identifying trends and anomalies
    - Supporting new-age metrics for startup IPOs
    """
    
    # Indian number format patterns
    CRORE_PATTERN = re.compile(r'(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)\s*(?:Cr|Crore|Crores)', re.IGNORECASE)
    LAKH_PATTERN = re.compile(r'(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)\s*(?:Lakh|Lakhs|L)', re.IGNORECASE)
    NUMBER_PATTERN = re.compile(r'^[\(\-]?([\d,]+\.?\d*)[\)]?$')
    
    def __init__(self, table_extractor=None):
        """
        Initialize the financial parser.
        
        Args:
            table_extractor: Optional TableExtractor instance for extracting tables
        """
        self.table_extractor = table_extractor
    
    def parse_financial_statements(
        self,
        tables: List[Dict[str, Any]],
        fiscal_years: Optional[List[str]] = None
    ) -> List[FinancialMetrics]:
        """
        Parse P&L, Balance Sheet, and Cash Flow into structured metrics.
        
        Args:
            tables: List of table dictionaries with 'type', 'data' keys
            fiscal_years: Optional list of fiscal years to extract
            
        Returns:
            List of FinancialMetrics, one per fiscal year
        """
        metrics_by_year: Dict[str, FinancialMetrics] = {}
        
        for table in tables:
            table_type = table.get('type', '').lower()
            data = table.get('data', {})
            
            if 'income' in table_type or 'p&l' in table_type or 'profit' in table_type:
                self._parse_income_statement(data, metrics_by_year)
            elif 'balance' in table_type or 'position' in table_type:
                self._parse_balance_sheet(data, metrics_by_year)
            elif 'cash' in table_type and 'flow' in table_type:
                self._parse_cash_flow(data, metrics_by_year)
        
        # Convert to list and sort by fiscal year
        metrics_list = list(metrics_by_year.values())
        metrics_list.sort(key=lambda m: m.fiscal_year)
        
        # Calculate ratios for each year
        for metrics in metrics_list:
            self._calculate_ratios_for_metrics(metrics)
        
        return metrics_list
    
    def _parse_income_statement(
        self,
        data: Dict[str, Any],
        metrics_by_year: Dict[str, FinancialMetrics]
    ) -> None:
        """Parse income statement data into metrics."""
        for year, values in data.items():
            if year not in metrics_by_year:
                metrics_by_year[year] = FinancialMetrics(fiscal_year=year)
            
            metrics = metrics_by_year[year]
            
            if isinstance(values, dict):
                metrics.revenue = self._parse_amount(values.get('revenue', 0))
                metrics.cost_of_goods_sold = self._parse_amount(values.get('cogs', 0))
                metrics.gross_profit = self._parse_amount(values.get('gross_profit', 0))
                metrics.ebitda = self._parse_amount(values.get('ebitda', 0))
                metrics.depreciation = self._parse_amount(values.get('depreciation', 0))
                metrics.ebit = self._parse_amount(values.get('ebit', 0))
                metrics.interest_expense = self._parse_amount(values.get('interest', 0))
                metrics.pbt = self._parse_amount(values.get('pbt', 0))
                metrics.tax_expense = self._parse_amount(values.get('tax', 0))
                metrics.pat = self._parse_amount(values.get('pat', 0))
    
    def _parse_balance_sheet(
        self,
        data: Dict[str, Any],
        metrics_by_year: Dict[str, FinancialMetrics]
    ) -> None:
        """Parse balance sheet data into metrics."""
        for year, values in data.items():
            if year not in metrics_by_year:
                metrics_by_year[year] = FinancialMetrics(fiscal_year=year)
            
            metrics = metrics_by_year[year]
            
            if isinstance(values, dict):
                metrics.total_assets = self._parse_amount(values.get('total_assets', 0))
                metrics.total_equity = self._parse_amount(values.get('total_equity', 0))
                metrics.total_debt = self._parse_amount(values.get('total_debt', 0))
                metrics.current_assets = self._parse_amount(values.get('current_assets', 0))
                metrics.current_liabilities = self._parse_amount(values.get('current_liabilities', 0))
                metrics.cash_and_equivalents = self._parse_amount(values.get('cash', 0))
                metrics.inventory = self._parse_amount(values.get('inventory', 0))
                metrics.trade_receivables = self._parse_amount(values.get('receivables', 0))
                metrics.trade_payables = self._parse_amount(values.get('payables', 0))
                metrics.net_fixed_assets = self._parse_amount(values.get('fixed_assets', 0))
    
    def _parse_cash_flow(
        self,
        data: Dict[str, Any],
        metrics_by_year: Dict[str, FinancialMetrics]
    ) -> None:
        """Parse cash flow statement data into metrics."""
        for year, values in data.items():
            if year not in metrics_by_year:
                metrics_by_year[year] = FinancialMetrics(fiscal_year=year)
            
            metrics = metrics_by_year[year]
            
            if isinstance(values, dict):
                metrics.cfo = self._parse_amount(values.get('cfo', 0))
                metrics.cfi = self._parse_amount(values.get('cfi', 0))
                metrics.cff = self._parse_amount(values.get('cff', 0))
                metrics.capex = self._parse_amount(values.get('capex', 0))
                metrics.working_capital_change = self._parse_amount(values.get('wc_change', 0))
    
    def _parse_amount(self, value: Any) -> float:
        """Parse an amount from various formats to float in crores."""
        if value is None:
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            value = value.strip()
            
            # Handle parentheses for negative numbers
            is_negative = value.startswith('(') and value.endswith(')')
            if is_negative:
                value = value[1:-1]
            
            # Handle explicit negative sign
            if value.startswith('-'):
                is_negative = True
                value = value[1:]
            
            # Remove currency symbols and commas
            value = re.sub(r'[₹$,]', '', value)
            
            # Check for crore format
            crore_match = self.CRORE_PATTERN.search(value)
            if crore_match:
                amount = float(crore_match.group(1).replace(',', ''))
                return -amount if is_negative else amount
            
            # Check for lakh format (convert to crores)
            lakh_match = self.LAKH_PATTERN.search(value)
            if lakh_match:
                amount = float(lakh_match.group(1).replace(',', '')) / 100
                return -amount if is_negative else amount
            
            # Try plain number
            try:
                amount = float(value.replace(',', ''))
                return -amount if is_negative else amount
            except ValueError:
                return 0.0
        
        return 0.0
    
    def _calculate_ratios_for_metrics(self, metrics: FinancialMetrics) -> None:
        """Calculate financial ratios for a single year's metrics."""
        # Calculate gross profit if not provided
        if metrics.gross_profit == 0 and metrics.revenue > 0:
            metrics.gross_profit = metrics.revenue - metrics.cost_of_goods_sold
        
        # Calculate EBIT if not provided
        if metrics.ebit == 0 and metrics.ebitda > 0:
            metrics.ebit = metrics.ebitda - metrics.depreciation
        
        # Profitability ratios
        if metrics.revenue > 0:
            metrics.gross_margin = (metrics.gross_profit / metrics.revenue) * 100
            metrics.ebitda_margin = (metrics.ebitda / metrics.revenue) * 100
            metrics.pat_margin = (metrics.pat / metrics.revenue) * 100
            metrics.asset_turnover = metrics.revenue / metrics.total_assets if metrics.total_assets > 0 else None
        
        # Return ratios
        if metrics.total_equity > 0:
            metrics.roe = (metrics.pat / metrics.total_equity) * 100
        
        # ROCE = EBIT / (Total Equity + Total Debt)
        capital_employed = metrics.total_equity + metrics.total_debt
        if capital_employed > 0:
            metrics.roce = (metrics.ebit / capital_employed) * 100
        
        # Leverage ratios
        if metrics.total_equity > 0:
            metrics.debt_equity = metrics.total_debt / metrics.total_equity
        
        # Liquidity ratios
        if metrics.current_liabilities > 0:
            metrics.current_ratio = metrics.current_assets / metrics.current_liabilities
            metrics.quick_ratio = (metrics.current_assets - metrics.inventory) / metrics.current_liabilities
        
        # Interest coverage
        if metrics.interest_expense > 0:
            metrics.interest_coverage = metrics.ebitda / metrics.interest_expense
        
        # Working capital metrics
        if metrics.revenue > 0:
            # Receivable days = (Trade Receivables / Revenue) * 365
            metrics.receivable_days = (metrics.trade_receivables / metrics.revenue) * 365
        
        cogs = metrics.cost_of_goods_sold if metrics.cost_of_goods_sold > 0 else metrics.revenue * 0.7
        if cogs > 0:
            # Inventory days = (Inventory / COGS) * 365
            metrics.inventory_days = (metrics.inventory / cogs) * 365
            # Payable days = (Trade Payables / COGS) * 365
            metrics.payable_days = (metrics.trade_payables / cogs) * 365
        
        # Cash Conversion Cycle = Inventory Days + Receivable Days - Payable Days
        if all([metrics.inventory_days, metrics.receivable_days, metrics.payable_days]):
            metrics.cash_conversion_cycle = (
                metrics.inventory_days + metrics.receivable_days - metrics.payable_days
            )
    
    def calculate_ratios(self, financials: FinancialMetrics) -> Dict[str, float]:
        """
        Calculate key financial ratios for a given metrics object.
        
        Args:
            financials: FinancialMetrics object
            
        Returns:
            Dictionary of ratio names to values
        """
        self._calculate_ratios_for_metrics(financials)
        
        return {
            'roe': financials.roe,
            'roce': financials.roce,
            'debt_equity': financials.debt_equity,
            'current_ratio': financials.current_ratio,
            'quick_ratio': financials.quick_ratio,
            'gross_margin': financials.gross_margin,
            'ebitda_margin': financials.ebitda_margin,
            'pat_margin': financials.pat_margin,
            'asset_turnover': financials.asset_turnover,
            'interest_coverage': financials.interest_coverage,
            'receivable_days': financials.receivable_days,
            'inventory_days': financials.inventory_days,
            'payable_days': financials.payable_days,
            'cash_conversion_cycle': financials.cash_conversion_cycle,
        }
    
    def detect_divergences(
        self,
        metrics_list: List[FinancialMetrics]
    ) -> List[DivergenceWarning]:
        """
        Identify 'Window Dressing' signals in financial data.
        
        Checks for:
        1. Revenue growth vs Receivables growth (Channel Stuffing)
        2. EBITDA vs CFO (Paper profits)
        3. Inventory piling
        
        Args:
            metrics_list: List of FinancialMetrics sorted by fiscal year
            
        Returns:
            List of DivergenceWarning objects
        """
        warnings = []
        
        if len(metrics_list) < 2:
            return warnings
        
        for i in range(1, len(metrics_list)):
            current = metrics_list[i]
            prior = metrics_list[i - 1]
            
            # 1. Channel Stuffing Check
            # If receivables growing faster than revenue by >10pp, flag it
            if prior.revenue > 0 and prior.trade_receivables > 0:
                revenue_growth = ((current.revenue / prior.revenue) - 1) * 100
                receivable_growth = ((current.trade_receivables / prior.trade_receivables) - 1) * 100
                
                growth_gap = receivable_growth - revenue_growth
                
                if growth_gap > 20:
                    warnings.append(DivergenceWarning(
                        fiscal_year=current.fiscal_year,
                        warning_type='channel_stuffing',
                        severity='CRITICAL',
                        description=(
                            f"Receivables growing {receivable_growth:.1f}% vs "
                            f"Revenue {revenue_growth:.1f}% - CHANNEL STUFFING RISK"
                        ),
                        metric_name='receivable_vs_revenue_growth',
                        actual_value=growth_gap,
                        threshold_value=10.0,
                    ))
                elif growth_gap > 10:
                    warnings.append(DivergenceWarning(
                        fiscal_year=current.fiscal_year,
                        warning_type='channel_stuffing',
                        severity='MAJOR',
                        description=(
                            f"Receivables growing {receivable_growth:.1f}% vs "
                            f"Revenue {revenue_growth:.1f}% - Potential channel stuffing"
                        ),
                        metric_name='receivable_vs_revenue_growth',
                        actual_value=growth_gap,
                        threshold_value=10.0,
                    ))
            
            # 2. Paper Profits Check
            # If CFO/EBITDA < 50%, flag as paper profits
            if current.ebitda > 0:
                cfo_to_ebitda = (current.cfo / current.ebitda) * 100
                
                if cfo_to_ebitda < 30:
                    warnings.append(DivergenceWarning(
                        fiscal_year=current.fiscal_year,
                        warning_type='paper_profits',
                        severity='CRITICAL',
                        description=(
                            f"CFO/EBITDA = {cfo_to_ebitda:.1f}% - "
                            "SEVERE PAPER PROFITS RISK - earnings quality very poor"
                        ),
                        metric_name='cfo_to_ebitda',
                        actual_value=cfo_to_ebitda,
                        threshold_value=50.0,
                    ))
                elif cfo_to_ebitda < 50:
                    warnings.append(DivergenceWarning(
                        fiscal_year=current.fiscal_year,
                        warning_type='paper_profits',
                        severity='MAJOR',
                        description=(
                            f"CFO/EBITDA = {cfo_to_ebitda:.1f}% - "
                            "PAPER PROFITS RISK - low cash conversion"
                        ),
                        metric_name='cfo_to_ebitda',
                        actual_value=cfo_to_ebitda,
                        threshold_value=50.0,
                    ))
            
            # 3. Inventory Piling Check
            # If inventory days increased by >15 days YoY, flag it
            if prior.inventory_days and current.inventory_days:
                inventory_change = current.inventory_days - prior.inventory_days
                
                if inventory_change > 30:
                    warnings.append(DivergenceWarning(
                        fiscal_year=current.fiscal_year,
                        warning_type='inventory_piling',
                        severity='CRITICAL',
                        description=(
                            f"Inventory days increased by {inventory_change:.0f} days - "
                            "SEVERE DEMAND CONCERNS"
                        ),
                        metric_name='inventory_days_change',
                        actual_value=inventory_change,
                        threshold_value=15.0,
                    ))
                elif inventory_change > 15:
                    warnings.append(DivergenceWarning(
                        fiscal_year=current.fiscal_year,
                        warning_type='inventory_piling',
                        severity='MAJOR',
                        description=(
                            f"Inventory days increased by {inventory_change:.0f} days - "
                            "Potential demand slowdown"
                        ),
                        metric_name='inventory_days_change',
                        actual_value=inventory_change,
                        threshold_value=15.0,
                    ))
            
            # 4. Receivable Days Worsening
            if prior.receivable_days and current.receivable_days:
                receivable_change = current.receivable_days - prior.receivable_days
                
                if receivable_change > 20:
                    warnings.append(DivergenceWarning(
                        fiscal_year=current.fiscal_year,
                        warning_type='collection_stress',
                        severity='MAJOR',
                        description=(
                            f"Receivable days increased by {receivable_change:.0f} days - "
                            "Collection problems"
                        ),
                        metric_name='receivable_days_change',
                        actual_value=receivable_change,
                        threshold_value=10.0,
                    ))
        
        return warnings
    
    def detect_trends(
        self,
        metrics_list: List[FinancialMetrics]
    ) -> TrendAnalysis:
        """
        Identify growth trends and anomalies in financial data.
        
        Args:
            metrics_list: List of FinancialMetrics sorted by fiscal year
            
        Returns:
            TrendAnalysis object with CAGR, trends, and anomalies
        """
        analysis = TrendAnalysis()
        
        if len(metrics_list) < 2:
            return analysis
        
        first = metrics_list[0]
        last = metrics_list[-1]
        years = len(metrics_list) - 1
        
        # Calculate CAGRs
        if first.revenue > 0 and last.revenue > 0:
            analysis.revenue_cagr = (((last.revenue / first.revenue) ** (1 / years)) - 1) * 100
        
        if first.ebitda > 0 and last.ebitda > 0:
            analysis.ebitda_cagr = (((last.ebitda / first.ebitda) ** (1 / years)) - 1) * 100
        
        if first.pat > 0 and last.pat > 0:
            analysis.pat_cagr = (((last.pat / first.pat) ** (1 / years)) - 1) * 100
        
        # Margin trend
        if first.ebitda_margin is not None and last.ebitda_margin is not None:
            margin_change = last.ebitda_margin - first.ebitda_margin
            analysis.ebitda_margin_change = margin_change
            
            if margin_change > 2:
                analysis.margin_trend = "expanding"
            elif margin_change < -2:
                analysis.margin_trend = "contracting"
            else:
                analysis.margin_trend = "stable"
        
        # ROE trend
        if first.roe is not None and last.roe is not None:
            roe_change = last.roe - first.roe
            analysis.roe_change = roe_change
            
            if roe_change > 3:
                analysis.roe_trend = "improving"
            elif roe_change < -3:
                analysis.roe_trend = "deteriorating"
            else:
                analysis.roe_trend = "stable"
        
        # Debt trend
        if first.debt_equity is not None and last.debt_equity is not None:
            de_change = last.debt_equity - first.debt_equity
            analysis.debt_equity_change = de_change
            
            if de_change > 0.3:
                analysis.debt_trend = "increasing"
            elif de_change < -0.3:
                analysis.debt_trend = "decreasing"
            else:
                analysis.debt_trend = "stable"
        
        # Working capital trend
        if first.cash_conversion_cycle is not None and last.cash_conversion_cycle is not None:
            ccc_change = last.cash_conversion_cycle - first.cash_conversion_cycle
            analysis.ccc_change = ccc_change
            
            if ccc_change > 15:
                analysis.working_capital_trend = "worsening"
                analysis.anomalies.append(f"CCC increased by {ccc_change:.0f} days")
            elif ccc_change < -15:
                analysis.working_capital_trend = "improving"
            else:
                analysis.working_capital_trend = "stable"
        
        # Detect anomalies
        # Sudden revenue spike (>50% YoY)
        for i in range(1, len(metrics_list)):
            current = metrics_list[i]
            prior = metrics_list[i - 1]
            
            if prior.revenue > 0:
                yoy_growth = ((current.revenue / prior.revenue) - 1) * 100
                if yoy_growth > 50:
                    analysis.anomalies.append(
                        f"{current.fiscal_year}: Unusual revenue spike of {yoy_growth:.1f}%"
                    )
            
            # Margin collapse (>5pp drop)
            if prior.ebitda_margin and current.ebitda_margin:
                margin_drop = prior.ebitda_margin - current.ebitda_margin
                if margin_drop > 5:
                    analysis.anomalies.append(
                        f"{current.fiscal_year}: EBITDA margin dropped {margin_drop:.1f}pp"
                    )
        
        return analysis
    
    def calculate_new_age_metrics(
        self,
        financials: FinancialMetrics,
        users: Optional[int] = None,
        gmv: Optional[float] = None,
        variable_costs: Optional[float] = None
    ) -> NewAgeMetrics:
        """
        Calculate metrics for loss-making/startup IPOs.
        
        Args:
            financials: FinancialMetrics for the year
            users: Number of users/customers
            gmv: Gross Merchandise Value (for marketplaces)
            variable_costs: Variable costs for contribution margin
            
        Returns:
            NewAgeMetrics object
        """
        metrics = NewAgeMetrics(fiscal_year=financials.fiscal_year)
        
        # Contribution margin
        if financials.revenue > 0 and variable_costs is not None:
            metrics.contribution_margin = ((financials.revenue - variable_costs) / financials.revenue) * 100
        
        # Gross margin
        metrics.gross_margin = financials.gross_margin
        
        # Burn rate and runway
        if financials.cfo < 0:  # If operating cash flow is negative
            monthly_burn = abs(financials.cfo) / 12
            metrics.burn_rate = monthly_burn
            
            if financials.cash_and_equivalents > 0 and monthly_burn > 0:
                metrics.runway_months = int(financials.cash_and_equivalents / monthly_burn)
        
        # Revenue per user
        if users and users > 0 and financials.revenue > 0:
            metrics.revenue_per_user = (financials.revenue * 10000000) / users  # Convert crores to rupees
        
        # GMV and take rate (for marketplaces)
        if gmv and gmv > 0:
            metrics.gmv = gmv
            if financials.revenue > 0:
                metrics.take_rate = (financials.revenue / gmv) * 100
        
        return metrics
    
    def calculate_cagr(
        self,
        start_value: float,
        end_value: float,
        years: int
    ) -> Optional[float]:
        """
        Calculate Compound Annual Growth Rate.
        
        Args:
            start_value: Starting value
            end_value: Ending value
            years: Number of years
            
        Returns:
            CAGR as percentage, or None if invalid
        """
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return None
        
        return (((end_value / start_value) ** (1 / years)) - 1) * 100
