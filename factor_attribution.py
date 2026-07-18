"""
FACTOR ATTRIBUTION ANALYSIS

Attribute trading returns to the 6 factors in the multi-factor model:
1. Momentum
2. Trend
3. Volatility
4. Volume
5. Market
6. Alpha

This helps understand which factors are contributing to returns.

Usage:
    from factor_attribution import FactorAttributionAnalyzer
    analyzer = FactorAttributionAnalyzer()
    report = analyzer.analyze_factor_contribution('2026-01-01', '2026-01-31')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys

# Add research directory to path
sys.path.insert(0, str(Path(__file__).parent / "research"))

try:
    from factors.factor_definitions import FACTOR_GROUPS, FACTOR_WEIGHTS
except ImportError:
    print("[WARNING] Could not import factor definitions")
    FACTOR_GROUPS = None
    FACTOR_WEIGHTS = None


class FactorAttributionAnalyzer:
    """Analyze factor contribution to returns"""

    def __init__(self):
        """Initialize factor attribution analyzer"""
        self.trading_logs_dir = Path("trading_logs")
        self.signals_dir = Path("research/signals")
        self.results_dir = Path("attribution_results")
        self.results_dir.mkdir(exist_ok=True)

    def analyze_factor_contribution(self,
                                    start_date: str = None,
                                    end_date: str = None) -> Dict:
        """
        Analyze which factors contributed to trading returns

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            dict: Factor attribution results
        """
        print("="*80)
        print("FACTOR ATTRIBUTION ANALYSIS")
        print("="*80)

        if FACTOR_GROUPS is None or FACTOR_WEIGHTS is None:
            print("\n[ERROR] Factor definitions not found")
            return {}

        print("\nAnalyzing contribution from 6 factors:")
        for i, (factor_name, weight) in enumerate(FACTOR_WEIGHTS.items(), 1):
            print(f"  {i}. {factor_name.title():<15} (Weight: {weight*100:.0f}%)")

        print("\n" + "-"*80)
        print("FACTOR ATTRIBUTION METHODOLOGY")
        print("-"*80)
        print("""
Factor attribution shows how much each factor contributed to your returns.

Method:
1. Each factor has a weight in the model (e.g., Momentum: 25%)
2. When a trade is profitable, we attribute returns proportionally to factors
3. Factor contribution = Model Weight × Trade Return

Example:
  Trade: +$100 profit
  Momentum (25% weight) → Contributed +$25
  Market (25% weight) → Contributed +$25
  Trend (20% weight) → Contributed +$20
  ...etc
        """)

        # Load trading data
        trades_df = self._load_trading_data(start_date, end_date)

        if trades_df.empty:
            print("\n[WARNING] No trading data found")
            return {}

        # Calculate factor contributions
        factor_contributions = self._calculate_factor_contributions(trades_df)

        # Display results
        self._display_factor_report(factor_contributions)

        # Save results
        self._save_factor_report(factor_contributions)

        return factor_contributions

    def _load_trading_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load trading data from logs"""
        if not self.trading_logs_dir.exists():
            return pd.DataFrame()

        all_trades = []

        for log_file in self.trading_logs_dir.glob("dynamic_trading_events_*.json"):
            try:
                with open(log_file, 'r') as f:
                    events = json.load(f)

                for event in events:
                    all_trades.append({
                        'symbol': event.get('symbol', ''),
                        'action': event.get('action', ''),
                        'pl_amount': event.get('pl_amount', 0),
                        'pl_pct': event.get('pl_pct', 0),
                        'timestamp': event.get('timestamp', '')
                    })

            except Exception as e:
                continue

        if not all_trades:
            return pd.DataFrame()

        df = pd.DataFrame(all_trades)
        df['pl_amount'] = pd.to_numeric(df['pl_amount'], errors='coerce').fillna(0)

        return df

    def _calculate_factor_contributions(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate how much each factor contributed to returns"""
        total_pnl = trades_df['pl_amount'].sum()

        # Attribute returns proportionally to factor weights
        factor_contributions = {}

        for factor_name, weight in FACTOR_WEIGHTS.items():
            contribution = total_pnl * weight

            # Count winning and losing trades
            winning_contribution = trades_df[trades_df['pl_amount'] > 0]['pl_amount'].sum() * weight
            losing_contribution = trades_df[trades_df['pl_amount'] < 0]['pl_amount'].sum() * weight

            factor_contributions[factor_name] = {
                'weight': weight,
                'total_contribution': contribution,
                'winning_contribution': winning_contribution,
                'losing_contribution': losing_contribution,
                'net_contribution': contribution,
                'contribution_pct': (contribution / total_pnl * 100) if total_pnl != 0 else 0
            }

        # Add summary
        factor_contributions['summary'] = {
            'total_pnl': total_pnl,
            'total_trades': len(trades_df),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return factor_contributions

    def _display_factor_report(self, contributions: Dict):
        """Display factor attribution report"""
        print("\n" + "="*80)
        print("FACTOR CONTRIBUTION REPORT")
        print("="*80)

        summary = contributions.get('summary', {})
        total_pnl = summary.get('total_pnl', 0)

        print(f"\nTotal P&L: ${total_pnl:+,.2f}")
        print(f"Total Trades: {summary.get('total_trades', 0)}")

        print("\n" + "-"*80)
        print("FACTOR-BY-FACTOR BREAKDOWN")
        print("-"*80)
        print(f"{'Factor':<15} {'Weight':<10} {'Contribution':<15} {'% of Total':<12} {'Win/Loss'}")
        print("-"*80)

        # Sort by contribution
        factors = [(name, data) for name, data in contributions.items() if name != 'summary']
        factors.sort(key=lambda x: x[1]['total_contribution'], reverse=True)

        for factor_name, data in factors:
            win_contrib = data['winning_contribution']
            loss_contrib = data['losing_contribution']

            print(f"{factor_name.title():<15} "
                  f"{data['weight']*100:>6.0f}%    "
                  f"${data['total_contribution']:>12,.2f}  "
                  f"{data['contribution_pct']:>9.1f}%   "
                  f"${win_contrib:+.2f} / ${loss_contrib:+.2f}")

        print("-"*80)
        print(f"{'TOTAL':<15} {'100%':<10} ${total_pnl:>12,.2f}")

        print("\n" + "-"*80)
        print("INSIGHTS")
        print("-"*80)

        # Find best and worst contributing factors
        best_factor = max(factors, key=lambda x: x[1]['total_contribution'])
        worst_factor = min(factors, key=lambda x: x[1]['total_contribution'])

        print(f"\nBest Performing Factor: {best_factor[0].title()}")
        print(f"  Contributed: ${best_factor[1]['total_contribution']:+,.2f}")
        print(f"  Model Weight: {best_factor[1]['weight']*100:.0f}%")

        print(f"\nWorst Performing Factor: {worst_factor[0].title()}")
        print(f"  Contributed: ${worst_factor[1]['total_contribution']:+,.2f}")
        print(f"  Model Weight: {worst_factor[1]['weight']*100:.0f}%")

        # Recommendations
        print("\n" + "-"*80)
        print("RECOMMENDATIONS")
        print("-"*80)

        if best_factor[1]['contribution_pct'] > 50:
            print(f"\n[INSIGHT] {best_factor[0].title()} is dominating returns")
            print(f"  → Consider increasing its weight above current {best_factor[1]['weight']*100:.0f}%")

        if worst_factor[1]['total_contribution'] < 0:
            print(f"\n[WARNING] {worst_factor[0].title()} is losing money")
            print(f"  → Consider reducing its weight below current {worst_factor[1]['weight']*100:.0f}%")
            print(f"  → Or investigate why this factor is underperforming")

        print("\n" + "="*80)

    def _save_factor_report(self, contributions: Dict):
        """Save factor attribution report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"factor_attribution_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(contributions, f, indent=2)

        print(f"\nReport saved: {report_file}")


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    analyzer = FactorAttributionAnalyzer()

    print("\nRunning factor attribution analysis...\n")

    report = analyzer.analyze_factor_contribution()

    print("\n" + "="*80)
    print("FACTOR ATTRIBUTION COMPLETE")
    print("="*80)
    print("\nThis analysis shows how much each of the 6 factors contributed to your returns.")
    print("Use this to optimize factor weights in research/factors/factor_definitions.py")
