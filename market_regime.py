"""
MARKET REGIME DETECTION

Identify current market conditions and recommend strategy adjustments:
- Bull market: Strong uptrend, low volatility
- Bear market: Downtrend, risk-off sentiment
- High volatility: Choppy, uncertain
- Low volatility: Stable, range-bound

Based on:
- SPY trend (S&P 500)
- VIX level (volatility index)
- Market breadth
- Sentiment indicators

Recommendations adapt trading parameters to market conditions.

Usage:
    from market_regime import MarketRegimeDetector
    detector = MarketRegimeDetector()
    regime = detector.detect_current_regime()
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple


class MarketRegimeDetector:
    """Detect current market regime and provide recommendations"""

    # Market regime types
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NEUTRAL = "NEUTRAL"

    # VIX thresholds
    VIX_LOW = 15
    VIX_NORMAL = 20
    VIX_HIGH = 30
    VIX_EXTREME = 40

    def __init__(self):
        """Initialize market regime detector"""
        print("Initializing Market Regime Detector...")

    def detect_current_regime(self) -> Dict:
        """
        Detect current market regime

        Returns:
            dict: Regime info and recommendations
        """
        print("\n" + "="*80)
        print("MARKET REGIME ANALYSIS")
        print("="*80)

        # Get market data
        spy_data = self._get_spy_data()
        vix_data = self._get_vix_data()

        if spy_data is None or vix_data is None:
            print("[ERROR] Failed to fetch market data")
            return self._get_default_regime()

        # Analyze SPY trend
        spy_trend = self._analyze_spy_trend(spy_data)

        # Analyze VIX level
        vix_regime = self._analyze_vix(vix_data)

        # Determine overall regime
        regime = self._determine_regime(spy_trend, vix_regime)

        # Get recommendations
        recommendations = self._get_recommendations(regime)

        # Display results
        self._display_regime_info(regime, spy_trend, vix_regime, recommendations)

        return {
            'regime': regime,
            'spy_trend': spy_trend,
            'vix_regime': vix_regime,
            'recommendations': recommendations,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _get_spy_data(self, days: int = 60) -> pd.DataFrame:
        """Fetch SPY (S&P 500 ETF) data"""
        try:
            print("\nFetching SPY data...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

            if spy.empty:
                return None

            # Handle MultiIndex columns (yfinance sometimes returns MultiIndex)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.droplevel(1)

            print(f"  [OK] Got {len(spy)} days of SPY data")
            return spy

        except Exception as e:
            print(f"  [ERROR] Failed to fetch SPY: {str(e)}")
            return None

    def _get_vix_data(self, days: int = 30) -> pd.DataFrame:
        """Fetch VIX (Volatility Index) data"""
        try:
            print("Fetching VIX data...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

            if vix.empty:
                return None

            # Handle MultiIndex columns
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.droplevel(1)

            print(f"  [OK] Got {len(vix)} days of VIX data")
            return vix

        except Exception as e:
            print(f"  [ERROR] Failed to fetch VIX: {str(e)}")
            return None

    def _analyze_spy_trend(self, spy_data: pd.DataFrame) -> Dict:
        """Analyze SPY trend"""
        # Convert to scalar values explicitly
        current_price = float(spy_data['Close'].iloc[-1])

        # Calculate moving averages
        ma_20 = float(spy_data['Close'].rolling(20).mean().iloc[-1])
        ma_50 = float(spy_data['Close'].rolling(50).mean().iloc[-1])

        # Calculate returns
        return_5d = float((current_price / spy_data['Close'].iloc[-6] - 1) * 100)
        return_20d = float((current_price / spy_data['Close'].iloc[-21] - 1) * 100)

        # Determine trend
        if current_price > ma_20 > ma_50 and return_20d > 5:
            trend = "STRONG_UP"
        elif current_price > ma_20 and return_20d > 0:
            trend = "UP"
        elif current_price < ma_20 < ma_50 and return_20d < -5:
            trend = "STRONG_DOWN"
        elif current_price < ma_20 and return_20d < 0:
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"

        return {
            'trend': trend,
            'current_price': current_price,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'return_5d': return_5d,
            'return_20d': return_20d
        }

    def _analyze_vix(self, vix_data: pd.DataFrame) -> Dict:
        """Analyze VIX level"""
        # Convert to scalar values explicitly
        current_vix = float(vix_data['Close'].iloc[-1])

        # VIX moving average
        vix_ma_20 = float(vix_data['Close'].rolling(20).mean().iloc[-1])

        # Determine regime
        if current_vix < self.VIX_LOW:
            regime = "VERY_LOW"
            description = "Complacent market"
        elif current_vix < self.VIX_NORMAL:
            regime = "LOW"
            description = "Normal conditions"
        elif current_vix < self.VIX_HIGH:
            regime = "ELEVATED"
            description = "Increased uncertainty"
        elif current_vix < self.VIX_EXTREME:
            regime = "HIGH"
            description = "High volatility"
        else:
            regime = "EXTREME"
            description = "Market panic"

        return {
            'regime': regime,
            'current_vix': current_vix,
            'vix_ma_20': vix_ma_20,
            'description': description
        }

    def _determine_regime(self, spy_trend: Dict, vix_regime: Dict) -> str:
        """Determine overall market regime"""
        trend = spy_trend['trend']
        vix = vix_regime['regime']

        # Bull market: Strong uptrend + low volatility
        if trend in ["STRONG_UP", "UP"] and vix in ["VERY_LOW", "LOW"]:
            return self.BULL_MARKET

        # Bear market: Downtrend + elevated volatility
        elif trend in ["STRONG_DOWN", "DOWN"] and vix in ["ELEVATED", "HIGH", "EXTREME"]:
            return self.BEAR_MARKET

        # High volatility: Any trend + high VIX
        elif vix in ["HIGH", "EXTREME"]:
            return self.HIGH_VOLATILITY

        # Low volatility: Sideways + low VIX
        elif trend == "SIDEWAYS" and vix in ["VERY_LOW", "LOW"]:
            return self.LOW_VOLATILITY

        # Neutral: Everything else
        else:
            return self.NEUTRAL

    def _get_recommendations(self, regime: str) -> Dict:
        """Get trading parameter recommendations based on regime"""
        if regime == self.BULL_MARKET:
            return {
                'regime_name': 'Bull Market',
                'description': 'Strong uptrend with low volatility - Favorable conditions',
                'stop_loss': 0.030,  # 3.0% (looser)
                'take_profit': 0.035,  # 3.5% (higher target)
                'max_daily_loss': 0.040,  # 4.0% (allow more drawdown)
                'position_size': 0.25,  # 25% (larger positions)
                'risk_level': 'MODERATE',
                'strategy': 'Aggressive - Ride the trend'
            }

        elif regime == self.BEAR_MARKET:
            return {
                'regime_name': 'Bear Market',
                'description': 'Downtrend with high volatility - High risk',
                'stop_loss': 0.015,  # 1.5% (tighter)
                'take_profit': 0.020,  # 2.0% (lower target)
                'max_daily_loss': 0.020,  # 2.0% (strict)
                'position_size': 0.10,  # 10% (smaller positions)
                'risk_level': 'HIGH',
                'strategy': 'Defensive - Quick profits, tight stops'
            }

        elif regime == self.HIGH_VOLATILITY:
            return {
                'regime_name': 'High Volatility',
                'description': 'Choppy market - Unpredictable',
                'stop_loss': 0.020,  # 2.0% (moderate)
                'take_profit': 0.025,  # 2.5% (standard)
                'max_daily_loss': 0.025,  # 2.5% (moderate)
                'position_size': 0.15,  # 15% (moderate)
                'risk_level': 'HIGH',
                'strategy': 'Cautious - Reduce exposure'
            }

        elif regime == self.LOW_VOLATILITY:
            return {
                'regime_name': 'Low Volatility',
                'description': 'Range-bound market - Limited opportunities',
                'stop_loss': 0.025,  # 2.5% (standard)
                'take_profit': 0.030,  # 3.0% (slightly higher)
                'max_daily_loss': 0.030,  # 3.0% (standard)
                'position_size': 0.20,  # 20% (moderate-high)
                'risk_level': 'LOW',
                'strategy': 'Patient - Wait for clear setups'
            }

        else:  # NEUTRAL
            return {
                'regime_name': 'Neutral Market',
                'description': 'Mixed conditions - Standard approach',
                'stop_loss': 0.025,  # 2.5% (standard)
                'take_profit': 0.025,  # 2.5% (standard)
                'max_daily_loss': 0.030,  # 3.0% (standard)
                'position_size': 0.15,  # 15% (standard)
                'risk_level': 'MODERATE',
                'strategy': 'Balanced - Use current parameters'
            }

    def _get_default_regime(self) -> Dict:
        """Return default regime when data unavailable"""
        return {
            'regime': self.NEUTRAL,
            'recommendations': self._get_recommendations(self.NEUTRAL),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'note': 'Using default regime due to data unavailability'
        }

    def _display_regime_info(self, regime: str, spy_trend: Dict,
                             vix_regime: Dict, recommendations: Dict):
        """Display regime analysis results"""
        print("\n" + "-"*80)
        print("SPY (S&P 500) ANALYSIS")
        print("-"*80)
        print(f"Current Price: ${spy_trend['current_price']:.2f}")
        print(f"MA(20): ${spy_trend['ma_20']:.2f}")
        print(f"MA(50): ${spy_trend['ma_50']:.2f}")
        print(f"5-day Return: {spy_trend['return_5d']:+.2f}%")
        print(f"20-day Return: {spy_trend['return_20d']:+.2f}%")
        print(f"Trend: {spy_trend['trend']}")

        print("\n" + "-"*80)
        print("VIX (VOLATILITY INDEX) ANALYSIS")
        print("-"*80)
        print(f"Current VIX: {vix_regime['current_vix']:.2f}")
        print(f"VIX MA(20): {vix_regime['vix_ma_20']:.2f}")
        print(f"Regime: {vix_regime['regime']}")
        print(f"Description: {vix_regime['description']}")

        print("\n" + "="*80)
        print(f"MARKET REGIME: {recommendations['regime_name'].upper()}")
        print("="*80)
        print(f"\n{recommendations['description']}")
        print(f"\nRisk Level: {recommendations['risk_level']}")
        print(f"Strategy: {recommendations['strategy']}")

        print("\n" + "-"*80)
        print("RECOMMENDED PARAMETERS")
        print("-"*80)
        print(f"Stop-Loss: {recommendations['stop_loss']*100:.1f}%")
        print(f"Take-Profit: {recommendations['take_profit']*100:.1f}%")
        print(f"Max Daily Loss: {recommendations['max_daily_loss']*100:.1f}%")
        print(f"Position Size: {recommendations['position_size']*100:.0f}%")

        print("\n" + "="*80)
        print("NOTES")
        print("="*80)
        print("- These are recommendations based on current market conditions")
        print("- Current system uses: Stop-loss=2.5%, Take-profit=2.5%, Max loss=3.0%")
        print("- Consider updating parameters in monitor_dynamic_trading.py")
        print("- Always backtest parameter changes before live trading")


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    detector = MarketRegimeDetector()
    result = detector.detect_current_regime()

    print("\n" + "="*80)
    print("REGIME DETECTION COMPLETE")
    print("="*80)

    if result:
        print(f"\nDetected Regime: {result['regime']}")
        print(f"Timestamp: {result['timestamp']}")
