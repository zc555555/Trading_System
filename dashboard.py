"""
REAL-TIME TRADING DASHBOARD

Interactive web dashboard for monitoring trading activities:
- Real-time portfolio status
- Live P&L tracking
- Position details with charts
- News sentiment display
- Risk metrics monitoring
- Trade history

Built with Streamlit for easy deployment and interaction.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
try:
    from alpaca.trading.client import TradingClient
    from config_alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY
    from news_sentiment_analyzer import NewsSentimentAnalyzer
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()


# ============================================================================
# Helper to get Alpaca client
# ============================================================================

def get_alpaca_client():
    """Get Alpaca trading client"""
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_account_info():
    """Get current account information"""
    try:
        client = get_alpaca_client()
        account = client.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'last_equity': float(account.last_equity),
            'pnl': float(account.equity) - float(account.last_equity),
            'pnl_pct': (float(account.equity) - float(account.last_equity)) / float(account.last_equity) * 100
        }
    except Exception as e:
        st.error(f"Failed to get account info: {str(e)}")
        return None


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_positions():
    """Get current positions"""
    try:
        client = get_alpaca_client()
        positions = client.get_all_positions()

        if not positions:
            return pd.DataFrame()

        data = []
        for pos in positions:
            data.append({
                'Symbol': pos.symbol,
                'Qty': int(pos.qty),
                'Entry Price': float(pos.avg_entry_price),
                'Current Price': float(pos.current_price),
                'Market Value': float(pos.market_value),
                'P&L': float(pos.unrealized_pl),
                'P&L %': float(pos.unrealized_plpc) * 100,
                'Change Today %': float(pos.change_today) * 100
            })

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Failed to get positions: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_news_sentiment(symbols):
    """Get news sentiment for symbols"""
    if not symbols:
        return {}

    try:
        analyzer = NewsSentimentAnalyzer()
        return analyzer.get_batch_sentiment(symbols)
    except Exception as e:
        st.warning(f"Failed to get news sentiment: {str(e)}")
        return {}


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_recent_trades():
    """Get recent trade history"""
    try:
        # Try to read from trading logs
        log_dir = Path("trading_logs")
        if not log_dir.exists():
            return pd.DataFrame()

        # Get most recent log file
        log_files = sorted(log_dir.glob("*.json"), reverse=True)
        if not log_files:
            return pd.DataFrame()

        trades = []
        for log_file in log_files[:5]:  # Last 5 days
            with open(log_file, 'r') as f:
                data = json.load(f)
                if 'trades' in data:
                    for trade in data['trades']:
                        trades.append({
                            'Date': data.get('date', 'Unknown'),
                            'Symbol': trade.get('symbol', ''),
                            'Action': trade.get('side', ''),
                            'Qty': trade.get('qty', 0),
                            'Price': trade.get('filled_avg_price', 0),
                            'Value': trade.get('qty', 0) * trade.get('filled_avg_price', 0)
                        })

        return pd.DataFrame(trades) if trades else pd.DataFrame()

    except Exception as e:
        st.warning(f"Failed to load trade history: {str(e)}")
        return pd.DataFrame()


def create_pnl_gauge(pnl_pct):
    """Create P&L gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pnl_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Today's P&L %"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-5, 5]},
            'bar': {'color': "darkgreen" if pnl_pct >= 0 else "darkred"},
            'steps': [
                {'range': [-5, -2.5], 'color': "lightcoral"},
                {'range': [-2.5, 0], 'color': "lightyellow"},
                {'range': [0, 2.5], 'color': "lightgreen"},
                {'range': [2.5, 5], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': -2.5
            }
        }
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_position_chart(positions_df):
    """Create position allocation pie chart"""
    if positions_df.empty:
        return None

    fig = px.pie(
        positions_df,
        values='Market Value',
        names='Symbol',
        title='Position Allocation',
        hole=0.4
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))

    return fig


def create_pnl_bar_chart(positions_df):
    """Create P&L bar chart"""
    if positions_df.empty:
        return None

    positions_df = positions_df.sort_values('P&L')

    colors = ['red' if x < 0 else 'green' for x in positions_df['P&L']]

    fig = go.Figure(data=[
        go.Bar(
            x=positions_df['Symbol'],
            y=positions_df['P&L'],
            marker_color=colors,
            text=positions_df['P&L'].apply(lambda x: f"${x:.2f}"),
            textposition='auto'
        )
    ])

    fig.update_layout(
        title='Position P&L',
        xaxis_title='Symbol',
        yaxis_title='P&L ($)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    # Title and header
    st.title("📊 Real-Time Trading Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 60)

        st.markdown("---")

        st.header("🔗 Quick Links")
        st.markdown("- [Alpaca Paper Trading](https://app.alpaca.markets/paper/dashboard/overview)")
        st.markdown("- [View Logs](file:///C:/Users/13785/OneDrive/Desktop/stock_predict/alerts)")

        st.markdown("---")

        if st.button("🔄 Force Refresh"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get data
    account = get_account_info()
    positions_df = get_positions()

    if account is None:
        st.error("❌ Failed to connect to Alpaca API. Please check your credentials.")
        st.stop()
        return

    # ========================================================================
    # Row 1: Account Summary
    # ========================================================================

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="💰 Portfolio Value",
            value=f"${account['equity']:,.2f}",
            delta=f"${account['pnl']:+,.2f}"
        )

    with col2:
        st.metric(
            label="📈 Today's P&L",
            value=f"{account['pnl_pct']:+.2f}%",
            delta=f"${account['pnl']:+,.2f}"
        )

    with col3:
        st.metric(
            label="💵 Cash",
            value=f"${account['cash']:,.2f}"
        )

    with col4:
        st.metric(
            label="🔋 Buying Power",
            value=f"${account['buying_power']:,.2f}"
        )

    with col5:
        position_count = len(positions_df) if not positions_df.empty else 0
        st.metric(
            label="📊 Positions",
            value=position_count
        )

    st.markdown("---")

    # ========================================================================
    # Row 2: Charts
    # ========================================================================

    col1, col2 = st.columns(2)

    with col1:
        # P&L Gauge
        pnl_gauge = create_pnl_gauge(account['pnl_pct'])
        st.plotly_chart(pnl_gauge, use_container_width=True)

    with col2:
        # Position Allocation
        if not positions_df.empty:
            pie_chart = create_position_chart(positions_df)
            st.plotly_chart(pie_chart, use_container_width=True)
        else:
            st.info("📭 No active positions")

    # ========================================================================
    # Row 3: Position P&L Bar Chart
    # ========================================================================

    if not positions_df.empty:
        bar_chart = create_pnl_bar_chart(positions_df)
        st.plotly_chart(bar_chart, use_container_width=True)

    st.markdown("---")

    # ========================================================================
    # Row 4: Detailed Positions Table
    # ========================================================================

    st.subheader("📋 Current Positions")

    if not positions_df.empty:
        # Color code P&L column
        def color_pnl(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
            return f'color: {color}; font-weight: bold'

        styled_df = positions_df.style.applymap(color_pnl, subset=['P&L', 'P&L %'])

        st.dataframe(styled_df, use_container_width=True)

        # Get news sentiment for current positions
        symbols = positions_df['Symbol'].tolist()
        sentiment_data = get_news_sentiment(symbols)

        if sentiment_data:
            st.subheader("📰 News Sentiment")

            for symbol in symbols:
                if symbol in sentiment_data:
                    info = sentiment_data[symbol]
                    sentiment = info['sentiment_score']

                    # Sentiment emoji
                    if sentiment > 0.2:
                        emoji = "🟢"
                        color = "green"
                    elif sentiment < -0.2:
                        emoji = "🔴"
                        color = "red"
                    else:
                        emoji = "🟡"
                        color = "orange"

                    st.markdown(
                        f"{emoji} **{symbol}**: Sentiment {sentiment:+.2f} "
                        f"({info['news_count']} news items)"
                    )

    else:
        st.info("📭 No active positions")

    st.markdown("---")

    # ========================================================================
    # Row 5: Recent Trades
    # ========================================================================

    st.subheader("📜 Recent Trades")

    trades_df = get_recent_trades()

    if not trades_df.empty:
        st.dataframe(trades_df.tail(10), use_container_width=True)
    else:
        st.info("No recent trades found")

    st.markdown("---")

    # ========================================================================
    # Row 6: Risk Metrics
    # ========================================================================

    st.subheader("⚠️ Risk Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Stop-loss distance
        if not positions_df.empty:
            min_distance = positions_df['P&L %'].min()
            stop_loss_threshold = -2.5

            distance_to_stop = min_distance - stop_loss_threshold

            st.metric(
                label="Closest to Stop-Loss",
                value=f"{min_distance:.2f}%",
                delta=f"{distance_to_stop:.2f}% from -2.5% threshold",
                delta_color="inverse"
            )
        else:
            st.info("No positions")

    with col2:
        # Max loss check
        max_loss_threshold = -3.0
        current_pnl = account['pnl_pct']

        distance_to_max_loss = current_pnl - max_loss_threshold

        st.metric(
            label="Distance to Max Loss",
            value=f"{distance_to_max_loss:.2f}%",
            delta="Threshold: -3.0%",
            delta_color="inverse"
        )

    with col3:
        # Portfolio concentration
        if not positions_df.empty:
            max_position_pct = (positions_df['Market Value'].max() / account['equity']) * 100

            st.metric(
                label="Largest Position",
                value=f"{max_position_pct:.1f}%",
                delta="Max recommended: 25%",
                delta_color="inverse" if max_position_pct > 25 else "normal"
            )
        else:
            st.info("No positions")

    # ========================================================================
    # Auto Refresh (at the end after all content is displayed)
    # ========================================================================
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()


# ============================================================================
# Run Dashboard
# ============================================================================

if __name__ == "__main__":
    main()
