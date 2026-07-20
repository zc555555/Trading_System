"""
Generate Web Page with Daily Signals

Automatically creates/updates signals.html with latest predictions
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def generate_web_page():
    """Generate HTML page with daily signals"""

    # Paths
    artifacts_dir = Path(__file__).parent / "artifacts"
    data_dir = Path(__file__).parent / "data"
    web_dir = Path(__file__).parent.parent / "web"
    web_dir.mkdir(exist_ok=True)

    # Load model
    with open(artifacts_dir / "ensemble_time_windows.pkl", 'rb') as f:
        ensemble_dict = pickle.load(f)

    models = ensemble_dict['models']
    weights = ensemble_dict['weights']
    feature_cols = ensemble_dict['feature_cols']

    # Load latest data
    df = pd.read_parquet(data_dir / "stocks_with_time_windows.parquet")
    latest_date = df['date'].max()
    latest_data = df[df['date'] == latest_date].copy()

    # Predict
    X = latest_data[feature_cols].values
    X = pd.DataFrame(X, columns=feature_cols).ffill().fillna(0).values

    predictions = {}
    for model_name in sorted(models.keys()):
        pred = models[model_name].predict(X)
        predictions[model_name] = pred

    pred_matrix = np.column_stack([predictions[m] for m in sorted(predictions.keys())])
    weights_array = np.array([weights[m] for m in sorted(predictions.keys())])
    ensemble_pred = pred_matrix @ weights_array

    latest_data['prediction'] = ensemble_pred
    latest_data['confidence'] = np.abs(ensemble_pred)

    # Select stocks
    config = {'n_top': 10, 'min_confidence': 0.005, 'min_stocks': 7}

    sorted_data = latest_data.sort_values('confidence', ascending=False)
    top_n = sorted_data.head(config['n_top'])
    selected = top_n[top_n['confidence'] >= config['min_confidence']]

    should_trade = len(selected) >= config['min_stocks']

    # Calculate historical accuracy for each stock
    stock_accuracy = {}
    for symbol in selected['symbol'].unique():
        stock_df = df[df['symbol'] == symbol].copy()

        # Sort by date
        stock_df = stock_df.sort_values('date')

        # Calculate actual returns (shift to get next day's return)
        stock_df['actual_return'] = stock_df['close'].pct_change().shift(-1)

        # Make predictions for this stock's historical data
        X_stock = stock_df[feature_cols].values
        X_stock = pd.DataFrame(X_stock, columns=feature_cols).ffill().fillna(0).values

        pred_stock = np.zeros(len(X_stock))
        for model_name in sorted(models.keys()):
            pred_stock += models[model_name].predict(X_stock) * weights[model_name]

        stock_df['pred'] = pred_stock

        # Remove rows with NaN actual returns
        stock_df = stock_df.dropna(subset=['actual_return'])

        # Calculate accuracy (correct direction prediction)
        stock_df['correct'] = ((stock_df['pred'] > 0) & (stock_df['actual_return'] > 0)) | \
                              ((stock_df['pred'] < 0) & (stock_df['actual_return'] < 0))

        accuracy = stock_df['correct'].mean() * 100
        stock_accuracy[symbol] = accuracy

    # Prepare data for HTML with hybrid allocation
    stocks_data = []
    if should_trade:
        # Trading 212 FX Fee: 0.15% buy + 0.15% sell = 0.30% total round-trip cost
        FX_FEE = 0.003  # 0.30% in decimal

        # Calculate expected return after fees for each stock
        expected_returns = []
        for _, row in selected.iterrows():
            symbol = row['symbol']
            prediction = row['prediction']
            accuracy = stock_accuracy.get(symbol, 64.33) / 100  # Convert to decimal

            # Expected return = prediction × accuracy - transaction cost
            # This accounts for both prediction magnitude and historical reliability
            expected_return = prediction * accuracy - FX_FEE

            expected_returns.append({
                'symbol': symbol,
                'prediction': prediction,
                'accuracy': stock_accuracy.get(symbol, 64.33),
                'expected_return': expected_return,
                'current_price': row.get('close', 0)
            })

        # Filter: only allocate to stocks with positive expected return (profitable after fees)
        profitable_stocks = [s for s in expected_returns if s['expected_return'] > 0]

        if len(profitable_stocks) == 0:
            # No profitable trades after accounting for fees
            should_trade = False
            stocks_data = []
        else:
            # Allocate proportionally to expected return
            # Stocks with higher expected return get more capital
            total_expected = sum(s['expected_return'] for s in profitable_stocks)

            for stock in profitable_stocks:
                position_pct = (stock['expected_return'] / total_expected) * 100
                stocks_data.append({
                    'symbol': stock['symbol'],
                    'prediction': float(stock['prediction']),
                    'accuracy': float(stock['accuracy']),
                    'position_pct': float(position_pct),
                    'current_price': float(stock['current_price']),
                    'expected_return': float(stock['expected_return'])
                })

            # Sort by expected return descending (best opportunities first)
            stocks_data.sort(key=lambda x: x['expected_return'], reverse=True)

            # Re-rank after sorting
            for i, stock in enumerate(stocks_data, 1):
                stock['rank'] = i

    # Convert stocks data to JSON for JavaScript
    stocks_json = json.dumps(stocks_data, indent=4)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plan A - 每日交易信号</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }}

        .header h1 {{
            color: #333;
            font-size: 28px;
            margin-bottom: 8px;
        }}

        .status-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin-left: 12px;
        }}

        .status-badge.trade {{
            background: #10b981;
            color: white;
        }}

        .status-badge.skip {{
            background: #ef4444;
            color: white;
        }}

        .meta {{
            color: #666;
            font-size: 14px;
            margin-top: 10px;
        }}

        .capital-input {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        }}

        .capital-input h3 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 18px;
        }}

        .input-group {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .input-group label {{
            color: #666;
            font-weight: 500;
        }}

        .input-group input {{
            flex: 1;
            max-width: 200px;
            padding: 10px 15px;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }}

        .input-group input:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .signals-card {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            margin-bottom: 20px;
        }}

        .section-title {{
            color: #333;
            font-size: 20px;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #f0f0f0;
        }}

        .section-subtitle {{
            color: #10b981;
            font-size: 16px;
            font-weight: 600;
            margin: 20px 0 10px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}

        th {{
            background: #f8f9fa;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            color: #666;
            font-size: 13px;
            border-bottom: 2px solid #e9ecef;
        }}

        td {{
            padding: 14px 8px;
            border-bottom: 1px solid #f0f0f0;
            font-size: 14px;
        }}

        .rank {{
            font-weight: bold;
            color: #667eea;
            font-size: 15px;
        }}

        .symbol {{
            font-weight: bold;
            font-size: 16px;
            color: #333;
        }}

        .action-buy {{
            display: inline-block;
            padding: 4px 12px;
            background: #d1fae5;
            color: #065f46;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
        }}

        .action-sell {{
            display: inline-block;
            padding: 4px 12px;
            background: #fee2e2;
            color: #991b1b;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
        }}

        .positive {{
            color: #10b981;
            font-weight: 500;
        }}

        .negative {{
            color: #ef4444;
            font-weight: 500;
        }}

        .amount {{
            font-weight: 600;
            color: #667eea;
        }}

        .top7-badge {{
            display: inline-block;
            background: #fbbf24;
            color: #78350f;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }}

        .footer {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            color: #666;
            font-size: 13px;
            line-height: 1.6;
        }}

        .footer strong {{
            color: #333;
        }}

        .clickable-symbol {{
            cursor: pointer;
            transition: color 0.2s;
        }}

        .clickable-symbol:hover {{
            color: #667eea;
            text-decoration: underline;
        }}

        .price-note {{
            font-size: 12px;
            color: #ef4444;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                📈 Plan A - 每日交易信号
                <span class="status-badge {'skip' if not should_trade else 'trade'}">
                    {'⏸️ SKIP' if not should_trade else '✅ TRADE'}
                </span>
            </h1>
            <div class="meta">
                更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据日期: {latest_date}
                <br><span class="price-note">⚠️ 价格为历史收盘价，交易前请在Trading 212查看实时价格</span>
            </div>
        </div>

        <div class="capital-input">
            <h3>💰 输入本金</h3>
            <div class="input-group">
                <label>本金金额:</label>
                <input type="number" id="capitalInput" value="100" min="1" step="1">
                <label>英镑 (£)</label>
            </div>
        </div>

        <div class="signals-card">
            <div class="section-title">今日推荐股票</div>

            {'<p style="color: #ef4444; font-weight: 600;">今天不适合交易，持有现金观望。</p>' if not should_trade else ''}

            <div class="section-subtitle">⭐ TOP 7 核心推荐</div>
            <table id="top7Table">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>股票代码 (点击查看图表)</th>
                        <th>参考价格</th>
                        <th>操作</th>
                        <th>预测涨跌</th>
                        <th>历史准确率</th>
                        <th>预期收益</th>
                        <th>仓位%</th>
                        <th>买入金额</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Populated by JavaScript -->
                </tbody>
            </table>

            <div class="section-subtitle">📊 其他推荐股票</div>
            <table id="othersTable">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>股票代码 (点击查看图表)</th>
                        <th>参考价格</th>
                        <th>操作</th>
                        <th>预测涨跌</th>
                        <th>历史准确率</th>
                        <th>预期收益</th>
                        <th>仓位%</th>
                        <th>买入金额</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- Populated by JavaScript -->
                </tbody>
            </table>
        </div>

        <div class="signals-card" id="profitSummary" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <div class="section-title" style="color: white; border-bottom-color: rgba(255,255,255,0.3);">
                💰 预期收益分析（扣除Trading 212手续费后）
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 15px;">
                <div>
                    <div style="font-size: 13px; opacity: 0.9; margin-bottom: 5px;">本金</div>
                    <div style="font-size: 24px; font-weight: bold;" id="capitalDisplay">£100.00</div>
                </div>
                <div>
                    <div style="font-size: 13px; opacity: 0.9; margin-bottom: 5px;">预期净收益</div>
                    <div style="font-size: 24px; font-weight: bold;" id="expectedProfit">£0.00</div>
                    <div style="font-size: 12px; opacity: 0.8;" id="expectedProfitPct">(0.00%)</div>
                </div>
                <div>
                    <div style="font-size: 13px; opacity: 0.9; margin-bottom: 5px;">Trading 212手续费</div>
                    <div style="font-size: 18px; font-weight: bold;" id="totalFees">£0.00</div>
                    <div style="font-size: 11px; opacity: 0.8;">FX费: 0.15% 买入 + 0.15% 卖出</div>
                </div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px; font-size: 13px; line-height: 1.6;">
                <strong>💡 费用说明:</strong><br>
                • Trading 212对美股交易收取0.15%外汇转换费（买入+卖出共0.30%）<br>
                • 无佣金，无平台费<br>
                • 预期收益 = 预测涨跌 × 历史准确率 - 手续费<br>
                • 只分配给扣费后仍有正收益的股票
            </div>
        </div>

        <div class="footer">
            <strong>策略信息:</strong> 混合分配（准确率×涨幅） | 风险控制: 单日最大亏损3%<br>
            <strong>持有期:</strong> 1天（明天卖出） | <strong>止损:</strong> 单只股票2-3% | <strong>交易平台:</strong> Trading 212<br>
            <strong>数据来源:</strong> <a href="https://helpcentre.trading212.com/hc/en-us/articles/360018909758" target="_blank" style="color: #667eea;">Trading 212 FX费用说明</a>
        </div>
    </div>

    <script>
        const stocksData = {stocks_json};

        function openChart(symbol) {{
            // Map stocks to their correct exchanges
            const stockExchanges = {{
                'BA': 'NYSE',      // Boeing
                'CAT': 'NYSE',     // Caterpillar
                'GS': 'NYSE',      // Goldman Sachs
                'JPM': 'NYSE',     // JPMorgan Chase
                'INTC': 'NASDAQ',  // Intel
                'NVDA': 'NASDAQ',  // NVIDIA
                'AMD': 'NASDAQ',   // AMD
                'TSLA': 'NASDAQ',  // Tesla
                'CRM': 'NYSE',     // Salesforce
                'ORCL': 'NYSE'     // Oracle
            }};

            const exchange = stockExchanges[symbol] || 'NASDAQ';
            const tradingViewUrl = `https://www.tradingview.com/chart/?symbol=${{exchange}}:${{symbol}}`;

            // Alternative URLs (uncomment to use):
            // Yahoo Finance (auto-detects exchange): `https://finance.yahoo.com/quote/${{symbol}}`
            // Google Finance: `https://www.google.com/finance/quote/${{symbol}}`

            window.open(tradingViewUrl, '_blank');
        }}

        function updateTable() {{
            const capital = parseFloat(document.getElementById('capitalInput').value) || 100;

            const top7 = stocksData.slice(0, 7);
            const others = stocksData.slice(7);

            // Update TOP 7 table
            const top7Body = document.getElementById('top7Table').querySelector('tbody');
            top7Body.innerHTML = '';

            top7.forEach(stock => {{
                const action = stock.prediction > 0 ? 'BUY' : 'SELL';
                const actionClass = action === 'BUY' ? 'action-buy' : 'action-sell';
                const predClass = stock.prediction > 0 ? 'positive' : 'negative';
                const expReturnClass = stock.expected_return > 0 ? 'positive' : 'negative';
                const amount = (stock.position_pct / 100 * capital).toFixed(2);

                const row = `
                    <tr>
                        <td class="rank">#${{stock.rank}}</td>
                        <td class="symbol">
                            <span class="clickable-symbol" onclick="openChart('${{stock.symbol}}')" title="点击查看图表">
                                ${{stock.symbol}}
                            </span>
                            <span class="top7-badge">TOP 7</span>
                        </td>
                        <td>£${{stock.current_price.toFixed(2)}}</td>
                        <td><span class="${{actionClass}}">${{action}}</span></td>
                        <td class="${{predClass}}">${{(stock.prediction * 100).toFixed(2)}}%</td>
                        <td><strong>${{stock.accuracy.toFixed(1)}}%</strong></td>
                        <td class="${{expReturnClass}}"><strong>${{(stock.expected_return * 100).toFixed(2)}}%</strong></td>
                        <td><strong>${{stock.position_pct.toFixed(1)}}%</strong></td>
                        <td class="amount">£${{amount}}</td>
                    </tr>
                `;
                top7Body.innerHTML += row;
            }});

            // Update others table
            const othersBody = document.getElementById('othersTable').querySelector('tbody');
            othersBody.innerHTML = '';

            if (others.length === 0) {{
                othersBody.innerHTML = '<tr><td colspan="9" style="text-align:center; color:#999;">暂无其他股票</td></tr>';
            }} else {{
                others.forEach(stock => {{
                    const action = stock.prediction > 0 ? 'BUY' : 'SELL';
                    const actionClass = action === 'BUY' ? 'action-buy' : 'action-sell';
                    const predClass = stock.prediction > 0 ? 'positive' : 'negative';
                    const expReturnClass = stock.expected_return > 0 ? 'positive' : 'negative';
                    const amount = (stock.position_pct / 100 * capital).toFixed(2);

                    const row = `
                        <tr>
                            <td class="rank">#${{stock.rank}}</td>
                            <td class="symbol">
                                <span class="clickable-symbol" onclick="openChart('${{stock.symbol}}')" title="点击查看图表">
                                    ${{stock.symbol}}
                                </span>
                            </td>
                            <td>£${{stock.current_price.toFixed(2)}}</td>
                            <td><span class="${{actionClass}}">${{action}}</span></td>
                            <td class="${{predClass}}">${{(stock.prediction * 100).toFixed(2)}}%</td>
                            <td><strong>${{stock.accuracy.toFixed(1)}}%</strong></td>
                            <td class="${{expReturnClass}}"><strong>${{(stock.expected_return * 100).toFixed(2)}}%</strong></td>
                            <td><strong>${{stock.position_pct.toFixed(1)}}%</strong></td>
                            <td class="amount">£${{amount}}</td>
                        </tr>
                    `;
                    othersBody.innerHTML += row;
                }});
            }}

            // Update profit summary
            document.getElementById('capitalDisplay').textContent = `£${{capital.toFixed(2)}}`;

            // Calculate total expected profit
            let totalExpectedProfit = 0;
            stocksData.forEach(stock => {{
                const investmentAmount = capital * stock.position_pct / 100;
                const expectedReturn = investmentAmount * stock.expected_return;
                totalExpectedProfit += expectedReturn;
            }});

            // Calculate total fees
            const totalFees = capital * 0.003;  // 0.30% of total capital

            // Display results
            const profitClass = totalExpectedProfit >= 0 ? 'positive' : 'negative';
            const profitPct = (totalExpectedProfit / capital * 100).toFixed(2);

            document.getElementById('expectedProfit').textContent = `£${{totalExpectedProfit.toFixed(2)}}`;
            document.getElementById('expectedProfit').className = totalExpectedProfit >= 0 ? '' : 'negative';
            document.getElementById('expectedProfitPct').textContent = `(${{profitPct}}%)`;
            document.getElementById('totalFees').textContent = `£${{totalFees.toFixed(2)}}`;
        }}

        // Update on capital input change
        document.getElementById('capitalInput').addEventListener('input', updateTable);

        // Initial update
        updateTable();
    </script>
</body>
</html>"""

    # Save HTML
    output_path = web_dir / "signals.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Web page generated: {output_path}")
    print(f"Open in browser: file:///{output_path.as_posix()}")

    return output_path


if __name__ == "__main__":
    generate_web_page()
