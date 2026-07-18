"""快速查看账户状态和成本"""
import sys
import io

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from monitor_dynamic_trading import DynamicTradingMonitor

def main():
    print("\n" + "="*80)
    print("账户状态和成本")
    print("="*80)

    m = DynamicTradingMonitor()

    # 获取账户信息
    account = m.client.get_account()

    equity = float(account.equity)
    cash = float(account.cash)
    market_value = float(account.long_market_value)
    last_equity = float(account.last_equity)

    print(f"\n💰 账户总览:")
    print(f"  账户价值: ${equity:,.2f}")
    print(f"  可用现金: ${cash:,.2f}")
    print(f"  持仓市值: ${market_value:,.2f}")
    print(f"  今日盈亏: ${equity - last_equity:+,.2f}")
    print(f"  总盈亏:   ${equity - 100000:+,.2f}")

    # 获取持仓
    positions = m.client.get_all_positions()

    if not positions:
        print("\n📊 当前持仓: 无")
        print("\n✅ 没有任何持仓成本")
    else:
        print(f"\n📊 当前持仓 ({len(positions)} 只股票):")
        print("-" * 80)

        total_cost = 0
        total_pl = 0

        for p in positions:
            symbol = p.symbol
            qty = float(p.qty)
            avg_price = float(p.avg_entry_price)
            current_price = float(p.current_price)
            pl = float(p.unrealized_pl)
            pl_pct = float(p.unrealized_plpc) * 100

            cost = qty * avg_price
            total_cost += cost
            total_pl += pl

            status = "📈" if pl > 0 else "📉" if pl < 0 else "➡️"

            print(f"\n  {status} {symbol}:")
            print(f"     数量: {qty:.0f} 股")
            print(f"     成本价: ${avg_price:.2f}")
            print(f"     当前价: ${current_price:.2f}")
            print(f"     持仓成本: ${cost:.2f}")
            print(f"     未实现盈亏: ${pl:+.2f} ({pl_pct:+.2f}%)")

        print("\n" + "-" * 80)
        print(f"\n  总持仓成本: ${total_cost:,.2f}")
        print(f"  总未实现盈亏: ${total_pl:+,.2f}")

    # 运行成本
    print("\n" + "="*80)
    print("💸 运行成本估算:")
    print("="*80)
    print("\n  API费用:")
    print("    • Alpaca Paper Trading: $0/月 (免费)")
    print("    • Yahoo Finance: $0/月 (免费)")
    print("    • Finnhub News API: $0/月 (免费额度)")
    print("\n  交易成本:")
    print("    • Alpaca佣金: $0 (Paper Trading无佣金)")
    print("\n  ✅ 总运行成本: $0/月")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
