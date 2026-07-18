"""检查和取消挂起订单

用法:
    python check_and_cancel_orders.py            # 交互模式：列出并询问
    python check_and_cancel_orders.py --buy-only # 直接取消 BUY 订单，保留 SELL
    python check_and_cancel_orders.py --all      # 直接取消全部订单
"""
import argparse
import io
import sys

from alpaca.trading.client import TradingClient

from config_alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY

# 设置 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def cancel_orders(client: TradingClient, orders) -> int:
    success = 0
    for order in orders:
        try:
            client.cancel_order_by_id(order.id)
            print(f"✅ 已取消: {order.side} {order.symbol} x{order.qty}")
            success += 1
        except Exception as e:
            print(f"❌ 取消失败 {order.symbol}: {e}")
    return success


def main():
    parser = argparse.ArgumentParser(description='检查和取消 Alpaca 挂起订单')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--buy-only', action='store_true',
                       help='非交互：只取消买入订单，保留平仓订单')
    group.add_argument('--all', action='store_true',
                       help='非交互：取消全部挂起订单')
    args = parser.parse_args()

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    orders = client.get_orders()

    print("=" * 80)
    print("当前挂起的订单")
    print("=" * 80)
    print()

    if len(orders) == 0:
        print("✅ 没有挂起的订单")
        return

    buy_orders = [o for o in orders if o.side.name == 'BUY']
    sell_orders = [o for o in orders if o.side.name == 'SELL']

    for i, order in enumerate(orders, 1):
        print(f"订单 {i}:  {order.side}  {order.symbol} x{order.qty}  "
              f"状态={order.status}  id={order.id}")

    print()
    print(f"总结: {len(buy_orders)} 个买入订单, {len(sell_orders)} 个平仓订单")
    print("=" * 80)
    print()

    if args.buy_only:
        choice = '2'
    elif args.all:
        choice = '1'
    else:
        print("您想要:")
        print("  1. 取消所有订单")
        print("  2. 只取消买入订单（保留平仓订单）")
        print("  3. 不取消，保持原样")
        print()
        choice = input("请选择 (1/2/3): ").strip()

    if choice == '1':
        print("\n取消所有订单...")
        n = cancel_orders(client, orders)
        print(f"\n✅ {n}/{len(orders)} 个订单已取消")
    elif choice == '2':
        if not buy_orders:
            print("✅ 没有买入订单需要取消")
            return
        print("\n取消买入订单，保留平仓订单...")
        n = cancel_orders(client, buy_orders)
        print(f"\n✅ {n}/{len(buy_orders)} 个买入订单已取消")
        print(f"✓  {len(sell_orders)} 个平仓订单保留")
    else:
        print("\n保持现有订单不变")


if __name__ == "__main__":
    main()
