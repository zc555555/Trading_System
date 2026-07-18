"""
T+1动态交易监控系统

功能：
1. 止损：单股亏损 > STOP_LOSS_PCT 自动平仓
2. 止盈：单股盈利 > TAKE_PROFIT_PCT 自动平仓（锁定利润）
3. 日内最大亏损：总亏损 > MAX_DAILY_LOSS_PCT 全部平仓
4. (可选) 美股 16:00 ET 收盘后平仓所有持仓

时区：所有交易时间判断均基于 America/New_York（自动处理夏令时）
运行时间：美股盘中（9:30-16:00 ET）每5分钟运行一次

W2-B (2026-05): 风控参数 + EOD 平仓现在都来自 config_trading.py，让 1-day
"legacy" 策略和多日 "staggered" 策略共用同一个监控器。staggered 模式下
EOD_FLATTEN=False，持仓跨日保留至 tranche scheduled_close_date。
"""

from alpaca.trading.client import TradingClient
from config_alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY
from datetime import datetime, time
from pathlib import Path
import json

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

# Import alert system
try:
    from alert_system import AlertSystem
    ALERTS_ENABLED = True
except ImportError:
    ALERTS_ENABLED = False
    print("[WARNING] Alert system not available")

# Pull risk params from the central trading config so both strategies stay in sync.
# Falls back to legacy hard-coded defaults if config_trading is missing -- so this
# module remains importable even if W2-B hasn't been deployed yet.
try:
    import config_trading as _tcfg
    STOP_LOSS_PCT = _tcfg.STOP_LOSS_PCT
    TAKE_PROFIT_PCT = _tcfg.TAKE_PROFIT_PCT
    MAX_DAILY_LOSS_PCT = _tcfg.MAX_DAILY_LOSS_PCT
    EOD_FLATTEN_ENABLED = _tcfg.EOD_FLATTEN
    _eod_h, _eod_m = (int(x) for x in _tcfg.EOD_CLOSE_TIME_ET.split(":"))
    EOD_CLOSE_TIME = time(_eod_h, _eod_m)
except ImportError:
    STOP_LOSS_PCT = 0.025
    TAKE_PROFIT_PCT = 0.025
    MAX_DAILY_LOSS_PCT = 0.03
    EOD_FLATTEN_ENABLED = True
    EOD_CLOSE_TIME = time(16, 0)

# 美股交易所时区（自动处理夏令时）
MARKET_TZ = ZoneInfo("America/New_York")

class DynamicTradingMonitor:
    """动态交易监控器（集成实时预警）"""

    def __init__(self, enable_email_alerts: bool = False):
        self.client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        self.logs_dir = Path(__file__).parent / "trading_logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Initialize alert system
        self.alert = AlertSystem(enable_email=enable_email_alerts) if ALERTS_ENABLED else None

        if self.alert:
            self.alert.alert_system_startup("Dynamic Trading Monitor")

    # ------------------------------------------------------------------
    # W2-C: per-position stop/take lookup from the tranche registry.
    # If the registry exists and the symbol is in it, use the exact
    # entry-time stop_price and take_price. Otherwise fall back to the
    # fixed-% thresholds. Multiple tranches owning the same symbol get
    # the TIGHTEST stop (most conservative) and tightest take (lock first).
    # ------------------------------------------------------------------
    def _load_position_risk_index(self) -> dict:
        """Return {(symbol, side): {'stop_price': float, 'take_price': float}}.

        Returns empty dict if registry not present (legacy behavior).
        """
        try:
            from trading.tranche_registry import TrancheRegistry
            import config_trading as _tcfg
            reg_path = Path(__file__).parent / _tcfg.TRANCHE_REGISTRY_PATH
            if not reg_path.exists():
                return {}
            reg = TrancheRegistry(reg_path)
        except Exception as e:
            print(f"[WARN] Could not load tranche registry: {e}")
            return {}

        idx: dict = {}
        for t in reg.open_tranches():
            for p in t.longs:
                if p.stop_price is None or p.take_price is None:
                    continue
                key = (p.symbol, "long")
                cur = idx.get(key)
                if cur is None:
                    idx[key] = {"stop_price": p.stop_price,
                                "take_price": p.take_price}
                else:
                    # tightest stop = max for long (closer to entry from below)
                    cur["stop_price"] = max(cur["stop_price"], p.stop_price)
                    # tightest take = min for long (lock profit sooner)
                    cur["take_price"] = min(cur["take_price"], p.take_price)
            for p in t.shorts:
                if p.stop_price is None or p.take_price is None:
                    continue
                key = (p.symbol, "short")
                cur = idx.get(key)
                if cur is None:
                    idx[key] = {"stop_price": p.stop_price,
                                "take_price": p.take_price}
                else:
                    # tightest stop = min for short (closer to entry from above)
                    cur["stop_price"] = min(cur["stop_price"], p.stop_price)
                    # tightest take = max for short (lock profit sooner)
                    cur["take_price"] = max(cur["take_price"], p.take_price)
        return idx

    def check_and_execute(self):
        """检查持仓并执行动态交易策略"""
        now_local = datetime.now()
        now_et = datetime.now(MARKET_TZ)
        current_time = now_et.time()

        print("="*80)
        print(f"动态交易监控 - 本地 {now_local.strftime('%Y-%m-%d %H:%M:%S')} / ET {now_et.strftime('%H:%M:%S')}")
        print("="*80)
        print()

        # 获取账户和持仓
        account = self.client.get_account()
        positions = self.client.get_all_positions()

        if len(positions) == 0:
            print("[INFO] 当前无持仓，监控系统待机")
            print("[INFO] 系统将在下次买入后自动激活")
            print()
            return  # 无持仓时直接退出，不做任何操作

        print(f"当前持仓: {len(positions)} 个")
        print()

        # 账户状态
        total_equity = float(account.equity)
        initial_equity = float(account.last_equity)
        daily_pl_pct = (total_equity - initial_equity) / initial_equity * 100 if initial_equity > 0 else 0

        print(f"账户状态:")
        print(f"  总资产: ${total_equity:,.2f}")
        print(f"  初始资产: ${initial_equity:,.2f}")
        print(f"  日内盈亏: ${total_equity - initial_equity:+,.2f} ({daily_pl_pct:+.2f}%)")
        print()

        # 检查1: EOD 平仓（仅当 config_trading.EOD_FLATTEN=True 时生效）
        # 在 staggered 多日策略下应保持禁用，否则会破坏 tranche 的 hold_days 周期。
        if EOD_FLATTEN_ENABLED and current_time >= EOD_CLOSE_TIME:
            print(f"⏰ 到达美股 {EOD_CLOSE_TIME} ET 收盘时间，执行当日持仓自动平仓...")
            self.close_all_positions("EOD_AUTO_CLOSE", "美股收盘，自动平仓准备明日交易")
            return
        elif current_time >= EOD_CLOSE_TIME and not EOD_FLATTEN_ENABLED:
            print(f"[INFO] 到达 {EOD_CLOSE_TIME} ET 但 EOD_FLATTEN=False -- "
                  f"保留持仓（staggered 多日策略）。")

        # 检查2: 是否触发最大日内亏损
        if daily_pl_pct <= -MAX_DAILY_LOSS_PCT * 100:
            print(f"🚫 触发最大日内亏损限制！({daily_pl_pct:.2f}% <= -{MAX_DAILY_LOSS_PCT*100}%)")

            # Send critical alert
            if self.alert:
                self.alert.alert_max_loss_triggered(daily_pl_pct, len(positions))

            self.close_all_positions("MAX_DAILY_LOSS", f"日内亏损{daily_pl_pct:.2f}%，触发最大亏损保护")
            return

        # W2-C: load per-position stop/take from tranche registry (one read per cycle)
        risk_index = self._load_position_risk_index()
        if risk_index:
            print(f"[risk] per-position stops loaded for {len(risk_index)} symbol-side combos")
        else:
            print(f"[risk] no registry data -- falling back to fixed-% stops")

        # 检查3: 逐个检查止损和止盈
        actions = []

        print("持仓分析:")
        print(f"{'股票':<8} {'数量':<8} {'成本':<10} {'当前':<10} {'盈亏%':<10} {'阈值':<22} {'动作'}")
        print("-" * 100)

        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            entry_price = float(pos.avg_entry_price)
            current_price = float(pos.current_price)
            unrealized_plpc = float(pos.unrealized_plpc) * 100
            side = "long" if qty > 0 else "short"

            action = None
            reason = None

            # Look up per-position stop/take from registry; fall back to fixed %
            levels = risk_index.get((symbol, side))
            if levels:
                stop_price = float(levels["stop_price"])
                take_price = float(levels["take_price"])
                threshold_desc = f"S=${stop_price:.2f}/T=${take_price:.2f}"
                # direction-aware breach check
                stop_hit = (current_price <= stop_price) if side == "long" else (current_price >= stop_price)
                take_hit = (current_price >= take_price) if side == "long" else (current_price <= take_price)
            else:
                # Fallback: legacy fixed-percentage check
                threshold_desc = f"fixed {STOP_LOSS_PCT*100:.0f}%/{TAKE_PROFIT_PCT*100:.0f}%"
                stop_hit = unrealized_plpc <= -STOP_LOSS_PCT * 100
                take_hit = unrealized_plpc >= TAKE_PROFIT_PCT * 100

            if stop_hit:
                action = "STOP_LOSS"
                reason = f"亏损{unrealized_plpc:.2f}%，触发止损 ({threshold_desc})"
                emoji = "🚫"
                if self.alert:
                    self.alert.alert_stop_loss_triggered(
                        symbol, entry_price, current_price, unrealized_plpc
                    )
            elif take_hit:
                action = "TAKE_PROFIT"
                reason = f"盈利{unrealized_plpc:.2f}%，锁定利润 ({threshold_desc})"
                emoji = "💰"
                if self.alert:
                    self.alert.alert_take_profit_triggered(
                        symbol, entry_price, current_price, unrealized_plpc
                    )
            else:
                emoji = "✅"
                reason = f"正常持有 ({threshold_desc})"

            print(f"{symbol:<8} {qty:<8.0f} ${entry_price:<9.2f} ${current_price:<9.2f} "
                  f"{unrealized_plpc:>+7.2f}%  {threshold_desc:<22} {emoji} {reason}")

            if action:
                actions.append({
                    'symbol': symbol,
                    'qty': qty,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'pl_pct': unrealized_plpc,
                    'pl_amount': float(pos.unrealized_pl),
                    'action': action,
                    'reason': reason,
                    'timestamp': now_local.isoformat()
                })

        print()

        # 执行动作
        if actions:
            print("="*80)
            print(f"检测到 {len(actions)} 个持仓需要平仓")
            print("="*80)
            print()

            executed_actions = []

            for item in actions:
                symbol = item['symbol']
                action = item['action']
                reason = item['reason']

                print(f"平仓 {symbol} - {reason}")

                try:
                    self.client.close_position(symbol)
                    print(f"  ✓ 已平仓")
                    print(f"    入场价: ${item['entry_price']:.2f}")
                    print(f"    平仓价: ${item['current_price']:.2f}")
                    print(f"    盈亏: {item['pl_pct']:+.2f}% (${item['pl_amount']:+.2f})")
                    print()

                    executed_actions.append(item)

                except Exception as e:
                    print(f"  ✗ 平仓失败: {e}")
                    print()

            # 记录事件
            if executed_actions:
                self.log_trading_events(executed_actions)

            print("="*80)
            print(f"动作执行完成！共平仓 {len(executed_actions)} 个持仓")

            # 统计
            stop_loss_count = sum(1 for a in executed_actions if a['action'] == 'STOP_LOSS')
            take_profit_count = sum(1 for a in executed_actions if a['action'] == 'TAKE_PROFIT')
            total_pl = sum(a['pl_amount'] for a in executed_actions)

            print(f"  止损: {stop_loss_count} 个")
            print(f"  止盈: {take_profit_count} 个")
            print(f"  合计盈亏: ${total_pl:+.2f}")
            print("="*80)

        else:
            print("✅ 所有持仓正常，继续持有")

        print()

    def close_all_positions(self, reason_code: str, reason_desc: str):
        """平掉所有持仓"""
        positions = self.client.get_all_positions()

        print()
        print("="*80)
        print(f"执行全部平仓")
        print(f"原因: {reason_desc}")
        print("="*80)
        print()

        executed = []

        for pos in positions:
            symbol = pos.symbol
            try:
                self.client.close_position(symbol)
                print(f"✓ {symbol} 已平仓 (盈亏: {float(pos.unrealized_plpc)*100:+.2f}%)")

                executed.append({
                    'symbol': symbol,
                    'action': reason_code,
                    'reason': reason_desc,
                    'pl_pct': float(pos.unrealized_plpc) * 100,
                    'pl_amount': float(pos.unrealized_pl),
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                print(f"✗ {symbol} 平仓失败: {e}")

        # 记录事件
        if executed:
            self.log_trading_events(executed)

        print()
        print("="*80)
        print("全部平仓完成")
        print("="*80)
        print()

    def log_trading_events(self, events: list):
        """记录交易事件"""
        log_file = self.logs_dir / f"dynamic_trading_events_{datetime.now().strftime('%Y%m%d')}.json"

        # 读取现有记录
        if log_file.exists():
            with open(log_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = []

        # 添加新事件
        existing.extend(events)

        # 保存
        with open(log_file, 'w') as f:
            json.dump(existing, f, indent=2)

        print(f"[OK] 交易事件已记录: {log_file}")


def run_auto_monitor(interval_seconds: int = 300, enable_email_alerts: bool = False):
    """循环监控模式：无持仓时跳过，有持仓时执行完整动态策略。"""
    import time as _time
    import sys

    monitor = DynamicTradingMonitor(enable_email_alerts=enable_email_alerts)

    print("=" * 80)
    print("动态交易监控 - 自动循环模式")
    print("=" * 80)
    print(f"配置: 止损 {STOP_LOSS_PCT*100}%, 止盈 {TAKE_PROFIT_PCT*100}%, "
          f"日内最大亏损 {MAX_DAILY_LOSS_PCT*100}%, "
          f"EOD 平仓 {'ON' if EOD_FLATTEN_ENABLED else 'OFF'} "
          f"({EOD_CLOSE_TIME} ET)")
    print(f"检查间隔: 每 {interval_seconds // 60} 分钟")
    print("按 Ctrl+C 停止")
    print("=" * 80)
    print()

    try:
        while True:
            positions = monitor.client.get_all_positions()
            if len(positions) == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 当前无持仓，跳过检查")
            else:
                monitor.check_and_execute()

            print(f"\n下次检查: {interval_seconds // 60} 分钟后...")
            print("-" * 80)
            print()
            _time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户终止监控")
        print("=" * 80)
        print("动态交易监控已停止")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='动态交易监控系统（止损/止盈/日内最大亏损/EOD 平仓）')
    parser.add_argument('--auto', action='store_true',
                        help='自动循环模式：无持仓时跳过，有持仓时每 N 秒检查一次')
    parser.add_argument('--interval', type=int, default=300,
                        help='循环检查间隔秒数（默认 300 = 5 分钟），仅 --auto 生效')
    parser.add_argument('--email-alerts', action='store_true',
                        help='启用邮件预警（需要 alert_system 配置）')
    args = parser.parse_args()

    if args.auto:
        run_auto_monitor(interval_seconds=args.interval,
                         enable_email_alerts=args.email_alerts)
    else:
        DynamicTradingMonitor(enable_email_alerts=args.email_alerts).check_and_execute()
