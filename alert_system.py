"""
REAL-TIME ALERT SYSTEM

Monitors trading activities and sends alerts for:
- Abnormal price movements (>5% sudden drop)
- Stop-loss/take-profit triggers
- Negative news events
- System errors (API failures, data issues)
- Performance thresholds

Alert channels:
- Console output (always)
- Email notifications (optional, configurable)
- Log file (persistent)
"""

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging


class AlertSystem:
    """Real-time alert system for trading monitoring"""

    # Alert severity levels
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    def __init__(self,
                 enable_email: bool = False,
                 email_config_file: str = "config_email.json"):
        """
        Initialize alert system

        Args:
            enable_email: Whether to send email alerts
            email_config_file: Path to email configuration file
        """
        self.enable_email = enable_email
        self.email_config = None

        # Setup logging
        log_dir = Path("alerts")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

        # Load email config if enabled
        if enable_email:
            self._load_email_config(email_config_file)

        self.logger.info("Alert System initialized")

    def _load_email_config(self, config_file: str):
        """Load email configuration from file"""
        config_path = Path(config_file)

        if not config_path.exists():
            self.logger.warning(f"Email config not found: {config_file}")
            self.logger.info("Creating template email config...")

            template = {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "your_email@gmail.com",
                "sender_password": "your_app_password",
                "recipient_emails": ["your_email@gmail.com"],
                "note": "For Gmail, use App Password: https://support.google.com/accounts/answer/185833"
            }

            with open(config_path, 'w') as f:
                json.dump(template, f, indent=4)

            self.logger.info(f"Template created: {config_file}")
            self.logger.info("Please configure your email settings before enabling email alerts")
            self.enable_email = False
            return

        with open(config_path, 'r') as f:
            self.email_config = json.load(f)

        self.logger.info("Email configuration loaded")

    def send_alert(self,
                   severity: str,
                   title: str,
                   message: str,
                   details: Optional[Dict] = None):
        """
        Send alert through all enabled channels

        Args:
            severity: INFO, WARNING, or CRITICAL
            title: Short alert title
            message: Detailed alert message
            details: Additional structured data
        """
        # Format alert
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        alert_text = f"""
{'='*80}
[{severity}] {title}
{'='*80}
Time: {timestamp}

{message}
"""

        if details:
            alert_text += "\nDetails:\n"
            for key, value in details.items():
                alert_text += f"  - {key}: {value}\n"

        alert_text += "="*80

        # Console and log output
        if severity == self.CRITICAL:
            self.logger.critical(alert_text)
        elif severity == self.WARNING:
            self.logger.warning(alert_text)
        else:
            self.logger.info(alert_text)

        # Email notification (for WARNING and CRITICAL only)
        if self.enable_email and severity in [self.WARNING, self.CRITICAL]:
            self._send_email(severity, title, message, details)

    def _send_email(self, severity: str, title: str, message: str, details: Optional[Dict] = None):
        """Send email notification"""
        if not self.email_config:
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipient_emails'])
            msg['Subject'] = f"[{severity}] Stock Trading Alert: {title}"

            # Email body
            body = f"""
Stock Trading Alert System
{'='*60}

Severity: {severity}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{message}
"""

            if details:
                body += "\n\nDetails:\n"
                for key, value in details.items():
                    body += f"  {key}: {value}\n"

            body += f"\n{'='*60}\nThis is an automated alert from your trading system."

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'],
                                 self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'],
                        self.email_config['sender_password'])

            server.send_message(msg)
            server.quit()

            self.logger.info(f"Email alert sent: {title}")

        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")

    # ========================================================================
    # Pre-defined alert types
    # ========================================================================

    def alert_abnormal_movement(self, symbol: str, change_pct: float, timeframe: str = "5min"):
        """Alert for abnormal price movement"""
        severity = self.CRITICAL if abs(change_pct) > 10 else self.WARNING

        direction = "SURGE" if change_pct > 0 else "DROP"

        self.send_alert(
            severity=severity,
            title=f"Abnormal Price {direction}: {symbol}",
            message=f"{symbol} has moved {change_pct:+.2f}% in the last {timeframe}!",
            details={
                'Symbol': symbol,
                'Change': f"{change_pct:+.2f}%",
                'Timeframe': timeframe,
                'Action': 'Review position and consider manual intervention'
            }
        )

    def alert_stop_loss_triggered(self, symbol: str, entry_price: float,
                                   exit_price: float, loss_pct: float):
        """Alert when stop-loss is triggered"""
        self.send_alert(
            severity=self.WARNING,
            title=f"Stop-Loss Triggered: {symbol}",
            message=f"Position in {symbol} closed due to stop-loss at {loss_pct:.2f}% loss",
            details={
                'Symbol': symbol,
                'Entry Price': f"${entry_price:.2f}",
                'Exit Price': f"${exit_price:.2f}",
                'Loss': f"{loss_pct:.2f}%",
                'Action': 'Position closed automatically'
            }
        )

    def alert_take_profit_triggered(self, symbol: str, entry_price: float,
                                     exit_price: float, profit_pct: float):
        """Alert when take-profit is triggered"""
        self.send_alert(
            severity=self.INFO,
            title=f"Take-Profit Triggered: {symbol}",
            message=f"Position in {symbol} closed with {profit_pct:.2f}% profit!",
            details={
                'Symbol': symbol,
                'Entry Price': f"${entry_price:.2f}",
                'Exit Price': f"${exit_price:.2f}",
                'Profit': f"{profit_pct:.2f}%",
                'Action': 'Position closed successfully'
            }
        )

    def alert_max_loss_triggered(self, total_loss_pct: float, positions_closed: int):
        """Alert when maximum daily loss is hit"""
        self.send_alert(
            severity=self.CRITICAL,
            title="Maximum Daily Loss Triggered",
            message=f"Total loss reached {total_loss_pct:.2f}%. All {positions_closed} positions closed!",
            details={
                'Total Loss': f"{total_loss_pct:.2f}%",
                'Positions Closed': positions_closed,
                'Action': 'Trading halted for today',
                'Note': 'System will resume tomorrow'
            }
        )

    def alert_negative_news(self, symbol: str, sentiment_score: float,
                           news_count: int):
        """Alert for highly negative news"""
        self.send_alert(
            severity=self.WARNING,
            title=f"Negative News Alert: {symbol}",
            message=f"{symbol} has {news_count} negative news items (sentiment: {sentiment_score:.2f})",
            details={
                'Symbol': symbol,
                'Sentiment Score': f"{sentiment_score:.2f}",
                'News Count': news_count,
                'Action': 'Consider reviewing position'
            }
        )

    def alert_api_failure(self, api_name: str, error_message: str):
        """Alert for API connection failures"""
        self.send_alert(
            severity=self.CRITICAL,
            title=f"API Failure: {api_name}",
            message=f"Failed to connect to {api_name} API",
            details={
                'API': api_name,
                'Error': error_message,
                'Action': 'Check API credentials and connection',
                'Impact': 'Trading may be affected'
            }
        )

    def alert_data_quality_issue(self, issue_type: str, affected_symbols: List[str],
                                  details: Dict):
        """Alert for data quality problems"""
        self.send_alert(
            severity=self.WARNING,
            title=f"Data Quality Issue: {issue_type}",
            message=f"Data quality problem detected affecting {len(affected_symbols)} symbols",
            details={
                'Issue Type': issue_type,
                'Affected Symbols': ', '.join(affected_symbols[:5]) +
                                  (f" (+{len(affected_symbols)-5} more)" if len(affected_symbols) > 5 else ""),
                **details
            }
        )

    def alert_model_performance_degradation(self, current_accuracy: float,
                                            expected_accuracy: float):
        """Alert when model performance drops"""
        self.send_alert(
            severity=self.WARNING,
            title="Model Performance Degradation",
            message=f"Model accuracy dropped to {current_accuracy:.1f}% (expected: {expected_accuracy:.1f}%)",
            details={
                'Current Accuracy': f"{current_accuracy:.1f}%",
                'Expected Accuracy': f"{expected_accuracy:.1f}%",
                'Degradation': f"{expected_accuracy - current_accuracy:.1f}%",
                'Action': 'Consider retraining models',
                'Script': 'Run retrain_models_weekly.bat'
            }
        )

    def alert_trade_executed(self, action: str, symbol: str, quantity: int,
                            price: float, total_value: float):
        """Alert for trade execution"""
        self.send_alert(
            severity=self.INFO,
            title=f"Trade Executed: {action} {symbol}",
            message=f"{action} {quantity} shares of {symbol} at ${price:.2f}",
            details={
                'Action': action,
                'Symbol': symbol,
                'Quantity': quantity,
                'Price': f"${price:.2f}",
                'Total Value': f"${total_value:.2f}"
            }
        )

    def alert_daily_summary(self, total_pnl: float, total_pnl_pct: float,
                           trades_count: int, win_rate: float):
        """Alert for daily trading summary"""
        severity = self.INFO if total_pnl >= 0 else self.WARNING

        self.send_alert(
            severity=severity,
            title=f"Daily Trading Summary",
            message=f"Today's P&L: ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)",
            details={
                'Total P&L': f"${total_pnl:+.2f}",
                'P&L Percentage': f"{total_pnl_pct:+.2f}%",
                'Trades': trades_count,
                'Win Rate': f"{win_rate:.1f}%"
            }
        )

    def alert_system_startup(self, component: str):
        """Alert when system components start"""
        self.send_alert(
            severity=self.INFO,
            title=f"System Started: {component}",
            message=f"{component} is now running",
            details={
                'Component': component,
                'Status': 'Active'
            }
        )

    def alert_system_shutdown(self, component: str, reason: str = "Normal shutdown"):
        """Alert when system components shut down"""
        self.send_alert(
            severity=self.INFO,
            title=f"System Stopped: {component}",
            message=f"{component} has stopped: {reason}",
            details={
                'Component': component,
                'Reason': reason
            }
        )


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    print("Alert System - Test Mode")
    print("="*80)

    # Initialize alert system (email disabled by default)
    alert = AlertSystem(enable_email=False)

    print("\nTesting different alert types...\n")

    # Test 1: Abnormal movement
    alert.alert_abnormal_movement("AAPL", -7.5, "5min")

    # Test 2: Stop-loss
    alert.alert_stop_loss_triggered("MSFT", 420.00, 410.00, -2.38)

    # Test 3: Take-profit
    alert.alert_take_profit_triggered("GOOGL", 150.00, 154.00, 2.67)

    # Test 4: Negative news
    alert.alert_negative_news("TSLA", -0.65, 5)

    # Test 5: API failure
    alert.alert_api_failure("Alpaca", "Connection timeout")

    # Test 6: Daily summary
    alert.alert_daily_summary(total_pnl=250.50, total_pnl_pct=2.5,
                             trades_count=4, win_rate=75.0)

    print("\n" + "="*80)
    print("Test completed! Check alerts/ directory for log files.")
    print("\nTo enable email alerts:")
    print("1. Edit config_email.json with your email settings")
    print("2. Use AlertSystem(enable_email=True)")
