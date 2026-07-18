"""
DATA QUALITY MONITORING SYSTEM

Monitors data quality and detects issues:
- Missing data (NaN values, gaps in time series)
- Anomalies (price spikes, zero values, extreme outliers)
- Staleness (outdated data)
- Data consistency (duplicate entries, conflicting values)

Integrates with alert system to notify of critical issues.

Usage:
    from data_quality_monitor import DataQualityMonitor
    monitor = DataQualityMonitor()
    report = monitor.check_data_quality(df)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import alert system
try:
    from alert_system import AlertSystem
except ImportError:
    AlertSystem = None


class DataQualityMonitor:
    """Monitor and validate data quality"""

    def __init__(self, use_alerts: bool = True):
        """
        Initialize data quality monitor

        Args:
            use_alerts: Whether to send alerts for issues
        """
        self.use_alerts = use_alerts
        self.alert_system = AlertSystem() if AlertSystem and use_alerts else None

        self.issues = []

    def check_data_quality(self, df: pd.DataFrame,
                           symbol_col: str = 'symbol',
                           date_col: str = 'date',
                           price_cols: List[str] = None) -> Dict:
        """
        Comprehensive data quality check

        Args:
            df: DataFrame to check
            symbol_col: Name of symbol column
            date_col: Name of date column
            price_cols: List of price column names

        Returns:
            dict: Quality report
        """
        if price_cols is None:
            price_cols = ['open', 'high', 'low', 'close']

        print("="*80)
        print("DATA QUALITY MONITORING")
        print("="*80)
        print(f"\nDataset: {len(df)} rows, {len(df.columns)} columns")

        if symbol_col in df.columns:
            print(f"Symbols: {df[symbol_col].nunique()}")

        self.issues = []

        # Run checks
        missing_check = self._check_missing_data(df)
        anomaly_check = self._check_anomalies(df, price_cols)
        staleness_check = self._check_staleness(df, date_col)
        consistency_check = self._check_consistency(df, symbol_col, date_col)
        outlier_check = self._check_outliers(df, price_cols)

        # Calculate overall score
        total_checks = 5
        passed_checks = sum([
            missing_check['passed'],
            anomaly_check['passed'],
            staleness_check['passed'],
            consistency_check['passed'],
            outlier_check['passed']
        ])

        quality_score = (passed_checks / total_checks) * 100

        report = {
            'quality_score': quality_score,
            'total_issues': len(self.issues),
            'checks': {
                'missing_data': missing_check,
                'anomalies': anomaly_check,
                'staleness': staleness_check,
                'consistency': consistency_check,
                'outliers': outlier_check
            },
            'issues': self.issues,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Display results
        self._display_report(report)

        # Send alert if critical issues
        if quality_score < 80 and self.alert_system:
            self.alert_system.alert_data_quality_issue(
                issue_type="Multiple data quality issues",
                affected_symbols=[],
                details={
                    'Quality Score': f"{quality_score:.1f}%",
                    'Total Issues': len(self.issues),
                    'Critical': sum(1 for i in self.issues if i['severity'] == 'CRITICAL')
                }
            )

        return report

    def _check_missing_data(self, df: pd.DataFrame) -> Dict:
        """Check for missing data"""
        print("\n[1/5] Checking for missing data...")

        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100

        critical_missing = missing_pct[missing_pct > 10]

        if len(critical_missing) > 0:
            for col, pct in critical_missing.items():
                self.issues.append({
                    'check': 'missing_data',
                    'severity': 'CRITICAL' if pct > 20 else 'WARNING',
                    'column': col,
                    'message': f"Column '{col}' has {pct:.1f}% missing values",
                    'count': int(missing_counts[col])
                })

            print(f"  [WARNING] Found {len(critical_missing)} columns with >10% missing")
            passed = False
        else:
            print("  [OK] No critical missing data")
            passed = True

        return {
            'passed': passed,
            'total_missing': int(df.isnull().sum().sum()),
            'columns_with_missing': int((missing_counts > 0).sum())
        }

    def _check_anomalies(self, df: pd.DataFrame, price_cols: List[str]) -> Dict:
        """Check for price anomalies"""
        print("\n[2/5] Checking for anomalies...")

        anomalies_found = 0

        for col in price_cols:
            if col not in df.columns:
                continue

            # Check for zero/negative prices
            zero_count = (df[col] <= 0).sum()
            if zero_count > 0:
                self.issues.append({
                    'check': 'anomalies',
                    'severity': 'CRITICAL',
                    'column': col,
                    'message': f"Found {zero_count} zero/negative values in '{col}'",
                    'count': int(zero_count)
                })
                anomalies_found += zero_count

            # Check for extreme price spikes (>50% change)
            if len(df) > 1:
                pct_change = df[col].pct_change().abs()
                extreme_changes = (pct_change > 0.5).sum()

                if extreme_changes > 0:
                    self.issues.append({
                        'check': 'anomalies',
                        'severity': 'WARNING',
                        'column': col,
                        'message': f"Found {extreme_changes} extreme price changes (>50%) in '{col}'",
                        'count': int(extreme_changes)
                    })
                    anomalies_found += extreme_changes

        if anomalies_found > 0:
            print(f"  [WARNING] Found {anomalies_found} anomalies")
            passed = False
        else:
            print("  [OK] No anomalies detected")
            passed = True

        return {
            'passed': passed,
            'anomalies_found': anomalies_found
        }

    def _check_staleness(self, df: pd.DataFrame, date_col: str) -> Dict:
        """Check if data is up-to-date"""
        print("\n[3/5] Checking data staleness...")

        if date_col not in df.columns:
            print("  [SKIP] No date column found")
            return {'passed': True, 'note': 'Skipped - no date column'}

        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                dates = pd.to_datetime(df[date_col])
            else:
                dates = df[date_col]

            latest_date = dates.max()
            days_old = (datetime.now() - latest_date).days

            if days_old > 7:
                self.issues.append({
                    'check': 'staleness',
                    'severity': 'CRITICAL',
                    'message': f"Data is {days_old} days old (latest: {latest_date.strftime('%Y-%m-%d')})",
                    'days_old': days_old
                })
                print(f"  [CRITICAL] Data is {days_old} days old!")
                passed = False

            elif days_old > 3:
                self.issues.append({
                    'check': 'staleness',
                    'severity': 'WARNING',
                    'message': f"Data is {days_old} days old",
                    'days_old': days_old
                })
                print(f"  [WARNING] Data is {days_old} days old")
                passed = False

            else:
                print(f"  [OK] Data is fresh ({days_old} days old)")
                passed = True

            return {
                'passed': passed,
                'latest_date': latest_date.strftime('%Y-%m-%d'),
                'days_old': days_old
            }

        except Exception as e:
            print(f"  [ERROR] Failed to check staleness: {str(e)}")
            return {'passed': True, 'error': str(e)}

    def _check_consistency(self, df: pd.DataFrame,
                           symbol_col: str, date_col: str) -> Dict:
        """Check for data consistency issues"""
        print("\n[4/5] Checking data consistency...")

        issues_found = 0

        # Check for duplicates
        if symbol_col in df.columns and date_col in df.columns:
            duplicates = df.duplicated(subset=[symbol_col, date_col], keep=False).sum()

            if duplicates > 0:
                self.issues.append({
                    'check': 'consistency',
                    'severity': 'WARNING',
                    'message': f"Found {duplicates} duplicate (symbol, date) pairs",
                    'count': int(duplicates)
                })
                issues_found += 1
                print(f"  [WARNING] Found {duplicates} duplicate entries")

        # Check for price consistency (high >= low, close between high and low)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            invalid_high_low = (df['high'] < df['low']).sum()
            invalid_close = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()

            if invalid_high_low > 0:
                self.issues.append({
                    'check': 'consistency',
                    'severity': 'CRITICAL',
                    'message': f"Found {invalid_high_low} rows where high < low",
                    'count': int(invalid_high_low)
                })
                issues_found += 1

            if invalid_close > 0:
                self.issues.append({
                    'check': 'consistency',
                    'severity': 'WARNING',
                    'message': f"Found {invalid_close} rows where close is outside [low, high]",
                    'count': int(invalid_close)
                })
                issues_found += 1

        if issues_found == 0:
            print("  [OK] Data is consistent")
            passed = True
        else:
            print(f"  [WARNING] Found {issues_found} consistency issues")
            passed = False

        return {
            'passed': passed,
            'issues_found': issues_found
        }

    def _check_outliers(self, df: pd.DataFrame, price_cols: List[str]) -> Dict:
        """Check for statistical outliers"""
        print("\n[5/5] Checking for outliers...")

        outliers_found = 0

        for col in price_cols:
            if col not in df.columns:
                continue

            # Use IQR method to detect outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            if outliers > len(df) * 0.05:  # More than 5% outliers
                self.issues.append({
                    'check': 'outliers',
                    'severity': 'WARNING',
                    'column': col,
                    'message': f"Column '{col}' has {outliers} outliers ({outliers/len(df)*100:.1f}%)",
                    'count': int(outliers)
                })
                outliers_found += outliers

        if outliers_found > 0:
            print(f"  [WARNING] Found {outliers_found} outliers")
            passed = False
        else:
            print("  [OK] No excessive outliers")
            passed = True

        return {
            'passed': passed,
            'outliers_found': outliers_found
        }

    def _display_report(self, report: Dict):
        """Display quality report"""
        print("\n" + "="*80)
        print("DATA QUALITY REPORT")
        print("="*80)

        score = report['quality_score']

        if score >= 90:
            status = "[EXCELLENT]"
        elif score >= 80:
            status = "[GOOD]"
        elif score >= 70:
            status = "[FAIR]"
        else:
            status = "[POOR]"

        print(f"\nOverall Quality Score: {score:.1f}% {status}")
        print(f"Total Issues Found: {report['total_issues']}")

        if report['issues']:
            print("\n" + "-"*80)
            print("ISSUES DETECTED")
            print("-"*80)

            critical = [i for i in report['issues'] if i['severity'] == 'CRITICAL']
            warnings = [i for i in report['issues'] if i['severity'] == 'WARNING']

            if critical:
                print(f"\nCRITICAL ({len(critical)}):")
                for issue in critical:
                    print(f"  - {issue['message']}")

            if warnings:
                print(f"\nWARNINGS ({len(warnings)}):")
                for issue in warnings:
                    print(f"  - {issue['message']}")

        print("\n" + "="*80)

    def auto_fix_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to automatically fix common data issues"""
        print("\n" + "="*80)
        print("AUTO-FIXING DATA ISSUES")
        print("="*80)

        df_fixed = df.copy()
        fixes_applied = 0

        # Fix 1: Remove duplicate rows
        duplicates = df_fixed.duplicated().sum()
        if duplicates > 0:
            df_fixed = df_fixed.drop_duplicates()
            print(f"\n[FIX] Removed {duplicates} duplicate rows")
            fixes_applied += 1

        # Fix 2: Fill missing values with forward fill
        missing = df_fixed.isnull().sum().sum()
        if missing > 0:
            df_fixed = df_fixed.fillna(method='ffill').fillna(method='bfill')
            print(f"[FIX] Filled {missing} missing values using forward/backward fill")
            fixes_applied += 1

        # Fix 3: Remove rows with zero/negative prices
        if 'close' in df_fixed.columns:
            bad_prices = (df_fixed['close'] <= 0).sum()
            if bad_prices > 0:
                df_fixed = df_fixed[df_fixed['close'] > 0]
                print(f"[FIX] Removed {bad_prices} rows with invalid prices")
                fixes_applied += 1

        if fixes_applied == 0:
            print("\nNo automatic fixes needed")
        else:
            print(f"\nTotal fixes applied: {fixes_applied}")
            print(f"Rows after fixing: {len(df_fixed)} (was {len(df)})")

        return df_fixed


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    print("Data Quality Monitor - Test Mode")
    print("="*80)

    # Create sample data with issues
    np.random.seed(42)

    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = {
        'symbol': ['AAPL'] * 100,
        'date': dates,
        'open': np.random.uniform(150, 160, 100),
        'high': np.random.uniform(155, 165, 100),
        'low': np.random.uniform(145, 155, 100),
        'close': np.random.uniform(150, 160, 100),
        'volume': np.random.uniform(1e6, 1e7, 100)
    }

    df = pd.DataFrame(data)

    # Introduce some issues for testing
    df.loc[10, 'close'] = np.nan  # Missing data
    df.loc[20, 'close'] = 0  # Zero price
    df.loc[30, 'close'] = 500  # Extreme outlier
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # Duplicate

    print("\nTest data created with intentional issues:")
    print(f"- {df.shape[0]} rows, {df.shape[1]} columns")
    print("- 1 missing value")
    print("- 1 zero price")
    print("- 1 extreme outlier")
    print("- 1 duplicate row")

    # Run quality check
    monitor = DataQualityMonitor(use_alerts=False)
    report = monitor.check_data_quality(df)

    # Try auto-fix
    df_fixed = monitor.auto_fix_issues(df)

    # Check again
    print("\n" + "="*80)
    print("CHECKING FIXED DATA")
    print("="*80)

    report2 = monitor.check_data_quality(df_fixed)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
