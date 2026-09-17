# Registers (or replaces) StockPredict_IntradayGuard: trading/intraday_guard.py
# every 5 minutes between 14:00 and 22:00 UK on weekdays (covers 09:30-16:00
# ET in both DST regimes; the script itself exits at once outside the US
# session or on a closed day). S4U like the other tasks. Needs an elevated
# shell:
#   Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File C:\Trading_System\scripts\register_intraday_guard.ps1'
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'trading_logs\register_intraday_guard.log'
$user = "$env:USERDOMAIN\$env:USERNAME"
try {
    Get-Process -Name schtasks -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    $bat = Join-Path $root 'scripts\run_intraday_guard.bat'
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
        -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 14:00
    $trigger.Repetition = (New-ScheduledTaskTrigger -Once -At 14:00 -RepetitionInterval (New-TimeSpan -Minutes 5) `
        -RepetitionDuration (New-TimeSpan -Hours 8)).Repetition
    $action = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $root
    Register-ScheduledTask -TaskName 'StockPredict_IntradayGuard' -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings -Force | Out-Null
    "$(Get-Date -Format s) registered StockPredict_IntradayGuard (Mon-Fri 14:00-22:00 UK every 5 min, S4U)" | Out-File $log -Append -Encoding utf8
    "OK" | Out-File $log -Append -Encoding utf8
} catch {
    "$(Get-Date -Format s) FAILED: $($_.Exception.Message)" | Out-File $log -Append -Encoding utf8
    exit 1
}
