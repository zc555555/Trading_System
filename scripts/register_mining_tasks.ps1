# Registers (or replaces) the two mining maintenance tasks, both S4U
# ("run whether user is logged on or not", no stored password) like the
# nightly/weekly trading tasks. Needs an elevated shell:
#   Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File C:\Trading_System\scripts\register_mining_tasks.ps1'
#
#   StockPredict_MiningSourcesRefresh  monthly, 25th 19:00 -> run_refresh_sources_scheduled.bat
#   StockPredict_SegmentReview         once on each rulebook REVIEW_SCHEDULE date 20:00 -> run_review_scheduled.bat
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root 'trading_logs\register_mining_tasks.log'
$user = "$env:USERDOMAIN\$env:USERNAME"
try {
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 12) `
        -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

    # monthly trigger: PS 5.1 has no -Monthly trigger and the CIM monthly class
    # lacks MonthsOfYear, so schtasks.exe does this one (/NP = no stored
    # password = run whether the user is logged on or not, like S4U)
    $bat1 = Join-Path $root 'scripts\run_refresh_sources_scheduled.bat'
    $out = & schtasks.exe /Create /F /TN 'StockPredict_MiningSourcesRefresh' /TR "`"$bat1`"" /SC MONTHLY /D 25 /ST 19:00 /RU $user /NP /RL LIMITED 2>&1
    if ($LASTEXITCODE -ne 0) { throw "schtasks: $out" }
    "$(Get-Date -Format s) registered StockPredict_MiningSourcesRefresh (monthly 25th 19:00, /NP): $out" | Out-File $log -Append -Encoding utf8

    # review dates = rulebook.REVIEW_SCHEDULE
    $dates = @('2026-11-15', '2027-05-15', '2027-11-15', '2028-05-15')
    $triggers = foreach ($d in $dates) { New-ScheduledTaskTrigger -Once -At (Get-Date "$d 20:00") }
    $a2 = New-ScheduledTaskAction -Execute (Join-Path $root 'scripts\run_review_scheduled.bat') -WorkingDirectory $root
    Register-ScheduledTask -TaskName 'StockPredict_SegmentReview' -Action $a2 -Trigger $triggers `
        -Principal $principal -Settings $settings -Force | Out-Null
    "$(Get-Date -Format s) registered StockPredict_SegmentReview ($($dates -join ', ') 20:00, S4U)" | Out-File $log -Append -Encoding utf8
    "OK" | Out-File $log -Append -Encoding utf8
} catch {
    "$(Get-Date -Format s) FAILED: $($_.Exception.Message)" | Out-File $log -Append -Encoding utf8
    exit 1
}
