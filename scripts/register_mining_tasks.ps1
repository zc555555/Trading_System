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
    # a schtasks.exe left waiting for a password by an earlier attempt would block this shell too
    Get-Process -Name schtasks -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 12) `
        -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

    # monthly trigger: PS 5.1 has no -Monthly trigger, the CIM monthly class
    # lacks MonthsOfYear and schtasks.exe prompts for a password, so the
    # task is registered from XML (ScheduleByMonth, S4U principal)
    $bat1 = Join-Path $root 'scripts\run_refresh_sources_scheduled.bat'
    $months = (@('January','February','March','April','May','June','July','August','September','October','November','December') | ForEach-Object { "<$_ />" }) -join ''
    $xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Monthly refresh of the mining DSL external sources (SEC insider, FINRA short interest, GDELT, screen caches). Not production.</Description></RegistrationInfo>
  <Triggers><CalendarTrigger><StartBoundary>2026-09-25T19:00:00</StartBoundary><Enabled>true</Enabled>
    <ScheduleByMonth><DaysOfMonth><Day>25</Day></DaysOfMonth><Months>$months</Months></ScheduleByMonth></CalendarTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>$user</UserId><LogonType>S4U</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable><ExecutionTimeLimit>PT12H</ExecutionTimeLimit><Enabled>true</Enabled></Settings>
  <Actions Context="Author"><Exec><Command>$bat1</Command><WorkingDirectory>$root</WorkingDirectory></Exec></Actions>
</Task>
"@
    Register-ScheduledTask -TaskName 'StockPredict_MiningSourcesRefresh' -Xml $xml -Force | Out-Null
    "$(Get-Date -Format s) registered StockPredict_MiningSourcesRefresh (monthly 25th 19:00, S4U, from XML)" | Out-File $log -Append -Encoding utf8

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
