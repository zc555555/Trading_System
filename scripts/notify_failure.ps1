# Desktop toast for scheduled-run failures (Windows 11 Home has no msg.exe).
# Usage: powershell -NoProfile -File notify_failure.ps1 "title" "body"
param([string]$Title = "StockPredict", [string]$Body = "run failed")
try {
    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
    $t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
        [Windows.UI.Notifications.ToastTemplateType]::ToastText02)
    $t.GetElementsByTagName('text').Item(0).InnerText = $Title
    $t.GetElementsByTagName('text').Item(1).InnerText = $Body
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier(
        'StockPredict').Show([Windows.UI.Notifications.ToastNotification]::new($t))
} catch {
    # best-effort only; FAILURES.log is the durable record
}
