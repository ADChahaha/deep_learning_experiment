$ErrorActionPreference = "Stop"
$Inbox = Join-Path $env:TEMP "pfnet_inbox"
$Zip = Join-Path $Inbox "pfnet_asus_package.zip"
$Launch = Join-Path $env:TEMP "pfnet_launch"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Launch
New-Item -ItemType Directory -Force -Path $Launch | Out-Null
Expand-Archive -Force $Zip $Launch
$Script = Join-Path $Launch "pfnet_asus_package\asus_run_pfnet_experiment.ps1"
$OutLog = Join-Path $env:TEMP "pfnet_launcher.log"
$ErrLog = Join-Path $env:TEMP "pfnet_launcher.err"
Start-Process powershell `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Script) `
    -WorkingDirectory (Split-Path -Parent $Script) `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden
Write-Output "STARTED $Script"
Write-Output "LAUNCH_LOG $OutLog"
Write-Output "LAUNCH_ERR $ErrLog"
