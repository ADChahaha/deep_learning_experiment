$ErrorActionPreference = "Stop"
$Project = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet"
$RunDir = Join-Path $Project "runs\pfnet_4070_20260520_181211"
$PythonExe = "C:\Users\14195\miniconda3\envs\segexp\python.exe"

Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*CVPR2021_PFNet*train.py*" } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        "Stopped $($_.ProcessId) $($_.CommandLine)" | Add-Content -Encoding UTF8 (Join-Path $RunDir "background_status.txt")
    }

$OutLog = Join-Path $RunDir "train_background_stdout.log"
$ErrLog = Join-Path $RunDir "train_background_stderr.log"
Remove-Item -Force -ErrorAction SilentlyContinue $OutLog, $ErrLog
"BACKGROUND_START $(Get-Date -Format o)" | Add-Content -Encoding UTF8 (Join-Path $RunDir "background_status.txt")

Start-Process -FilePath $PythonExe `
    -ArgumentList @("train.py") `
    -WorkingDirectory $Project `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden

Write-Output "STARTED_BACKGROUND"
Write-Output "OUT=$OutLog"
Write-Output "ERR=$ErrLog"
