$ErrorActionPreference = "Stop"
$Project = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet"
$RunDir = Join-Path $Project "runs\pfnet_4070_20260520_181211"
$PythonExe = "C:\Users\14195\miniconda3\envs\segexp\python.exe"
$OutLog = Join-Path $RunDir "train_retry_stdout.log"
$ErrLog = Join-Path $RunDir "train_retry_stderr.log"
$Cmd = "/c cd /d `"$Project`" && `"$PythonExe`" train.py"
Remove-Item -Force -ErrorAction SilentlyContinue $OutLog, $ErrLog
Start-Process cmd.exe `
    -ArgumentList $Cmd `
    -WorkingDirectory $Project `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden
Write-Output "STARTED_CMD"
Write-Output "OUT=$OutLog"
Write-Output "ERR=$ErrLog"
