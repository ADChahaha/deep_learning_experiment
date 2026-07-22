$ErrorActionPreference = "Stop"
$Project = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet"
$RunDir = Join-Path $Project "runs\pfnet_4070_20260520_181211"
$PythonExe = "C:\Users\14195\miniconda3\envs\segexp\python.exe"
$Bat = Join-Path $Project "run_train_background.bat"
$TaskName = "PFNet_Train_Background_181211"
$OutLog = Join-Path $RunDir "train_bat_stdout.log"
$ErrLog = Join-Path $RunDir "train_bat_stderr.log"

Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*CVPR2021_PFNet*train.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

@"
@echo off
cd /d "$Project"
echo BAT_START %date% %time% > "$RunDir\train_bat_status.txt"
"$PythonExe" train.py > "$OutLog" 2> "$ErrLog"
echo BAT_EXIT %errorlevel% %date% %time% >> "$RunDir\train_bat_status.txt"
"@ | Set-Content -Encoding ASCII $Bat

Remove-Item -Force -ErrorAction SilentlyContinue $OutLog, $ErrLog
cmd.exe /c "schtasks /Delete /TN  /F 2>nul" | Out-Null
$Start = (Get-Date).AddMinutes(1).ToString("HH:mm")
schtasks /Create /TN $TaskName /TR "`"$Bat`"" /SC ONCE /ST $Start /F | Write-Output
schtasks /Run /TN $TaskName | Write-Output
Write-Output "TASK=$TaskName"
Write-Output "BAT=$Bat"
Write-Output "OUT=$OutLog"
Write-Output "ERR=$ErrLog"
