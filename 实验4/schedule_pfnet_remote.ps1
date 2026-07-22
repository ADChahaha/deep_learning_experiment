$ErrorActionPreference = "Stop"
$TaskName = "PFNet_Run_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$Script = "C:\Users\14195\AppData\Local\Temp\pfnet_launch\pfnet_asus_package\asus_run_pfnet_experiment.ps1"
$RunTime = (Get-Date).AddMinutes(1).ToString("HH:mm")
$TaskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Script`""
schtasks /Create /TN $TaskName /TR $TaskCommand /SC ONCE /ST $RunTime /F | Write-Output
schtasks /Run /TN $TaskName | Write-Output
Write-Output "TASK_NAME=$TaskName"
Write-Output "SCRIPT=$Script"
