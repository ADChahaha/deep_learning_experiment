$ErrorActionPreference = "SilentlyContinue"
$TaskName = "PFNet_Run_20260520_181210"
$WorkRoot = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211"
$RunRoot = Join-Path $WorkRoot "CVPR2021_PFNet\runs\pfnet_4070_20260520_181211"
Write-Output "=== TASKS ==="
schtasks /Query /TN $TaskName /V /FO LIST
Write-Output "=== STATUS ==="
Get-Content "$env:TEMP\pfnet_status.txt"
Get-Content "$env:TEMP\pfnet_failed.txt"
Write-Output "=== WORKDIRS ==="
Get-ChildItem $env:TEMP -Directory -Filter "pfnet_4070_*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object FullName, LastWriteTime |
    Format-Table -AutoSize
Write-Output "=== KEY FILES ==="
Get-ChildItem -Recurse $WorkRoot |
    Where-Object { @("run_id.txt", "data_count.txt", "train_stdout.log", "infer_stdout.log", "eval_stdout.log", "FAILED.txt") -contains $_.Name } |
    Select-Object FullName, Length, LastWriteTime |
    Format-Table -AutoSize
Write-Output "=== TRAIN TAIL ==="
$TrainLog = Get-ChildItem -Recurse $WorkRoot -Filter "train_stdout.log" | Select-Object -First 1
if ($TrainLog) {
    Get-Content $TrainLog.FullName -Tail 20
}
Write-Output "=== TORCH ENV ==="
Get-Content (Join-Path $RunRoot "torch_env.txt")
Write-Output "=== DATA COUNT ==="
Get-Content (Join-Path $RunRoot "data_count.txt")
Write-Output "=== PROCESSES ==="
Get-Process powershell, python |
    Select-Object Id, ProcessName, CPU, StartTime |
    Format-Table -AutoSize
Write-Output "=== ZIPS ==="
Get-ChildItem $env:TEMP -Filter "pfnet_4070_*.zip" |
    Select-Object FullName, Length, LastWriteTime |
    Format-Table -AutoSize
