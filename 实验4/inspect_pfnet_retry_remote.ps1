$Project = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet"
$RunDir = Join-Path $Project "runs\pfnet_4070_20260520_181211"
Write-Output "=== RUN FILES ==="
Get-ChildItem $RunDir | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
Write-Output "=== STDOUT ==="
Get-Content -ErrorAction SilentlyContinue (Join-Path $RunDir "train_retry_stdout.log") -Tail 60
Write-Output "=== STDERR ==="
Get-Content -ErrorAction SilentlyContinue (Join-Path $RunDir "train_retry_stderr.log") -Tail 60
Write-Output "=== PROCESSES ==="
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -in @("powershell.exe", "python.exe", "cmd.exe") } |
    Select-Object ProcessId, ParentProcessId, Name, CommandLine |
    Format-List
Write-Output "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
