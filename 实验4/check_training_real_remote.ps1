$Run = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet\runs\pfnet_4070_20260520_181211"
$Project = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet"
Write-Output "=== BAT STATUS ==="
Get-Content -ErrorAction SilentlyContinue (Join-Path $Run "train_bat_status.txt")
Write-Output "=== STDOUT SIZE ==="
Get-Item -ErrorAction SilentlyContinue (Join-Path $Run "train_bat_stdout.log") | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize
Write-Output "=== STDOUT TAIL ==="
Get-Content -ErrorAction SilentlyContinue (Join-Path $Run "train_bat_stdout.log") -Tail 120
Write-Output "=== STDERR SIZE ==="
Get-Item -ErrorAction SilentlyContinue (Join-Path $Run "train_bat_stderr.log") | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize
Write-Output "=== STDERR TAIL ==="
Get-Content -ErrorAction SilentlyContinue (Join-Path $Run "train_bat_stderr.log") -Tail 120
Write-Output "=== CKPT TXT ==="
$txt = Get-ChildItem -ErrorAction SilentlyContinue (Join-Path $Project "ckpt\PFNet") -Filter *.txt | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($txt) {
    $txt | Select-Object FullName, Length, LastWriteTime | Format-Table -AutoSize
    Get-Content $txt.FullName -Tail 30
}
Write-Output "=== TRAIN PROCESSES ==="
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*CVPR2021_PFNet*train.py*" -or $_.CommandLine -like "*segexp*train.py*" } |
    Select-Object ProcessId, ParentProcessId, Name, CommandLine |
    Format-List
Write-Output "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
