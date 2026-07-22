$ErrorActionPreference = "Stop"
$Project = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet"
$RunDir = Join-Path $Project "runs\pfnet_4070_20260520_181211"
$PythonExe = "C:\Users\14195\miniconda3\envs\segexp\python.exe"

Push-Location $Project
try {
    (Get-Content "train.py") `
        -replace "device_ids = \[[0-9]+\]", "device_ids = [0]" `
        -replace "'train_batch_size': 16", "'train_batch_size': 8" |
        Set-Content -Encoding UTF8 "train.py"

    (Get-Content "infer.py") `
        -replace "device_ids = \[[0-9]+\]", "device_ids = [0]" `
        -replace "\s*\('NC4K', nc4k_path\)\r?\n", "" |
        Set-Content -Encoding UTF8 "infer.py"

    (Get-Content "backbone\resnet\resnet.py") `
        -replace "torch\.load\(backbone_path\)", "torch.load(backbone_path, weights_only=False)" |
        Set-Content -Encoding UTF8 "backbone\resnet\resnet.py"

    if (Test-Path "resnet50-19c8e357.pth") {
        Copy-Item -Force "resnet50-19c8e357.pth" "backbone\resnet\resnet50-19c8e357.pth"
    }

    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $RunDir "FAILED.txt")
    "RETRY_TRAIN $(Get-Date -Format o)" | Add-Content -Encoding UTF8 (Join-Path $RunDir "retry_status.txt")
    Select-String -Path "train.py","infer.py","backbone\resnet\resnet.py" -Pattern "device_ids|train_batch_size|torch.load" |
        Out-File -Encoding UTF8 (Join-Path $RunDir "retry_patch_check.txt")

    $TrainLog = Join-Path $RunDir "train_retry_stdout.log"
    Start-Process powershell `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "cd '$Project'; & '$PythonExe' train.py 2>&1 | Tee-Object -FilePath '$TrainLog'") `
        -WorkingDirectory $Project `
        -WindowStyle Hidden

    Write-Output "STARTED_RETRY"
    Write-Output "PROJECT=$Project"
    Write-Output "RUN_DIR=$RunDir"
    Write-Output "TRAIN_LOG=$TrainLog"
} finally {
    Pop-Location
}
