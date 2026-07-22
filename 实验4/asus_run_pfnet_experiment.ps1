$ErrorActionPreference = "Stop"

$RunId = "pfnet_4070_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$WorkRoot = Join-Path $env:TEMP $RunId
$SourceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceProject = Join-Path $SourceRoot "CVPR2021_PFNet"
$Project = Join-Path $WorkRoot "CVPR2021_PFNet"
$RunDir = Join-Path $Project ("runs\" + $RunId)
$ReturnZip = Join-Path $env:TEMP ($RunId + ".zip")
$PythonExe = "C:\Users\14195\miniconda3\envs\segexp\python.exe"

Start-Transcript -Path (Join-Path $env:TEMP ($RunId + "_transcript.txt")) -Force | Out-Null

function Run-Logged {
    param(
        [Parameter(Mandatory=$true)][string]$Command,
        [Parameter(Mandatory=$true)][string]$LogFile,
        [string]$WorkingDirectory = $Project
    )
    Push-Location $WorkingDirectory
    try {
        cmd /c "$Command 2>&1" | Tee-Object -FilePath $LogFile
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $Command"
        }
    } finally {
        Pop-Location
    }
}

try {
    "START $(Get-Date -Format o)" | Out-File -Encoding UTF8 (Join-Path $env:TEMP "pfnet_status.txt")
    New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
    "COPY_PROJECT $(Get-Date -Format o)" | Add-Content -Encoding UTF8 (Join-Path $env:TEMP "pfnet_status.txt")
    Copy-Item -Recurse -Force $SourceProject $WorkRoot
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    "RUN_ID=$RunId" | Out-File -Encoding UTF8 (Join-Path $RunDir "run_id.txt")
    "WORK_ROOT=$WorkRoot" | Out-File -Encoding UTF8 (Join-Path $RunDir "paths.txt")
    "SOURCE_ROOT=$SourceRoot" | Add-Content -Encoding UTF8 (Join-Path $RunDir "paths.txt")

    Push-Location $Project
    nvidia-smi | Out-File -Encoding UTF8 (Join-Path $RunDir "nvidia-smi.txt")
    & $PythonExe --version 2>&1 | Out-File -Encoding UTF8 (Join-Path $RunDir "python.txt")
    & $PythonExe -c "import sys, torch, torchvision; print('python', sys.executable); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count()); print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" 2>&1 | Out-File -Encoding UTF8 (Join-Path $RunDir "torch_env.txt")
    & $PythonExe -m pip freeze | Out-File -Encoding UTF8 (Join-Path $RunDir "pip_freeze.txt")

    Expand-Archive -Force "TrainDataset.zip" "dataset_raw"
    Expand-Archive -Force "TestDataset.zip" "dataset_raw"

    $DataRoot = Join-Path (Split-Path -Parent $Project) "data\NEW"
    New-Item -ItemType Directory -Force -Path "$DataRoot\train\image", "$DataRoot\train\mask" | Out-Null
    Copy-Item "dataset_raw\TrainDataset\Imgs\*.jpg" "$DataRoot\train\image\" -Force
    Copy-Item "dataset_raw\TrainDataset\GT\*.png" "$DataRoot\train\mask\" -Force

    foreach ($Dataset in @("CHAMELEON", "CAMO", "COD10K")) {
        New-Item -ItemType Directory -Force -Path "$DataRoot\test\$Dataset\image", "$DataRoot\test\$Dataset\mask" | Out-Null
        Copy-Item "dataset_raw\TestDataset\$Dataset\Imgs\*.jpg" "$DataRoot\test\$Dataset\image\" -Force
        Copy-Item "dataset_raw\TestDataset\$Dataset\GT\*.png" "$DataRoot\test\$Dataset\mask\" -Force
    }

    "train_images $((Get-ChildItem "$DataRoot\train\image" -Filter *.jpg).Count)" | Out-File -Encoding UTF8 (Join-Path $RunDir "data_count.txt")
    "train_masks $((Get-ChildItem "$DataRoot\train\mask" -Filter *.png).Count)" | Add-Content -Encoding UTF8 (Join-Path $RunDir "data_count.txt")
    foreach ($Dataset in @("CHAMELEON", "CAMO", "COD10K")) {
        "$Dataset images $((Get-ChildItem "$DataRoot\test\$Dataset\image" -Filter *.jpg).Count)" | Add-Content -Encoding UTF8 (Join-Path $RunDir "data_count.txt")
        "$Dataset masks $((Get-ChildItem "$DataRoot\test\$Dataset\mask" -Filter *.png).Count)" | Add-Content -Encoding UTF8 (Join-Path $RunDir "data_count.txt")
    }
    tree (Split-Path -Parent $DataRoot) /F | Out-File -Encoding UTF8 (Join-Path $RunDir "data_tree.txt")

    if (Test-Path "resnet50-19c8e357.pth") {
        Copy-Item -Force "resnet50-19c8e357.pth" "backbone\resnet\resnet50-19c8e357.pth"
    }

    (Get-Content "train.py") `
        -replace "device_ids = \\[1\\]", "device_ids = [0]" `
        -replace "'train_batch_size': 16", "'train_batch_size': 8" `
        -replace "num_workers=16", "num_workers=8" |
        Set-Content -Encoding UTF8 "train.py"
    (Get-Content "infer.py") `
        -replace "device_ids = \\[1\\]", "device_ids = [0]" `
        -replace "\\s*\\('NC4K', nc4k_path\\)\\r?\\n", "" |
        Set-Content -Encoding UTF8 "infer.py"
    (Get-Content "eval\main.m") `
        -replace "names = dir\\(\\['~/data/NEW/test/' dataset '/mask/\\*.png'\\]\\);", "names = dir([maskpath '*.png']);" |
        Set-Content -Encoding UTF8 "eval\main.m"

    $Snapshot = Join-Path $RunDir "code_snapshot"
    New-Item -ItemType Directory -Force -Path $Snapshot | Out-Null
    Copy-Item -Force "train.py", "infer.py", "config.py", "PFNet.py", "loss.py", "datasets.py", "joint_transforms.py" $Snapshot
    New-Item -ItemType Directory -Force -Path (Join-Path $Snapshot "backbone\resnet"), (Join-Path $Snapshot "eval") | Out-Null
    Copy-Item -Force "backbone\resnet\resnet.py" (Join-Path $Snapshot "backbone\resnet\resnet.py")
    Copy-Item -Force "eval\main.m" (Join-Path $Snapshot "eval\main.m")

    Run-Logged -Command "`"$PythonExe`" train.py" -LogFile (Join-Path $RunDir "train_stdout.log")
    Copy-Item -Recurse -Force "ckpt\PFNet" (Join-Path $RunDir "ckpt_PFNet")

    Copy-Item -Force "ckpt\PFNet\45.pth" "PFNet.pth"
    Run-Logged -Command "`"$PythonExe`" infer.py" -LogFile (Join-Path $RunDir "infer_stdout.log")
    Copy-Item -Recurse -Force "results\PFNet" (Join-Path $RunDir "results_PFNet")

    $Matlab = Get-Command matlab -ErrorAction SilentlyContinue
    if ($null -ne $Matlab) {
        Run-Logged -Command "matlab -batch main" -LogFile (Join-Path $RunDir "eval_stdout.log") -WorkingDirectory (Join-Path $Project "eval")
        if (Test-Path "eval\mat\PFNet") {
            Copy-Item -Recurse -Force "eval\mat\PFNet" (Join-Path $RunDir "eval_mat_PFNet")
        }
    } else {
        "MATLAB not found; evaluation skipped." | Out-File -Encoding UTF8 (Join-Path $RunDir "eval_stdout.log")
    }
} catch {
    $_ | Out-String | Out-File -Encoding UTF8 (Join-Path $env:TEMP "pfnet_failed.txt")
    if (Test-Path $RunDir) {
        $_ | Out-String | Out-File -Encoding UTF8 (Join-Path $RunDir "FAILED.txt")
    }
    throw
} finally {
    Pop-Location -ErrorAction SilentlyContinue
    if (Test-Path $ReturnZip) {
        Remove-Item -Force $ReturnZip
    }
    if (Test-Path $RunDir) {
        Compress-Archive -Force -Path $RunDir -DestinationPath $ReturnZip
    }
    "Packaged run artifact: $ReturnZip"
    Stop-Transcript | Out-Null
}
