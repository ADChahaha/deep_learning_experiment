$ErrorActionPreference = "SilentlyContinue"
$candidates = @(
    "C:\ProgramData\Anaconda3\Scripts\conda.exe",
    "C:\ProgramData\Miniconda3\Scripts\conda.exe",
    "C:\Users\14195\anaconda3\Scripts\conda.exe",
    "C:\Users\14195\miniconda3\Scripts\conda.exe",
    "C:\Users\14195\mambaforge\Scripts\conda.exe",
    "C:\Users\14195\miniforge3\Scripts\conda.exe",
    "D:\Anaconda3\Scripts\conda.exe",
    "D:\Miniconda3\Scripts\conda.exe",
    "D:\miniconda3\Scripts\conda.exe",
    "D:\anaconda3\Scripts\conda.exe",
    "F:\Anaconda3\Scripts\conda.exe",
    "F:\Miniconda3\Scripts\conda.exe",
    "F:\miniconda3\Scripts\conda.exe",
    "F:\anaconda3\Scripts\conda.exe"
)
$found = @()
foreach ($p in $candidates) {
    if (Test-Path $p) { $found += $p }
}
foreach ($root in @("C:\Users\14195", "D:\", "F:\")) {
    if (Test-Path $root) {
        $found += Get-ChildItem -Path $root -Filter conda.exe -Recurse -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    }
}
$found = $found | Sort-Object -Unique
Write-Output "=== CONDA EXE ==="
$found
foreach ($conda in $found) {
    Write-Output "=== ENVS FROM $conda ==="
    & $conda env list
    $json = & $conda env list --json | ConvertFrom-Json
    foreach ($envPath in $json.envs) {
        $py = Join-Path $envPath "python.exe"
        if (Test-Path $py) {
            Write-Output "=== PYTHON $py ==="
            & $py -c "import sys; print(sys.executable); import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')" 2>&1
            & $py -c "import torchvision; print('torchvision', torchvision.__version__)" 2>&1
            & $py -c "import tensorboardX; print('tensorboardX ok')" 2>&1
        }
    }
}
