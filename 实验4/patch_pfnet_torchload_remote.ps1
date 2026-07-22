$Project = "C:\Users\14195\AppData\Local\Temp\pfnet_launch\pfnet_asus_package\CVPR2021_PFNet"
$Resnet = Join-Path $Project "backbone\resnet\resnet.py"
$Infer = Join-Path $Project "infer.py"
$Train = Join-Path $Project "train.py"

(Get-Content $Resnet) `
    -replace "torch\.load\(backbone_path\)", "torch.load(backbone_path, weights_only=False)" |
    Set-Content -Encoding UTF8 $Resnet

(Get-Content $Infer) `
    -replace "torch\.load\('PFNet\.pth'\)", "torch.load('PFNet.pth', weights_only=False)" |
    Set-Content -Encoding UTF8 $Infer

(Get-Content $Train) `
    -replace "torch\.load\(os\.path\.join\(ckpt_path, exp_name, args\['snapshot'\] \+ '\.pth'\)\)", "torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), weights_only=False)" |
    Set-Content -Encoding UTF8 $Train

Select-String -Path $Resnet, $Infer, $Train -Pattern "torch.load"
