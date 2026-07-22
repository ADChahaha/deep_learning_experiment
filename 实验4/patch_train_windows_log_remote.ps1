$p = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet\train.py"
(Get-Content $p) `
    -replace "str\(datetime\.datetime\.now\(\)\) \+ '\.txt'", "datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.txt'" |
    Set-Content -Encoding UTF8 $p
Select-String -Path $p -Pattern "log_path|strftime|device_ids|train_batch_size"
