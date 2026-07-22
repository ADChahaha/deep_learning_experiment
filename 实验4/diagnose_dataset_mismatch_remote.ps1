$Data = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\data\NEW\train"
$Out = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet\runs\pfnet_4070_20260520_181211\dataset_mismatch.txt"
$imgs = Get-ChildItem "$Data\image" -Filter *.jpg | ForEach-Object { $_.BaseName } | Sort-Object
$masks = Get-ChildItem "$Data\mask" -Filter *.png | ForEach-Object { $_.BaseName } | Sort-Object
$imgSet = @{}
$maskSet = @{}
foreach ($x in $imgs) { $imgSet[$x] = $true }
foreach ($x in $masks) { $maskSet[$x] = $true }
$missingImages = @($masks | Where-Object { -not $imgSet.ContainsKey($_) })
$missingMasks = @($imgs | Where-Object { -not $maskSet.ContainsKey($_) })
"images=$($imgs.Count)" | Out-File -Encoding UTF8 $Out
"masks=$($masks.Count)" | Add-Content -Encoding UTF8 $Out
"missing_images=$($missingImages.Count)" | Add-Content -Encoding UTF8 $Out
$missingImages | Select-Object -First 50 | ForEach-Object { "missing_image_for_mask=$_" } | Add-Content -Encoding UTF8 $Out
"missing_masks=$($missingMasks.Count)" | Add-Content -Encoding UTF8 $Out
$missingMasks | Select-Object -First 50 | ForEach-Object { "missing_mask_for_image=$_" } | Add-Content -Encoding UTF8 $Out
Get-Content $Out
