$Project = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet"
$RunDir = Join-Path $Project "runs\pfnet_4070_20260520_181211"

$Config = Join-Path $Project "config.py"
$configText = @'
"""
 @Time    : 2021/7/6 09:46
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn

 @Project : CVPR2021_PFNet
 @File    : config.py
 @Function: Configuration

"""
import os

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

datasets_root = './dataset_raw'

cod_training_root = os.path.join(datasets_root, 'TrainDataset')

chameleon_path = os.path.join(datasets_root, 'TestDataset/CHAMELEON')
camo_path = os.path.join(datasets_root, 'TestDataset/CAMO')
cod10k_path = os.path.join(datasets_root, 'TestDataset/COD10K')
nc4k_path = os.path.join(datasets_root, 'TestDataset/NC4K')
'@
Set-Content -Encoding UTF8 $Config $configText

$Datasets = Join-Path $Project "datasets.py"
$datasetsText = @'
import os
import os.path
import torch.utils.data as data
from PIL import Image

def _first_existing(root, names):
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("None of {} exists under {}".format(names, root))

def make_dataset(root):
    image_path = _first_existing(root, ['image', 'Imgs'])
    mask_path = _first_existing(root, ['mask', 'GT'])
    image_names = {os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')}
    mask_names = {os.path.splitext(f)[0] for f in os.listdir(mask_path) if f.endswith('.png')}
    img_list = sorted(image_names & mask_names)
    missing_images = sorted(mask_names - image_names)
    missing_masks = sorted(image_names - mask_names)
    if missing_images or missing_masks:
        print("Dataset warning: {} masks without images, {} images without masks under {}".format(
            len(missing_images), len(missing_masks), root))
    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]

class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
'@
Set-Content -Encoding UTF8 $Datasets $datasetsText
Copy-Item -Force $Config (Join-Path $RunDir "config_dataset_raw.py")
Copy-Item -Force $Datasets (Join-Path $RunDir "datasets_dataset_raw.py")
& "C:\Users\14195\miniconda3\envs\segexp\python.exe" -c "from datasets import ImageFolder; from config import cod_training_root; d=ImageFolder(cod_training_root); print('effective_train_set', len(d)); print(d.imgs[:3])" 2>&1 |
    Tee-Object -FilePath (Join-Path $RunDir "dataset_raw_check.txt")
