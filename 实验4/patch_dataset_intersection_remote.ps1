$Project = "C:\Users\14195\AppData\Local\Temp\pfnet_4070_20260520_181211\CVPR2021_PFNet"
$Datasets = Join-Path $Project "datasets.py"
$RunDir = Join-Path $Project "runs\pfnet_4070_20260520_181211"
$content = @'
"""
 @Time    : 2021/7/6 10:56
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn

 @Project : CVPR2021_PFNet
 @File    : datasets.py
 @Function: Datasets Processing

"""
import os
import os.path
import torch.utils.data as data
from PIL import Image

def make_dataset(root):
    image_path = os.path.join(root, 'image')
    mask_path = os.path.join(root, 'mask')
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
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
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
Set-Content -Encoding UTF8 $Datasets $content
Copy-Item -Force $Datasets (Join-Path $RunDir "datasets_patched.py")
Write-Output "PATCHED $Datasets"
