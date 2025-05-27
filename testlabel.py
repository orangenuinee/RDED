import os
from torchvision.datasets import CIFAR100, CIFAR10

# 加载 CIFAR-100 类别映射
dataset = CIFAR10(root='./data', train=True, download=True)
name_to_idx = {name: idx for idx, name in enumerate(dataset.classes)}
'''
# 原始文件夹路径
src_dir = "/root/autodl-tmp/RDED/data/cifar100LT/train"

# 遍历并重命名
for old_name in os.listdir(src_dir):
    old_path = os.path.join(src_dir, old_name)
    
    if os.path.isdir(old_path):
        if old_name in name_to_idx:
            new_name = str(name_to_idx[old_name]).zfill(5)
            new_path = os.path.join(src_dir, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")
        else:
            print(f"Ignored: '{old_name}' (not a CIFAR-100 class)")

print("All folders renamed!")
'''