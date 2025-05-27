import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

# 加载CIFAR-100数据集
dataset = CIFAR100(root='./data', train=True, download=True)

# CIFAR-100的100个细粒度类别名称
fine_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# 设置可视化参数
n_rows = 10  # 每行显示的图片数量
n_cols = 10  # 每列显示的图片数量
plt.figure(figsize=(20, 20))  # 调整图像大小

# 收集每个类别的样本
class_samples = {}
for idx, (image, label) in enumerate(dataset):
    print(label)
    if label not in class_samples:  
        class_samples[label] = image
    if len(class_samples) == 100:  # 收集到所有100个类别的样本后停止
        break

# 按类别顺序排列图像
images_in_order = [class_samples[i] for i in range(100)]

# 显示图像网格
for i in range(100):
    plt.subplot(n_rows, n_cols, i+1)
    plt.imshow(images_in_order[i])
    plt.title(f"{i}:{fine_labels[i]}", fontsize=8)
    plt.axis('off')
output_path = 'cifar100_classes.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()