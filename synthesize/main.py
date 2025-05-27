import os
import random
import argparse
import collections
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from synthesize.utils import *
from validation.utils import ImageFolder
class CIFAR100LT(datasets.CIFAR100):
    def __init__(self, root, imbalance_rate=0.0005, nclass=100, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.imbalance_rate = imbalance_rate
        self.nclass = nclass
        np.random.seed(42)

    def balance_classes(self):
        """
        根据不平衡率调整类的样本数量,
        保持第一个类别与原始数量相同，后续类别按 imbalance_rate 递减
        """
        # 创建一个数组来存储每个类的索引
        class_indices = {i: [] for i in range(self.nclass)}
        
        for idx, target in enumerate(self.targets):
            class_indices[target].append(idx)

        # 保持第一个类别的样本数与原始一致
        first_class_size = len(class_indices[0])
        
        # 生成新的索引列表
        new_indices = []
        new_indices.extend(class_indices[0])  # 添加第一个类别的所有样本

        # 计算并选择后续类别的样本数量
        for cls_idx in range(1, self.nclass):
            num_samples = math.ceil(first_class_size * (self.imbalance_rate ** (cls_idx / (self.nclass - 1))))
            if num_samples > 0:  # 只处理大于0的样本数
                num_samples = min(num_samples, len(class_indices[cls_idx]))  # 确保不超过可用样本数量
                sampled_indices = np.random.choice(class_indices[cls_idx], size=num_samples, replace=False)
                new_indices.extend(sampled_indices)

        # 创建新的数据和目标基于新索引
        self.data = self.data[new_indices]
        self.targets = [self.targets[i] for i in new_indices]
        # 更新样本数量
        print(f'imbalance_rate: {self.imbalance_rate}')
        print(f'新的数据集大小: {len(self.data)}')
        return new_indices

def init_images(args, model=None):
    
    trainset = ImageFolder(
        classes=args.classes,
        ipc=args.mipc,
        shuffle=True,
        root=args.train_dir,
        transform=None,
        imbalance_rate=args.imbalance_rate,
        mem=True
    )
    trainset.balance_classes()
    trainset.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            MultiRandomCrop(
                num_crop=args.num_crop, size=args.input_size, factor=args.factor
            ),
            normalize,
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.mipc,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    image_list = []
    label_list = []

    for c, (images, labels) in enumerate(tqdm(train_loader)):
        images = selector(
            args.ipc * args.factor**2,
            model,
            images,
            labels,
            args.input_size,
            m=args.num_crop,
        )
        
        images = mix_images(images, args.input_size, args.factor, args.ipc)
        image_list.append(denormalize(images))
        label_list.append(labels[:args.ipc])
        save_images(args, denormalize(images), c)
    save_images_to_pt(args, image_list, label_list)
    
def save_images_to_pt(args, image_list, label_list):
    # 初始化保存路径
    save_path = os.path.join(args.syn_data_path, "dataset.pt")  # 保存为 dataset.pt 文件

    # 将列表中的张量拼接成一个大张量
    images_tensor = torch.cat(image_list, dim=0)  # 将 image_list 合并成一个张量
    labels_tensor = torch.cat(label_list, dim=0)  # 将 label_list 合并成一个张量

    # 保存为 .pt 文件
    torch.save([images_tensor, labels_tensor], save_path)
    print(f"Saved entire dataset to {save_path}")

def save_images(args, images, class_id):
    for id in range(images.shape[0]):
        dir_path = "{}/{:05d}".format(args.syn_data_path, class_id)
        place_to_store = dir_path + "/class{:05d}_id{:05d}.jpg".format(class_id, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def main(args):
    print(args)
    with torch.no_grad():
        if not os.path.exists(args.syn_data_path):
            os.makedirs(args.syn_data_path)
        else:
            shutil.rmtree(args.syn_data_path)
            os.makedirs(args.syn_data_path)

        model_teacher = load_model(
            args,
            model_name=args.arch_name,
            dataset=args.subset,
            pretrained=True,
            classes=args.classes,
        )

        model_teacher = nn.DataParallel(model_teacher).cuda()
        model_teacher.eval()
        for p in model_teacher.parameters():
            p.requires_grad = False

        init_images(args, model_teacher)


if __name__ == "__main__":
    pass
