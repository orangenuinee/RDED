import torch
import numpy as np
import os
import torch.distributed
import torchvision
from torchvision.transforms import functional as t_F
import torch.nn.functional as F
import random
import math
from tqdm import tqdm
import json
from torchvision import datasets, transforms
# keep top k largest values, and smooth others
def keep_top_k(p, k, n_classes=1000):  # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes - k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(
        topk_smooth_p.sum().item(), p.shape[0]
    ), f"{topk_smooth_p.sum().item()} not close to {p.shape[0]}"
    return topk_smooth_p


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find("weight") >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(
        group_no_weight_decay
    )
    groups = [
        dict(params=group_weight_decay),
        dict(params=group_no_weight_decay, weight_decay=0.0),
    ]
    return groups
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


class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, classes, ipc, imbalance_rate, mem=False, shuffle=False, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.mem = mem
        self.image_paths = []
        self.nclass = len(classes)
        self.imbalance_rate=imbalance_rate
        self.targets = []
        self.ipc=ipc
        self.samples = []
        for c in range(len(classes)):
            dir_path = self.root + "/" + str(classes[c]).zfill(5)
            print(dir_path)
            file_ls = os.listdir(dir_path)
            if shuffle:
                random.shuffle(file_ls)
            # print(len(file_ls))
            for i in range(ipc):
                self.image_paths.append(dir_path + "/" + file_ls[i])
                self.targets.append(c)
                if self.mem:
                    self.samples.append(self.loader(dir_path + "/" + file_ls[i]))
                #print(self.samples)

    def __getitem__(self, index):
        if self.mem:
            sample = self.samples[index]
        else:
            sample = self.loader(self.image_paths[index])
        sample = self.transform(sample)
        return sample, self.targets[index]

    def __len__(self):
        return len(self.targets)
    def balance_classes(self):
        """
        根据不平衡率调整类的样本数量,
        保持第一个类别与原始数量相同，后续类别按 imbalance_rate 递减
        """
        # 创建一个数组来存储每个类的索引
        idx_class = [[] for _ in range(self.nclass)]
        n = len(self.samples)
        for i in range(n):
            try:
                label = self.targets[i]
                idx_class[label].append(i)
            except Exception as e:
                print(label)
            
        # 保持第一个类别的样本数与原始一致
        first_class_size = len(idx_class[0])
        
        # 生成新的索引列表
        new_indices = []
        new_indices.extend(idx_class[0])  # 添加第一个类别的所有样本

        # 计算并选择后续类别的样本数量
        for cls_idx in range(1, self.nclass):
            num_samples = math.ceil(first_class_size * (self.imbalance_rate ** (cls_idx / (self.nclass - 1))))
            if num_samples > 0:  # 只处理大于0的样本数
                num_samples = min(num_samples, len(idx_class[cls_idx]))  # 确保不超过可用样本数量
                sampled_indices = np.random.choice(idx_class[cls_idx], size=num_samples, replace=False)
                if len(sampled_indices) < self.ipc:
                    print(len(sampled_indices),self.ipc)
                    sampled_indices = np.random.choice(sampled_indices, size=self.ipc, replace=True)
                new_indices.extend(sampled_indices)

        # 创建新的数据和目标基于新索引
        print(n)
        print(len(new_indices))
        self.image_paths = [self.image_paths[i] for i in tqdm(new_indices, desc="Processing Samples")]
        self.targets = [self.targets[i] for i in tqdm(new_indices, desc="Processing Samples")]
        #self.targets = [s[1] for s in self.samples]
        # self.data = self.data[new_indices]
        # self.targets = [self.targets[i] for i in new_indices]
        # 更新样本数量
        print(f'imbalance_rate: {self.imbalance_rate}')
        print(f'新的数据集大小: {len(self.samples)}')
        idx_class = [[] for _ in range(self.nclass)]
        n = len(self.targets)
        for i in tqdm(range(n), desc="Processing Samples"):
            label = self.targets[i]
            idx_class[label].append(i)
        for i in range(self.nclass):
            print(f"第{i}类图片数目:{len(idx_class[i])}")
        # with open('samples.json', 'w') as f:
        #     json.dump(self.samples, f)
        return new_indices


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, args, rand_index=None, lam=None, bbox=None):
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(args.cutmix, args.cutmix)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None):
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(args.mixup, args.mixup)

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images, args, rand_index=None, lam=None, bbox=None):
    if args.mix_type == "mixup":
        return mixup(images, args, rand_index, lam)
    elif args.mix_type == "cutmix":
        return cutmix(images, args, rand_index, lam, bbox)
    else:
        return images, None, None, None


class ShufflePatches(torch.nn.Module):
    def shuffle_weight(self, img, factor):
        h, w = img.shape[1:]
        th, tw = h // factor, w // factor
        patches = []
        for i in range(factor):
            i = i * tw
            if i != factor - 1:
                patches.append(img[..., i : i + tw])
            else:
                patches.append(img[..., i:])
        random.shuffle(patches)
        img = torch.cat(patches, -1)
        return img

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        return img
