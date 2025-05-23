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
