import os
import shutil
from glob import glob

# 配置路径
source_dir = "../NCFMnew/dataset/imagenet"  # 原始数据集目录
subset_dir = "imagenet_subset"  # 包含子集定义文件的目录
target_root = "data"  # 新数据集的根目录

# 确保目标根目录存在
os.makedirs(target_root, exist_ok=True)

# 获取所有子集定义文件
subset_files = glob(os.path.join(subset_dir, "*.txt"))

# 处理每个子集文件
for subset_file in subset_files:
    # 从文件名获取子集名称（不带扩展名）
    subset_name = os.path.splitext(os.path.basename(subset_file))[0]
    
    print(f"\n正在处理子集: {subset_name}")
    
    # 读取类别顺序
    with open(subset_file, 'r') as f:
        class_order = [line.strip() for line in f if line.strip()]
    
    # 处理训练集和验证集
    for split in ['train', 'val']:
        target_dir = os.path.join(target_root, subset_name, split)
        source_split_dir = os.path.join(source_dir, split)
        
        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n处理 {split} 集:")
        
        # 遍历处理每个类别
        for idx, class_id in enumerate(class_order):
            src = os.path.join(source_split_dir, class_id)
            dst = os.path.join(target_dir, f"{idx:05d}")  # 格式化为5位数字
            
            if not os.path.exists(src):
                print(f"警告：找不到源目录 {src}")
                continue
            
            try:
                # 复制整个目录（包含所有图片）
                shutil.copytree(src, dst)
                print(f"已复制 {class_id} -> {os.path.basename(dst)}")
            except Exception as e:
                print(f"复制 {class_id} 失败: {str(e)}")

print("\n所有处理完成！")