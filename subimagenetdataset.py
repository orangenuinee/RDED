import os
import shutil

# 配置路径
source_dir = "../NCFMnew/dataset/imagenet/train"  # 原始训练集目录
target_dir = "data/imagefruit/train"      # 新目录
os.makedirs(target_dir, exist_ok=True)

# 类别顺序列表
class_order = [
    "n07753275",
    "n07753592",
    "n07745940",
    "n07747607",
    "n07749582",
    "n07768694",
    "n07753113",
    "n07720875",
    "n07718472",
    "n07760859"
]

# 遍历处理每个类别
for idx, class_id in enumerate(class_order):
    src = os.path.join(source_dir, class_id)
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

print("处理完成！")