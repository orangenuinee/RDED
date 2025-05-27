import os

# 指定目录路径
folder_path = '/root/autodl-tmp/infoutil/data/cifar10/val'  # 修改为实际路径


# 获取文件夹列表
categories = sorted(os.listdir(folder_path))

# 遍历每个类别并重命名
for idx, category in enumerate(categories):
    old_category_path = os.path.join(folder_path, category)
    
    # 创建新的类别名（格式化为5位数字）
    new_category_name = f"{idx:05d}"
    new_category_path = os.path.join(folder_path, new_category_name)
    
    # 重命名文件夹
    os.rename(old_category_path, new_category_path)
    print(f"Renamed '{category}' to '{new_category_name}'")

folder_path = '/root/autodl-tmp/infoutil/data/cifar10/train'  # 修改为实际路径

# 获取文件夹列表
categories = sorted(os.listdir(folder_path))

# 遍历每个类别并重命名
for idx, category in enumerate(categories):
    old_category_path = os.path.join(folder_path, category)
    
    # 创建新的类别名（格式化为5位数字）
    new_category_name = f"{idx:05d}"
    new_category_path = os.path.join(folder_path, new_category_name)
    
    # 重命名文件夹
    os.rename(old_category_path, new_category_path)
    print(f"Renamed '{category}' to '{new_category_name}'")