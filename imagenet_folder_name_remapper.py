import os
import json

# 加载JSON文件（包含类别映射）
json_file = './imagenet_class_index.json'
directory_path = '../NCFMnew/dataset/imagenet/train'

# 读取类别映射
with open(json_file, 'r') as f:
    class_mapping = json.load(f)

# 创建数字编号到名称的映射（去除前导零）
number_to_name = {key.zfill(5): value[0] for key, value in class_mapping.items()}

# 遍历目录
for folder_name in os.listdir(directory_path):
    folder_path = os.path.join(directory_path, folder_name)
    
    # 只处理目录且名称在映射中
    if os.path.isdir(folder_path) and folder_name in number_to_name:
        original_name = number_to_name[folder_name]
        original_folder_path = os.path.join(directory_path, original_name)
        
        # 重命名文件夹
        os.rename(folder_path, original_folder_path)
        print(f'Renamed back: {folder_name} -> {original_name}')
    else:
        print(f'Skipped: {folder_name}')

print("All folders restored to original names!")