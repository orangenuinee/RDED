import os
import json


json_file = './imagenet_class_index.json'
directory_path = 'data/imagenet/val'

# directory_path = 'imagenet/val'


with open(json_file, 'r') as f:
    class_mapping = json.load(f)


name_to_number = {value[0]: key.zfill(5) for key, value in class_mapping.items()} 


for folder_name in os.listdir(directory_path):
    folder_path = os.path.join(directory_path, folder_name)


    if os.path.isdir(folder_path) and folder_name in name_to_number:
        new_name = name_to_number[folder_name]
        new_folder_path = os.path.join(directory_path, new_name)
        

        os.rename(folder_path, new_folder_path)
        print(f'Renamed: {folder_name} -> {new_name}')
    else:
        print(f'Skipped: {folder_name}')

print("All folders processed!")