import os
import shutil

def rename_folders(path):
    counter = 0
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):  # 判断是否为文件夹
            new_folder_name = str(counter)
            new_folder_path = os.path.join(path, new_folder_name)
            os.rename(folder_path, new_folder_path)
            counter += 1

def collect_one(path,new_path):
    num = 0
    for folder in os.listdir(path):
        folder_path = os.path.join(path, str(folder))
        print(folder_path)
        if os.path.isdir(folder_path):  # 判断是否为文件夹
            POS_data_file = os.path.join(folder_path, 'POSCAR')
            with open(POS_data_file, "r", encoding="utf-8") as f:
                data = f.read().splitlines()
            Z_1 = data[5].split()
            if len(Z_1)==3:
                print(Z_1)
                new_folder_name = str(num)
                new_folder_path = os.path.join(new_path, new_folder_name)
                if os.path.isdir(new_folder_path):
                    shutil.rmtree(new_folder_path)
                shutil.copytree(folder_path, new_folder_path)
                num += 1
    # for folder in range(len(os.listdir(path))):
    #     folder_path = os.path.join(path, str(folder))
    #     if os.path.isdir(folder_path):  # 判断是否为文件夹
    #         POS_data_file = os.path.join(folder_path, 'POSCAR')
    #         with open(POS_data_file, "r", encoding="utf-8") as f:
    #             data = f.read().splitlines()
    #         Z_1 = data[5].split()
    #         # if len(Z_1) == 3:
    #         #     print(Z_1)
    #         new_folder_name = str(num)
    #         new_folder_path = os.path.join(new_path, new_folder_name)
    #         if os.path.isdir(new_folder_path):
    #             shutil.rmtree(new_folder_path)
    #         shutil.copytree(folder_path, new_folder_path)
    #         num += 1

# 设置文件夹路径
folder_path = '三元POSCAR/'
new_path = 'new/'
collect_one(folder_path,new_path)
