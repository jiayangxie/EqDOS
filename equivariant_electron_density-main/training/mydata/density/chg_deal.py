import re
import torch
import os
import numpy as np
import time

# 定义一个函数来安全地转换字符串为浮点数
def safe_float(value):
    try:
        return float(value)
    except ValueError:
        return float(0.0)


def CHG_read(CHG_data_file,xyz_num,patch_num,patch_offset,spin,over,test):
    with open(CHG_data_file, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    charge_num = [int(x) for x in data[9 + xyz_num].split()]
    copy_line = '\n' + str(charge_num[0] // patch_num) + ' ' + str(charge_num[1] // patch_num) + ' ' + str(
        charge_num[2] // patch_num) + '\n'


    # 点位太多->进行patch，划分成patch_num*patch_num*patch_num的小正方体并取这个小正方体里[0,0,0]的点位，其余忽略
    X, Y, Z = torch.meshgrid(torch.arange(charge_num[2] // patch_num), torch.arange(charge_num[1] // patch_num),
                             torch.arange(charge_num[0] // patch_num), indexing='ij')
    # 在这里筛选点位
    feature = (torch.stack((Z, Y, X), dim=-1) * patch_num).reshape(-1, 3) + patch_offset

    # 读取全部density
    ### line_num = CHG是一行存几个，可能 5 or 10,决定文件总共有多少行
    string = data[10 + xyz_num].rstrip('\n')  # 先使用rstrip()函数删除字符串最后的\n符号
    string = re.sub(r'^\s+', '', string)  # 使用正则表达式 "^\s+" 删除字符串开始处的所有空格
    string = re.sub(r'\s+$', '', string)  # 使用正则表达式 "^\s+" 删除字符串结尾处的所有空格
    line_num = len(re.split(r'\s+', string))  # 使用正则表达式 "\s+" 进行分割

    ### 读取density
    if spin == False:
        density_tot_file = os.path.join(os.path.dirname(CHG_data_file), 'density_tot.pth')
        if over == 0:
            ## 读取CHGCAR: 上半部分电荷 -> 上旋+下旋density
            density_lines = charge_num[0] * charge_num[1] * charge_num[2] // line_num
            density = [[safe_float(y) for y in x.split()] for x in data[10 + xyz_num:10 + xyz_num + density_lines]]
            density = np.array(density).ravel()
            if charge_num[0] * charge_num[1] * charge_num[2] % line_num != 0:
                density_last_line = np.array([float(x) for x in data[10 + xyz_num + density_lines].split()])
                density = np.append(density, density_last_line)
            #  获取density对应的点位xyz
            density = torch.Tensor(density.reshape(-1)).reshape(charge_num[2], charge_num[1], charge_num[0])
            density = density[feature[:, 2], feature[:, 1], feature[:, 0]]
            if test:
                torch.save(density, density_tot_file)
        else:
            density = torch.load(density_tot_file)
        return charge_num,copy_line,density,feature


    else:
        density_tot_file = os.path.join(os.path.dirname(CHG_data_file), 'density_tot.pth')
        density_mag_file = os.path.join(os.path.dirname(CHG_data_file), 'density_mag.pth')
        if over == 0:
            ## 读取CHGCAR: 上半部分电荷 -> 上旋+下旋density
            density_lines = (charge_num[0] * charge_num[1] * charge_num[2]) // line_num
            density_tot = [[safe_float(y) for y in x.split()] for x in
                           data[10 + xyz_num:10 + xyz_num + density_lines]]
            density_tot = np.array(density_tot).ravel()
            last_line = 0
            if charge_num[0] * charge_num[1] * charge_num[2] % line_num != 0:
                density_last_line_tot = np.array([safe_float(x) for x in data[10 + xyz_num + density_lines].split()])
                density_tot = np.append(density_tot, density_last_line_tot)
                last_line = 1
            #  获取density对应的点位xyz
            density_tot = torch.Tensor(density_tot.reshape(-1))
            if test or density_tot.shape[0]>1e7:
                torch.save(density_tot, density_tot_file)


            ## 读取CHGCAR: 下半部分电荷  -> 上旋-下旋density
            # 有些CHG会在tol_chg部分的最下面多一行空格，注意判断一下
            if len(data[10 + xyz_num + density_lines + last_line].split()) == 0:
                start_line = 12 + last_line + xyz_num + density_lines
                end_line = 12 + last_line + xyz_num + 2 * density_lines
            else:
                start_line = 11 + last_line + xyz_num + density_lines
                end_line = 11 + last_line + xyz_num + 2 * density_lines
            density_mag = [[safe_float(y) for y in x.split()] for x in data[start_line: end_line]]
            density_mag = np.array(density_mag).ravel()
            if charge_num[0] * charge_num[1] * charge_num[2] % line_num != 0:
                density_last_line_mag = np.array([safe_float(x) for x in data[end_line].split()])
                density_mag = np.append(density_mag, density_last_line_mag)
            density_mag = torch.Tensor(density_mag.reshape(-1))
            if test or density_tot.shape[0]>1e6:
                torch.save(density_mag, density_mag_file)
        else:
            density_tot = torch.load(density_tot_file)
            density_mag = torch.load(density_mag_file)
        density_tot = density_tot.reshape(charge_num[2], charge_num[1], charge_num[0])
        #  获取density对应的点位xyz
        density_tot = density_tot[feature[:, 2], feature[:, 1], feature[:, 0]]

        density_mag = density_mag.reshape(charge_num[2], charge_num[1], charge_num[0])
        #  获取density对应的点位xyz
        density_mag = density_mag[feature[:, 2], feature[:, 1], feature[:, 0]]

        density_up = (density_tot + density_mag) / 2
        density_down = (density_tot - density_mag) / 2
        ##合并上旋与下旋
        density = torch.cat((density_up.unsqueeze(1), density_down.unsqueeze(1)), 1)
        return charge_num,copy_line,density,feature


