import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate

def read_doscar(dir_name,en_range,max_spd):
    #读取DOSCAR文件
    DOS_data_file = os.path.join(dir_name, 'DOSCAR')
    with open(DOS_data_file, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
        # doscar需要将0点设置成费米能级
        ef = float(data[5].split()[3])
        print(ef)
        if ef>2 or ef<-11:
            print(f'{DOS_data_file}文件的费米能级，已经删除')
            return None,None
        dos_n = []
        num = -1
        for d in data[5:]:
            if d.endswith('      1.00000000'):
                num = num + 1
                if num == 1:
                    #第一个存储的dos是总dos
                    dos_all = dos
                elif num > 1:
                    #之后出现的dos是每个原子的dos
                    dos_n.append(dos)
                dos = []
            else:
                dos.append([float(x) for x in d.split()])
        dos_n.append(dos)
    # 最终dos_n大小[atom_num,1(energy)+32(dos),windows]
    dos_n = np.array(dos_n).transpose(0,2,1)

    #进行插值计算得到需要的windows大小
    dos = np.zeros((dos_n.shape[0], max_spd, 3000))
    xnew = np.linspace(-en_range, en_range - 0.01, 3000)
    for i in range(0, dos_n.shape[0]):
        xfit = dos_n[i, 0, :]
        xfit = xfit - ef
        for j in range(1, max_spd+1):
            if j > dos_n.shape[1]-1:
                break
            yfit = dos_n[i, j, :]
            dos[i,j-1,:] = np.interp(xnew,xfit,yfit)
            # if np.max(dos[i,j-1,:])<0.02:
            #     dos[i, j - 1, :] = np.zeros_like(dos[i, j - 1, :])
    xfits = np.expand_dims(xnew,0).repeat(dos_n.shape[0],0)
    return dos,xfits

def dos_deal(dir_name,i,xyz_num,en_range,windows,split,spd_dict,max_spd,n):
    if os.path.exists(f'{dir_name + str(i)}.npy'):
        dos_read = np.load(os.path.join(dir_name, f"{i}.npy"))[:,1:,:]
        xfits = np.load(os.path.join(dir_name, f"{i}.npy"))[:, 0, :]
    elif os.path.exists(os.path.join(dir_name,str(i),'DOSCAR')):
        dos_read,xfits = read_doscar(os.path.join(dir_name,str(i)),en_range,max_spd)
    if dos_read is None:
        return None, None, None, None
    if split:
        if windows != 500:
            #dos_read  【xyz_num原子数,spdf轨道数，3000windows】
            dos = np.zeros((xyz_num,dos_read.shape[1],windows))
            for i in range(xyz_num):
                for j in range(1, dos_read.shape[1]):
                    # 高斯模糊，不然dos波动太大
                    dos_read[i, j - 1, :] = gaussian_filter1d(dos_read[i, j - 1, :], sigma=10)
                    xfit = xfits[i]
                    yfit = dos_read[i,j-1, :]
                    dos_fit = interpolate.interp1d(xfit, yfit, kind="linear", bounds_error=False, fill_value=0)
                    xnew = np.linspace(-en_range, en_range, windows)
                    dos[i,j-1, :] = dos_fit(xnew)

        elif windows == 500:
            # for i in range(xyz_num):
            #     for j in range(dos_read.shape[1]):
            #         dos_read[i, j, :] = gaussian_filter1d(dos_read[i, j, :], sigma=7)
            dos = torch.nn.AvgPool1d(6, 6)(torch.Tensor(dos_read)).numpy()
        scaling = np.max(dos, axis=2)
        for i in range(0, xyz_num):
            for j in range(1, dos_read.shape[1]):
                if scaling[i, j - 1] == 0:
                    scaling[i, j - 1] = 1e-32
                dos[i, j - 1, :] = dos[i, j - 1, :] / scaling[i, j - 1]
        dos = dos.transpose(0, 2, 1)
        spd = np.array([spd_dict[int(N)] for N in n]).squeeze()

    else:
        if windows == 400:
            #dos_read  【xyz_num原子数,spdf轨道数，3000windows】 -> dos_sum 【xyz_num原子数, 3000windows】
            dos_sum = np.sum(dos_read, axis=1)
            dos = np.zeros((xyz_num, windows))
            for i in range(xyz_num):
                # 高斯模糊，不然dos波动太大
                dos_sum[i, :] = gaussian_filter1d(dos_sum[i, :], sigma=10)
                xfit = xfits[i]
                yfit = dos_sum[i, :]
                dos_fit = interpolate.interp1d(xfit, yfit, kind="linear", bounds_error=False, fill_value=0)
                xnew = np.linspace(-en_range, en_range, windows)
                dos[i, :] = dos_fit(xnew)
            #dos_sum 【xyz_num原子数, 400windows】
            dos_sum = dos
        elif windows == 500:
            # dos_sum 【xyz_num原子数, spdf轨道数，500windows】
            dos = torch.nn.AvgPool1d(6, 6)(torch.Tensor(dos_read)).numpy()
            # dos_sum 【xyz_num原子数, 500windows】
            dos_sum = np.sum(dos, axis=1)


        # interpolation and shift energy window
        scaling = np.max(dos_sum, axis=1)
        # scaling = np.sum(dos_sum, axis=1)
        for i in range(0, xyz_num):
            if scaling[i] == 0:
                scaling[i] = 1e-16
            dos_sum[i, :] = dos_sum[i, :] / scaling[i]
        dos = dos_sum
        spd = None
    return dos,spd,dos_read.shape[1], torch.Tensor(scaling)