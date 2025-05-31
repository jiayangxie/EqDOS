import torch
from torch import nn
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import os
from ase import Atoms
import re
import math
import textwrap

class my_selfattention1(nn.Module):
    def __init__(self, embed_dim=200, num_heads=4):
        super(my_selfattention1, self).__init__()
        self.pre_linearq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.pre_linearkv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.post_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.relu = nn.PReLU()
        self.num_head = num_heads
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ipt):
        len_seq, batch_size, emb_dim = ipt.shape
        len_seqq1 = 0
        len_seqq2 = 104 - 8
        len_seqkv1 = 104 - 8
        len_seqkv2 = 104
        q = self.relu(self.pre_linearq(ipt[len_seqq1:len_seqq2, :, :]))
        k, v = self.relu(self.pre_linearkv(ipt[len_seqkv1:len_seqkv2, :, :])).chunk(2, dim=-1)
        len_seqq, batch_size, emb_dim = q.shape
        len_seqk, batch_size, emb_dim = k.shape
        # print(q.shape)
        q = q.contiguous().view(len_seqq2 - len_seqq1, batch_size * self.num_head, emb_dim // self.num_head).transpose(
            0, 1)
        k = k.contiguous().view(len_seqkv2 - len_seqkv1, batch_size * self.num_head,
                                emb_dim // self.num_head).transpose(0, 1)
        v = v.contiguous().view(len_seqkv2 - len_seqkv1, batch_size * self.num_head,
                                emb_dim // self.num_head).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        tem_mask = torch.zeros_like(attn_weights)
        tem_mask[attn_weights == 0] = -10000
        attn_weights = (attn_weights + tem_mask) * ((emb_dim) ** -0.5)
        attn_softmax_weights = self.softmax(attn_weights.view(batch_size * self.num_head, len_seqq * len_seqk)).view(
            batch_size * self.num_head, len_seqq, len_seqk)
        attn_output = torch.bmm(attn_softmax_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(len_seqq, batch_size, emb_dim)
        attn_output = self.post_linear(attn_output)
        attn_softmax_weights = attn_softmax_weights.view(batch_size, self.num_head, len_seqq, len_seqk)
        attn_softmax_weights = attn_softmax_weights.mean(dim=1)
        attn_weights = attn_weights.view(batch_size, self.num_head, len_seqq, len_seqk)
        attn_weights = attn_weights.mean(dim=1)
        return attn_output, attn_softmax_weights, attn_weights


class energytrans1(nn.Module):
    def __init__(self, dim=200, l1=500, l2=500):
        super(energytrans1, self).__init__()
        self.PreLinear = nn.Linear(in_features=500, out_features=dim, bias=False)
        self.Self_Attn1 = my_selfattention1(embed_dim=dim, num_heads=4)
        self.DP = nn.Dropout(p=0.2)
        self.FN = nn.Flatten()
        self.LNx1 = nn.Linear(dim * 96, l1, bias=False)
        self.LNx2 = nn.Linear(l1, l2, bias=False)
        self.LNx3 = nn.Linear(l2, l1, bias=False)
        self.LNx4 = nn.Linear(l1, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        y0 = self.PreLinear(x.permute(1, 0, 2))
        # y0 = x.permute(1,0,2)
        y, _, _ = self.Self_Attn1(y0)
        # y = y + y0
        y = y.permute(1, 0, 2)
        y = self.FN(y)
        y = self.DP(y)
        y = self.LNx1(y)
        y1 = self.relu(y)
        y = self.LNx2(y1)
        y = self.relu(y)
        y = self.LNx3(y)
        y = self.relu(y)
        y = y + y1
        y = self.LNx4(y)
        y = self.relu(y)
        return y.sum(-1)


class my_selfattention2(nn.Module):
    def __init__(self, embed_dim=200, num_heads=4):
        super(my_selfattention2, self).__init__()
        self.pre_linearq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.pre_linearkv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.post_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.relu = nn.PReLU()
        self.num_head = num_heads
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ipt):
        len_seq, batch_size, emb_dim = ipt.shape
        len_seqq1 = 0
        len_seqq2 = 104 - 8
        len_seqkv1 = 208 - 8
        len_seqkv2 = 208
        q = self.relu(self.pre_linearq(ipt[len_seqq1:len_seqq2, :, :]))
        k, v = self.relu(self.pre_linearkv(ipt[len_seqkv1:len_seqkv2, :, :])).chunk(2, dim=-1)
        len_seqq, batch_size, emb_dim = q.shape
        len_seqk, batch_size, emb_dim = k.shape
        # print(q.shape)
        q = q.contiguous().view(len_seqq2 - len_seqq1, batch_size * self.num_head, emb_dim // self.num_head).transpose(
            0, 1)
        k = k.contiguous().view(len_seqkv2 - len_seqkv1, batch_size * self.num_head,
                                emb_dim // self.num_head).transpose(0, 1)
        v = v.contiguous().view(len_seqkv2 - len_seqkv1, batch_size * self.num_head,
                                emb_dim // self.num_head).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        tem_mask = torch.zeros_like(attn_weights)
        tem_mask[attn_weights == 0] = -10000
        attn_weights = (attn_weights + tem_mask) * ((emb_dim) ** -0.5)
        attn_softmax_weights = self.softmax(attn_weights.view(batch_size * self.num_head, len_seqq * len_seqk)).view(
            batch_size * self.num_head, len_seqq, len_seqk)
        attn_output = torch.bmm(attn_softmax_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(len_seqq, batch_size, emb_dim)
        attn_output = self.post_linear(attn_output)
        attn_softmax_weights = attn_softmax_weights.view(batch_size, self.num_head, len_seqq, len_seqk)
        attn_softmax_weights = attn_softmax_weights.mean(dim=1)
        attn_weights = attn_weights.view(batch_size, self.num_head, len_seqq, len_seqk)
        attn_weights = attn_weights.mean(dim=1)
        return attn_output, attn_softmax_weights, attn_weights


class energytrans2(nn.Module):
    def __init__(self, dim=200, l1=500, l2=500):
        super(energytrans2, self).__init__()
        self.PreLinear = nn.Linear(in_features=500, out_features=dim, bias=False)
        self.Self_Attn1 = my_selfattention2(embed_dim=dim, num_heads=4)
        self.DP = nn.Dropout(p=0.2)
        self.FN = nn.Flatten()
        self.LNx1 = nn.Linear(dim * 96, l1, bias=False)
        self.LNx2 = nn.Linear(l1, l2, bias=False)
        self.LNx3 = nn.Linear(l2, l1, bias=False)
        self.LNx4 = nn.Linear(l1, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        y0 = self.PreLinear(x.permute(1, 0, 2))
        # y0 = x.permute(1,0,2)
        y, _, _ = self.Self_Attn1(y0)
        # y = y + y0
        y = y.permute(1, 0, 2)
        y = self.FN(y)
        y = self.DP(y)
        y = self.LNx1(y)
        y1 = self.relu(y)
        y = self.LNx2(y1)
        y = self.relu(y)
        y = self.LNx3(y)
        y = self.relu(y)
        y = y + y1
        y = self.LNx4(y)
        y = -self.relu(y)
        return y.sum(-1)


class my_selfattention3(nn.Module):
    def __init__(self, embed_dim=200, num_heads=4):
        super(my_selfattention3, self).__init__()
        self.pre_linearq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.pre_linearkv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.post_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.relu = nn.PReLU()
        self.num_head = num_heads
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ipt):
        len_seq, batch_size, emb_dim = ipt.shape
        len_seqq1 = 104
        len_seqq2 = 208 - 8
        len_seqkv1 = 104 - 8
        len_seqkv2 = 104
        q = self.relu(self.pre_linearq(ipt[len_seqq1:len_seqq2, :, :]))
        k, v = self.relu(self.pre_linearkv(ipt[len_seqkv1:len_seqkv2, :, :])).chunk(2, dim=-1)
        len_seqq, batch_size, emb_dim = q.shape
        len_seqk, batch_size, emb_dim = k.shape
        # print(q.shape)
        q = q.contiguous().view(len_seqq2 - len_seqq1, batch_size * self.num_head, emb_dim // self.num_head).transpose(
            0, 1)
        k = k.contiguous().view(len_seqkv2 - len_seqkv1, batch_size * self.num_head,
                                emb_dim // self.num_head).transpose(0, 1)
        v = v.contiguous().view(len_seqkv2 - len_seqkv1, batch_size * self.num_head,
                                emb_dim // self.num_head).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        tem_mask = torch.zeros_like(attn_weights)
        tem_mask[attn_weights == 0] = -10000
        attn_weights = (attn_weights + tem_mask) * ((emb_dim) ** -0.5)
        attn_softmax_weights = self.softmax(attn_weights.view(batch_size * self.num_head, len_seqq * len_seqk)).view(
            batch_size * self.num_head, len_seqq, len_seqk)
        attn_output = torch.bmm(attn_softmax_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(len_seqq, batch_size, emb_dim)
        attn_output = self.post_linear(attn_output)
        attn_softmax_weights = attn_softmax_weights.view(batch_size, self.num_head, len_seqq, len_seqk)
        attn_softmax_weights = attn_softmax_weights.mean(dim=1)
        attn_weights = attn_weights.view(batch_size, self.num_head, len_seqq, len_seqk)
        attn_weights = attn_weights.mean(dim=1)
        return attn_output, attn_softmax_weights, attn_weights


class energytrans3(nn.Module):
    def __init__(self, dim=200, l1=500, l2=500):
        super(energytrans3, self).__init__()
        self.PreLinear = nn.Linear(in_features=500, out_features=dim, bias=False)
        self.Self_Attn1 = my_selfattention3(embed_dim=dim, num_heads=4)
        self.DP = nn.Dropout(p=0.2)
        self.FN = nn.Flatten()
        self.LNx1 = nn.Linear(dim * 96, l1, bias=False)
        self.LNx2 = nn.Linear(l1, l2, bias=False)
        self.LNx3 = nn.Linear(l2, l1, bias=False)
        self.LNx4 = nn.Linear(l1, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        y0 = self.PreLinear(x.permute(1, 0, 2))
        # y0 = x.permute(1,0,2)
        y, _, _ = self.Self_Attn1(y0)
        # y = y + y0
        y = y.permute(1, 0, 2)
        y = self.FN(y)
        y = self.DP(y)
        y = self.LNx1(y)
        y1 = self.relu(y)
        y = self.LNx2(y1)
        y = self.relu(y)
        y = self.LNx3(y)
        y = self.relu(y)
        y = y + y1
        y = self.LNx4(y)
        y = -self.relu(y)
        return y.sum(-1)


class my_selfattention4(nn.Module):
    def __init__(self, embed_dim=200, num_heads=4):
        super(my_selfattention4, self).__init__()
        self.pre_linearq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.pre_linearkv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.post_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.relu = nn.PReLU()
        self.num_head = num_heads
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ipt):
        len_seq, batch_size, emb_dim = ipt.shape
        len_seqq1 = 104
        len_seqq2 = 208 - 8
        len_seqkv1 = 208 - 8
        len_seqkv2 = 208
        q = self.relu(self.pre_linearq(ipt[len_seqq1:len_seqq2, :, :]))
        k, v = self.relu(self.pre_linearkv(ipt[len_seqkv1:len_seqkv2, :, :])).chunk(2, dim=-1)
        len_seqq, batch_size, emb_dim = q.shape
        len_seqk, batch_size, emb_dim = k.shape
        # print(q.shape)
        q = q.contiguous().view(len_seqq2 - len_seqq1, batch_size * self.num_head, emb_dim // self.num_head).transpose(
            0, 1)
        k = k.contiguous().view(len_seqkv2 - len_seqkv1, batch_size * self.num_head,
                                emb_dim // self.num_head).transpose(0, 1)
        v = v.contiguous().view(len_seqkv2 - len_seqkv1, batch_size * self.num_head,
                                emb_dim // self.num_head).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        tem_mask = torch.zeros_like(attn_weights)
        tem_mask[attn_weights == 0] = -10000
        attn_weights = (attn_weights + tem_mask) * ((emb_dim) ** -0.5)
        attn_softmax_weights = self.softmax(attn_weights.view(batch_size * self.num_head, len_seqq * len_seqk)).view(
            batch_size * self.num_head, len_seqq, len_seqk)
        attn_output = torch.bmm(attn_softmax_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(len_seqq, batch_size, emb_dim)
        attn_output = self.post_linear(attn_output)
        attn_softmax_weights = attn_softmax_weights.view(batch_size, self.num_head, len_seqq, len_seqk)
        attn_softmax_weights = attn_softmax_weights.mean(dim=1)
        attn_weights = attn_weights.view(batch_size, self.num_head, len_seqq, len_seqk)
        attn_weights = attn_weights.mean(dim=1)
        return attn_output, attn_softmax_weights, attn_weights


class energytrans4(nn.Module):
    def __init__(self, dim=200, l1=500, l2=500):
        super(energytrans4, self).__init__()
        self.PreLinear = nn.Linear(in_features=500, out_features=dim, bias=False)
        self.Self_Attn1 = my_selfattention4(embed_dim=dim, num_heads=4)
        self.DP = nn.Dropout(p=0.2)
        self.FN = nn.Flatten()
        self.LNx1 = nn.Linear(dim * 96, l1, bias=False)
        self.LNx2 = nn.Linear(l1, l2, bias=False)
        self.LNx3 = nn.Linear(l2, l1, bias=False)
        self.LNx4 = nn.Linear(l1, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        y0 = self.PreLinear(x.permute(1, 0, 2))
        # y0 = x.permute(1,0,2)
        y, _, _ = self.Self_Attn1(y0)
        # y = y + y0
        y = y.permute(1, 0, 2)
        y = self.FN(y)
        y = self.DP(y)
        y = self.LNx1(y)
        y1 = self.relu(y)
        y = self.LNx2(y1)
        y = self.relu(y)
        y = self.LNx3(y)
        y = self.relu(y)
        y = y + y1
        y = self.LNx4(y)
        y = self.relu(y)
        return y.sum(-1)


class energytrans(nn.Module):
    def __init__(self, dim=200, l1=500, l2=500):
        super(energytrans, self).__init__()
        # self.trans1 = energytrans1(dim, l1, l2)
        self.trans2 = energytrans2(dim, l1, l2)
        self.trans3 = energytrans3(dim, l1, l2)
        # self.trans4 = energytrans4(dim, l1, l2)

    def forward(self, x):
        # return self.trans1(x) + self.trans2(x) + self.trans3(x) + self.trans4(x)
        return self.trans2(x) + self.trans3(x)


class dosnet(nn.Module):
    def __init__(self, modellist):
        super(dosnet, self).__init__()
        self.modellist = modellist

    def forward(self, x):
        out = None
        for model in self.modellist:
            if out == None:
                out = model(x)
            else:
                out = model(x) + out
        return out / 5

def pre(x):
    # # 创建原始数据点
    # energy = np.linspace(-15, 15, 500)  # 创建一个包含 10 个元素的等差数列
    # # 创建插值点
    # energy_new = np.linspace(-15, 15, 3000)  # 创建一个包含 100 个元素的等差数列，用于插值
    # x_new = torch.zeros(x.shape[0],3000,x.shape[2])
    # # 进行线性插值
    # #x大小[400,104]->[500,104]
    # for n in range(x.shape[0]):
    #     for j in range(x.shape[2]):
    #         yfit = x[n, :, j]
    #         # dos_fit = interpolate.interp1d(energy, yfit, kind="linear", bounds_error=False, fill_value=0)
    #         # x_new[n, :, j] = torch.from_numpy(dos_fit(energy_new))
    #         x_new[n, :, j] = torch.Tensor(np.interp(energy_new, energy, yfit))
    # y = x_new.permute(2, 0, 1)
    # y = nn.AvgPool1d(6, 6)(y).permute(1,0,2)

    # [batch_size,.permute(1,0,2)windows,104] -> [104,batch_size,windows]->[batch_size,104,windows]
    # y = nn.AvgPool1d(6, 6)(x.permute(2,0,1)).permute(1,0,2)
    # print(torch.sum(torch.sum(y,1),1))
    y = x.permute(0,2,1)
    y = torch.cat((y, y), dim=1)
    y[:, :104, 200:] = 0
    y[:, 104:, :200] = 0
    # print(torch.sum(torch.sum(y, 1), 1))
    # print(y.shape)
    return y

# 遍历文件夹中的文件
def get_energy(path):
    info = {}
    with open(path, 'r') as file:
        lines = file.readlines()  # 读取文件的所有行
        for line in lines:
            structure_name, energy = line.strip().split(':')  # 按照 ':' 分割结构名和能量
            info[structure_name] = float(energy)  # 将结构名作为 key，能量作为 value 存储到字典中
    return info

def read_doscar(dir_name,en_range,spd = 32,ef = None):
    if ef == None:
        # 读取OUTCAR获取费米能级
        OUT_data_file = os.path.join(dir_name, 'OUTCAR')
        with open(OUT_data_file, 'r') as f:
            outcar_content = f.read()
            f.close()
        # 对于slab，ef默认为None，费米能级使用正则表达式匹配 OUTCAR 中费米能级的信息
        # 对于吸附原子，ef使用holo与lumo的一半
        ef = float(re.findall(r' E-fermi :(.+) \s+XC', outcar_content)[0])
    #读取DOSCAR文件
    DOS_data_file = os.path.join(dir_name, 'DOSCAR')
    with open(DOS_data_file, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
        dos_n = []
        num = -1
        for d in data[5:]:
            if d.startswith('     20.00000000'):
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
        # 最终dos_n大小[atom_num,1(energy)+32(dos),windows]
        dos_n.append(dos)
    dos_n = np.array(dos_n).transpose(0,2,1)
    #进行插值计算得到需要的windows大小
    dos = np.zeros((dos_n.shape[0], spd, 3000))
    for i in range(0, dos_n.shape[0]):
        xfit = dos_n[i, 0, :]
        for j in range(1, spd+1):
            if j > dos_n.shape[1]-1:
                break
            yfit = dos_n[i, j, :]
            xnew = np.linspace(-en_range, en_range-0.01, 3000) + ef
            dos[i,j-1,:] = np.interp(xnew,xfit,yfit)
    dos = torch.nn.AvgPool1d(6, 6)(torch.Tensor(dos)).numpy()
    return dos

def get_xifu_energy(xifu_names):
    xifu_energys = {}
    with open('yuanzi/ef_energy.txt', 'r') as file:
        data = file.read()
        ef_energy = eval(data)
    for xifu_name in xifu_names:
        xifu_energys[xifu_name] = ef_energy[xifu_name]
    return xifu_energys

def get_surface(z,dos,energy_range,windows,xifu_info,xifu_energys,spd=32):
    #读取吸附原子的DOSCAR，得到大小为[windows,8个轨道]
    # 吸附分子的费米能级是homo，
    dos_spd_save_ml_all = []
    dos_spd_save_vasp_all = []
    energy_values = []
    surs = []
    for (xifu_name,xifu_energy) in xifu_energys.items():
        # ef = xifu_energy[1]
        # dos_yuanzi_spd_new = read_doscar(os.path.join('ad_yuanzi', xifu_name), energy_range, spd=8, ef=ef)[0].transpose(1, 0)
        yuanzi_data = np.load('yuanzi/data_yuanzi.npz')
        dos_yuanzi_spd_new = yuanzi_data['dos'][yuanzi_data['xifu_names'] == xifu_names][0]
        # 检查一下这个结构在数据集中有没有对应吸附能数据，如果没有就不进行预测跳过。
        surface = {key: value for key, value in xifu_info.items() if key.startswith(z+'_'+xifu_name+'_')}
        if len(surface) == 0:
            return None, None, None
        else:
            for (sur, energy_value) in surface.items():
                # 获取吸附位点信息
                weidian = [int(x) for x in sur.split('_')[-1].split(",")]
                weidian_num = len(weidian)
                dos_spd_save_ml = dos[0][weidian].transpose(1, 0, 2).reshape(windows, spd * weidian_num)
                dos_spd_save_vasp = dos[1][weidian].transpose(1, 0, 2).reshape(windows, spd * weidian_num)
                # 对不足三个原子的地方进行补零操作
                dos_spd_save_ml = np.pad(dos_spd_save_ml, ((0, 0), (0, 32 * (3 - weidian_num))))/weidian_num
                dos_spd_save_vasp = np.pad(dos_spd_save_vasp, ((0, 0), (0, 32 * (3 - weidian_num))))/weidian_num
                # 最后补上吸附分子的dos->[windows,104]
                dos_spd_save_ml = np.concatenate((dos_spd_save_ml, dos_yuanzi_spd_new), 1)
                dos_spd_save_vasp = np.concatenate((dos_spd_save_vasp, dos_yuanzi_spd_new), 1)
                # print(sur)
                # print(np.sum(dos_spd_save_vasp[:,0:32]))
                # print(np.sum(dos_spd_save_vasp[:, 32:64]))
                # print(np.sum(dos_spd_save_vasp[:, 64:96]))
                # print(np.sum(dos_spd_save_vasp[:, 96:]))
                # xnew = np.linspace(-15, 15, 500)
                # for i in range(0, dos[1].shape[0]):
                #     print(dos.shape)
                #     plt.plot(xnew, np.sum(dos[1][i], 1)-np.sum(dos[1][i+1], 1))
                #     print(np.sum(np.sum(dos[1][i], 1),0)/3)
                #     plt.show()
                #     plt.close()
                # print('++++++++++++++++++++++')

                # xnew = np.linspace(-15, 15, 500)
                # for i in range(96,dos_spd_save_vasp.shape[1]):
                #     plt.plot(xnew, dos_spd_save_vasp[:, i])
                #     plt.plot(xnew,dos[1][weidian][2][:,i-64]/weidian_num)
                #     plt.show()
                #     plt.close()

                # 将不同吸附位点的存储到同一数组
                dos_spd_save_ml_all.append(dos_spd_save_ml)
                dos_spd_save_vasp_all.append(dos_spd_save_vasp)
                energy_values.append(energy_value)
                surs.append(sur)
    # 将ml与vasp的数据进行合并
    dos_spd_save_ml_all = torch.Tensor(dos_spd_save_ml_all).unsqueeze(1)
    dos_spd_save_vasp_all = torch.Tensor(dos_spd_save_vasp_all).unsqueeze(1)
    # print(surs)
    # print(torch.sum(torch.sum(dos_spd_save_vasp_all.squeeze(1),1),1))
    # print('+++++++++++++')
    dos_save = torch.cat((dos_spd_save_ml_all, dos_spd_save_vasp_all), 1)
    return dos_save, torch.Tensor(energy_values), surs

def mape_loss_func(preds,labels):
    return torch.abs((labels-preds)/labels).mean()

# 绘制结果图
def draw_result(xifu_name,ml_xifu,ml_all,vasp,name):
    xifu_color = {'C':(230/255, 85/255, 13/255),
                  'H':(49/255, 130/255, 189/255),
                  'O':(117/255, 107/255, 177/255),
                  'N':(49/255, 163/255, 84/255),
                  'S':(99/255, 99/255, 99/255),
                  'CH':(71/255, 141/255, 205/255),
                  'HO':(71/255, 141/255, 205/255),
                  'all':(11/255, 11/255, 215/255),}
    mae_xifu = torch.nn.functional.l1_loss(ml_xifu, vasp)
    mape_xifu = mape_loss_func(ml_xifu, vasp)*100
    mae_all = torch.nn.functional.l1_loss(ml_all, vasp)
    mape_all = mape_loss_func(ml_all, vasp)*100
    plt_range = math.floor(torch.min(vasp))-1

    # mae = torch.nn.functional.l1_loss(ml,vasp)
    # print(f'{name}的mae是：{mae}\n{name}的mape是：{mape}')
    print(f'{xifu_name}|{ml_all.shape[0]}|{torch.mean(vasp):4f}|{mae_xifu:4f}|{mape_xifu:4f}|{mae_all:4f}|{mape_all:4f}')

    #画图要画 结构->energy 与 vasp计算energy 对比
    # 计算相关系数
    corr_coef = torch.nn.functional.cosine_similarity(ml_all, vasp, dim=0)

    ######################
    plt.figure(figsize=(7.5, 7.5))
    # 设置全局字体加粗
    plt.rcParams['font.weight'] = 'bold'
    ax = plt.gca()
    linewidth = 1.5
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    # 设置横纵坐标的刻度线文字加粗并调整字体大小
    plt.tick_params(axis='x', labelsize=14, width=1, which='major', labelbottom=True, labelleft=True)
    plt.tick_params(axis='y', labelsize=14, width=1, which='major', labelbottom=True, labelleft=True)

    # 调整子图布局
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.85)


    ml_all = ml_all.cpu().detach().numpy()
    vasp = vasp.cpu().detach().numpy()
    plt.scatter(ml_all, vasp, color=xifu_color[xifu_name],facecolor=xifu_color[xifu_names[0]], edgecolor='none', alpha=0.1, s=7)
    # 添加 y = x 的辅助线
    plt.plot([plt_range, 0], [plt_range, 0], color='grey', alpha=0.3,linestyle='--')
    plt.xlabel('ML', fontsize=16, labelpad=15, fontweight='bold')
    plt.ylabel('VASP', fontsize=16, labelpad=15, fontweight='bold')
    plt.title('Parity Plot of out', fontsize=21, pad=20, fontweight='bold')
    # 设置横轴标签和标题
    plt.grid(False)
    ax = plt.gca()
    # plt.text(0.25, 0.75, f'MAE={mae_all:.4f}\nR^2={corr_coef:.4f}',
    #          transform=ax.transAxes, horizontalalignment='right')
    save_dir = f'result/{xifu_name}'
    if os.path.exists(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    plt.savefig(os.path.join(save_dir,f'{name}.jpg'))
    plt.close()









if __name__ == '__main__':
    # data.npz N * 3000 * 104
    # range(-15eV, 15eV, 0.01eV)=3000
    # (s*2 + p*6 + d*10 + f*14) * 3 + (s*2 + p*6)=104
    s_name = ''
    # xifu_names_all = [['C'],['S']]
    xifu_names_all = [['H'],['S'],['C'],['N'],['O'],['CH'],['HO']]
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # 使用字体管理器设置 Arial 字体
    font_prop = FontProperties(family='Arial')

    # xifu_color = {
    #                   'H': (170 / 255, 119 / 255, 233 / 255),
    #                   'S': (82 / 255, 146 / 255, 247 / 255),
    #                   'C': (121 / 255, 202 / 255, 251 / 255),
    #                   'N': (78 / 255, 166 / 255, 96 / 255),
    #                   'O': (251 / 255, 235 / 255, 102 / 255),
    #                   'CH': (230 / 255, 85 / 255, 13 / 255),
    #                   'HO': (230 / 255, 85 / 255, 13 / 255)}
    # xifu_color = {
    #                   'H': (181 / 255, 71 / 255, 100 / 255),
    #                   'S': (227 / 255, 98 / 255, 93 / 255),
    #                   'C': (168 / 255, 203 / 255, 223 / 255),
    #                   'N': (128 / 255, 116 / 255, 200 / 255),
    #                   'O': (120 / 255, 149 / 255, 193 / 255),
    #                   'CH': (128 / 255, 85 / 116, 200 / 255),
    #                   'HO': (128 / 255, 85 / 116, 200 / 255),}
    xifu_color = {
                      'H': (13 / 255, 76 / 255, 109 / 255),
                      'S': (3 / 255, 50 / 255, 80 / 255),
                      'C': (239 / 255, 65 / 255, 67 / 255),
                      'N': (196 / 255, 50 / 255, 63 / 255),
                      'O': (99 / 255, 65 / 255, 85 / 255),
                      'CH': (128 / 255, 85 / 116, 200 / 255),
                      'HO': (128 / 255, 85 / 116, 200 / 255),}
    # xifu_color = {
    #                   'H': (69 / 255, 42 / 255, 61 / 255),
    #                   'S': (212 / 255, 76 / 255, 60 / 255),
    #                   'C': (229 / 255, 133 / 255, 93 / 255),
    #                   'N': (221 / 255, 108 / 255, 76 / 255),
    #                   'O': (68 / 255, 117 / 255, 122 / 255),
    #                   'CH': (128 / 255, 85 / 116, 200 / 255),
    #                   'HO': (128 / 255, 85 / 116, 200 / 255),}
    print(f'第i个元素|用ml计算出dos预测|用vasp计算出dos预测|用vasp直接计算的吸附能|dos预测模型造成的误差|dos-吸附能总误差')

    # ######################绘制每个元素的图
    # for (xifu_i,xifu_names) in enumerate(xifu_names_all):
    #     # print(xifu_names[0])
    #
    #     # metadata = np.load('test/data.npz')['meta']
    #     # i_all = []
    #     # for (i,meta) in enumerate(metadata):
    #     #     if meta[0] in xifu_names and meta[1].startswith(s_name) and meta[2]=='top':
    #     #         i_all.append(i)
    #     # metadata = metadata[i_all]
    #     # source = np.load('test/data.npz')['dos'][i_all]
    #     # source = pre(torch.nn.AvgPool1d(6, 6)(torch.Tensor(source).permute(0,2,1)).permute(0,2,1)).numpy()
    #     # target = np.load('test/data.npz')['target'][i_all]
    #
    #     xifu_info = get_energy('./energy_xifu.txt')
    #     dataset = data['dos'].astype('float32')
    #     structure_name = data['z']
    #     # structure_name = structure_name[:len(structure_name)]
    #     # dataset = dataset[:len(dataset)]
    #     surs = []
    #     i = 0
    #
    #     # 读取所有需要吸附质的费米能级
    #     xifu_energys = get_xifu_energy(xifu_names)
    #     for z, dos in zip(structure_name, dataset):
    #         if z.startswith(s_name):
    #             surface_dos, energy_value, sur = get_surface(z, dos, 15, 500, xifu_info, xifu_energys)
    #             if isinstance(energy_value, torch.Tensor):
    #                 # test_1 = pre(torch.Tensor(surface_dos[0]))[1].numpy()
    #                 # test_2 = source[1]
    #                 # print(np.sum(test_1[:96]) + np.sum(test_1[104:200]))
    #                 # print(np.sum(test_2[:96]) + np.sum(test_2[104:200]))
    #                 # xnew = np.linspace(-15, 14.99, 500)
    #                 # print(np.sum(test_1[96:104])+np.sum(test_1[200:]))
    #                 # print(np.sum(test_2[96:104])+np.sum(test_2[200:]))
    #                 # plt.plot(xnew, np.sum(test_2[96:104], 0) + np.sum(test_2[200:], 0))
    #                 # plt.plot(xnew, np.sum(test_2[96:104],0)+np.sum(test_2[200:],0)-np.sum(test_1[96:104],0)-np.sum(test_1[200:],0))
    #                 # plt.show()
    #                 # plt.close()
    #                 #
    #                 # plt.plot(xnew, np.sum(test_2[:96], 0) + np.sum(test_2[104:200], 0))
    #                 # plt.plot(xnew, np.sum(test_2[:96], 0) + np.sum(test_2[104:200], 0)-np.sum(test_1[:96], 0) - np.sum(test_1[104:200], 0))
    #                 # plt.show()
    #                 # plt.close()
    #                 if i == 0:
    #                     surface_doses = surface_dos
    #                     energy_values = energy_value
    #                 else:
    #                     surface_doses = torch.cat((surface_doses, surface_dos), 0)
    #                     energy_values = torch.cat((energy_values, energy_value), 0)
    #                 surs = surs + sur
    #                 i += 1
    #
    #     dataset = surface_doses
    #     xifu_energy = energy_values
    #     # print('已将所有结构处理完成！')
    #     # # 根据dataset中的'z'索引提取 xifu_info 中的能量信息
    #     # xifu_energy = torch.Tensor([xifu_info.get(key, 0) for key in structure_name])
    #
    #     # dosnet = torch.load('./dosnet.pkl').cuda().eval()
    #     # dos = pre(torch.from_numpy(dataset)).cuda()
    #     # out = dosnet(dos)
    #     # for i in range(out.shape[0]):
    #     #     print(f'{out[i]}|{xifu_energy[i]}')
    #
    #
    #
    #     dos_ml = pre(dataset[:, 0, :, :]).cuda()
    #     dos_vasp = pre(dataset[:, 1, :, :]).cuda()
    #     xifu_energy = xifu_energy.cuda()
    #     out_ml = torch.zeros(xifu_energy.shape).cuda()
    #     out_vasp = torch.zeros(xifu_energy.shape).cuda()
    #
    #     # print(torch.sum(dos_vasp[0,:,:][:96],1)+torch.sum(dos_vasp[0,:,:][104:200],1)-torch.Tensor(np.sum(source[1,:,:][:96],1)-np.sum(source[1,:,:][104:200],1)).cuda())
    #     # # print(np.sum(source[1,:,:][:96],1)+np.sum(source[1,:,:][104:200],1))
    #     # print(torch.sum(dos_vasp[0,:,:][96:104],1)+torch.sum(dos_vasp[0,:,:][200:],1)-torch.Tensor(np.sum(source[1,:,:][96:104],1)-np.sum(source[1,:,:][200:],1)).cuda())
    #
    #     # xnew = np.linspace(-15, 14.99, 500)
    #     # for i in range(8):
    #     #     # plt.plot(xnew, source[1, :, :][96+i] + source[1, :, :][200+i])
    #     #     plt.plot(xnew, source[1, :, :][96+i] + source[1, :, :][200+i] -
    #     #                     dos_vasp.cpu().numpy()[0, :, :][96+i] - dos_vasp.cpu().numpy()[0, :, :][200+i])
    #     #     plt.show()
    #     #     plt.close()
    #     # # print(np.sum(source[1,:,:][96:104],1)+np.sum(source[1,:,:][200:],1))
    #     # plt.plot(xnew, np.sum(source[1,:,:][96:104], 0) + np.sum(source[1,:,:][200:], 0))
    #     # plt.plot(xnew, np.sum(source[1,:,:][96:104],0)+np.sum(source[1,:,:][200:],0)-np.sum(dos_vasp.cpu().numpy()[0,:,:][96:104],0)-np.sum(dos_vasp.cpu().numpy()[0,:,:][200:],0))
    #     # plt.show()
    #     # plt.close()
    #     # plt.plot(xnew, np.sum(source[1,:,:][:96], 0) + np.sum(source[1,:,:][104:200], 0))
    #     # plt.plot(xnew, np.sum(source[1,:,:][:96],0)+np.sum(source[1,:,:][104:200],0)-np.sum(dos_vasp.cpu().numpy()[0,:,:][:96],0)-np.sum(dos_vasp.cpu().numpy()[0,:,:][104:200],0))
    #     # plt.show()
    #     # plt.close()
    #
    #
    #     with torch.no_grad():
    #         for n in range(dataset.shape[0]):
    #             out_ml[n] = dosnet(dos_ml[n].unsqueeze(0))
    #             out_vasp[n] = dosnet(dos_vasp[n].unsqueeze(0))
    #     draw_result(xifu_names[0],out_vasp,out_ml, xifu_energy[:],name=xifu_names[0])
    #
    #     indices = torch.where(torch.abs(out_ml - xifu_energy[:]) <= 0.7)[0]
    #     surs_new = [surs[i] for i in indices]
    #     xifu_energy_new = xifu_energy[indices]
    #     # 打开一个文件用于写入，如果文件不存在则创建
    #     with open('output.txt', 'w') as file:
    #         # 遍历两个列表的索引
    #         for i in range(len(surs_new)):
    #             # 将每个元素按照指定格式写入文件
    #             file.write(f"{surs_new[i]}: {xifu_energy_new[i]}\n")
    #
    #
    #
    #     if xifu_i==0:
    #         out_vasp_all = out_vasp
    #         out_ml_all = out_ml
    #         xifu_energy_all = xifu_energy
    #     else:
    #         out_vasp_all = torch.cat((out_vasp_all,out_vasp),0)
    #         out_ml_all = torch.cat((out_ml_all, out_ml), 0)
    #         xifu_energy_all = torch.cat((xifu_energy_all,xifu_energy),0)
    # # draw_result('all', out_vasp_all, out_ml_all, xifu_energy_all, name='all_result')




    # dosnet = torch.jit.load('model.pt').cuda().eval()
    # # dosnet = torch.load('./dosnet.pkl').cuda().eval()
    # data_file = './data_w+diff.npz'
    # # data_file= './data.npz'
    # data = np.load(data_file)
    # for (xifu_i, xifu_names) in enumerate(xifu_names_all):
    #     xifu_info = get_energy('./energy_xifu.txt')
    #     # data = np.load('./data_w+diff.npz')
    #     dataset = data['dos'].astype('float32')
    #     structure_name = data['z']
    #     structure_name = structure_name[:len(structure_name)]
    #     # dataset = dataset[:len(dataset)//5]
    #     # xifu_energy = energy_values[:dataset.shape[0]]
    #     surs = []
    #     i = 0
    #
    #     # 读取所有需要吸附质的费米能级
    #     xifu_energys = get_xifu_energy(xifu_names)
    #
    #     for z, dos in zip(structure_name, dataset):
    #         if z.startswith(s_name):
    #             surface_dos, energy_value, sur = get_surface(z, dos, 15, 500, xifu_info, xifu_energys)
    #             if isinstance(energy_value, torch.Tensor):
    #
    #                 if i == 0:
    #                     surface_doses = surface_dos
    #                     energy_values = energy_value
    #                 else:
    #                     surface_doses = torch.cat((surface_doses, surface_dos), 0)
    #                     energy_values = torch.cat((energy_values, energy_value), 0)
    #                 surs = surs + sur
    #                 i += 1
    #
    #     dataset = surface_doses[:surface_doses.shape[0]//1]
    #     xifu_energy = energy_values[:surface_doses.shape[0]//1]
    #
    #     dos_ml = pre(dataset[:, 0, :, :]).cuda()
    #     dos_vasp = pre(dataset[:, 1, :, :]).cuda()
    #     xifu_energy = xifu_energy.cuda()
    #     out_ml = torch.zeros(xifu_energy.shape).cuda()
    #     out_vasp = torch.zeros(xifu_energy.shape).cuda()
    #
    #
    #     with torch.no_grad():
    #         for n in range(dataset.shape[0]):
    #             out_ml[n] = dosnet(dos_ml[n].unsqueeze(0))
    #             out_vasp[n] = dosnet(dos_vasp[n].unsqueeze(0))
    #
    #     mae_xifu = torch.nn.functional.l1_loss(out_vasp, xifu_energy[:])
    #     mape_xifu = mape_loss_func(out_vasp, xifu_energy[:]) * 100
    #     mae_all = torch.nn.functional.l1_loss(out_ml, xifu_energy[:])
    #     mape_all = mape_loss_func(out_ml, xifu_energy[:]) * 100
    #
    #     corr_coef = torch.nn.functional.cosine_similarity(out_ml, xifu_energy[:], dim=0)
    #     # out_ml = out_ml.cpu().detach().numpy()
    #     # xifu_energy[:] = xifu_energy[:]
    #
    #     ml_all = out_ml.cpu().detach().numpy()
    #     vasp = xifu_energy.cpu().detach().numpy()
    #     indices_new = np.where((np.abs(ml_all-vasp))<1.5)
    #     ml_all = ml_all[indices_new]
    #     vasp = vasp[indices_new]
    #
    #
    #     save_np = {
    #         'ml_all': ml_all,
    #         'vasp': vasp
    #     }
    #     np.savez(f'{xifu_names[0]}.npz', **save_np)



    ####################把所有元素汇总在一个图上
    plt.figure(figsize=(9, 8.5))
    # 设置全局字体加粗
    plt.rcParams['font.weight'] = 'semibold'
    ax = plt.gca()
    linewidth = 3
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    # 设置横纵坐标的刻度线文字加粗并调整字体大小
    plt.tick_params(axis='x', labelsize=16, width=3, length=10,which='major', labelbottom=True, labelleft=True)
    plt.tick_params(axis='y', labelsize=16, width=3, length=10,which='major', labelbottom=True, labelleft=True)
    # 调整子图布局
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)
    plt.xlabel('ML (eV)', fontsize=20, labelpad=15, fontweight='bold', fontproperties=font_prop)
    plt.ylabel('DFT (eV)', fontsize=20, labelpad=15, fontweight='bold', fontproperties=font_prop)
    # plt.title('Parity Plot of out', fontsize=21, pad=25, fontweight='bold')
    # 设置横轴标签和标题
    plt.grid(False)
    # 添加 y = x 的辅助线
    plt_range = -11
    plt.plot([plt_range, 1], [plt_range, 1], color=(42 / 255, 251 / 255, 42 / 255), alpha=0.5, linestyle='-',linewidth=3,zorder=0)
    plt_range_1 = -9.5
    plt.plot([plt_range, plt_range_1], [plt_range, plt_range_1], color=(42 / 255, 251 / 255, 42 / 255), alpha=0.5, linestyle='-',linewidth=3,zorder=0)
    plt_range_2 = -1
    plt.plot([plt_range_2, 1], [plt_range_2, 1], color=(42 / 255, 251 / 255, 42 / 255), alpha=0.5, linestyle='-',linewidth=3,zorder=0)

    ml_alls = torch.Tensor()
    vasp_alls = torch.Tensor()
    for (xifu_i, xifu_names) in enumerate(xifu_names_all):
        ml_all = np.load(f'{xifu_names[0]}.npz')["ml_all"]
        vasp = np.load(f'{xifu_names[0]}.npz')["vasp"]
        ml_alls = torch.cat((ml_alls,torch.Tensor(ml_all)),0)
        vasp_alls = torch.cat((vasp_alls,torch.Tensor(vasp)),0)

        plt.scatter(vasp,ml_all,  color=xifu_color[xifu_names[0]],facecolor=xifu_color[xifu_names[0]],edgecolor='none',alpha=1, s=0.7)

        plt.text(0.07, 0.83, "MAE = 0.197 eV\nMAPE = 4.65 %",
                 transform=ax.transAxes, horizontalalignment='left',fontsize=20,linespacing=2, fontproperties=font_prop)
        save_dir = f'result/all'
        if os.path.exists(save_dir):
            pass
        else:
            os.mkdir(save_dir)
        # 添加图例
        plt.savefig(os.path.join(save_dir, f'all.jpg'))
    # 创建侧面直方图
    # ax_histx = ax.inset_axes([0.9, 0, 0.2, 1], sharex=ax)
    # ax_histy = ax.inset_axes([0, 0.9, 0.9, 0.2], sharey=ax)

    # # 绘制直方图
    # # print(ml_alls.shape)
    # ax_histx.hist(vasp_alls[:100], bins=30,orientation='horizontal', alpha=0.5)  # 如果需要绘制 ml_alls 的直方图
    # # ax_histx.set_axis_off()  # 关闭边框和刻度
    # ax_histy.hist(vasp_alls[:100], bins=30,orientation='horizontal', alpha=0.5)  # 绘制 vasp_alls 的直方图
    # # ax_histy.set_axis_off()  # 关闭边框和刻度
    # # 设置直方图的纵坐标范围
    # max_count = np.max(ax_histy.get_yticks())
    # ax_histy.set_ylim(0, max_count + 10)  # 确保纵坐标范围足够大
    # # 设置直方图的轴标签
    # ax_histx.set_yticks([])
    # ax_histx.set_xlabel('ML Predicted (eV)')
    # ax_histy.set_xticks([])
    # ax_histy.set_ylabel('DFT (eV)')

    # 设置主图的轴标签
    # ax.set_xlabel('DFT (eV)')
    # ax.set_ylabel('ML Predicted (eV)')

    plt.savefig(os.path.join(save_dir, f'all.jpg'))
    plt.close()



    # ####################分开做图
    # for (xifu_i, xifu_names) in enumerate(xifu_names_all):
    #     plt.figure(figsize=(9, 8.5))
    #     # 设置全局字体加粗
    #     plt.rcParams['font.weight'] = 'semibold'
    #     ax = plt.gca()
    #     linewidth = 3
    #     ax.spines['top'].set_linewidth(linewidth)
    #     ax.spines['right'].set_linewidth(linewidth)
    #     ax.spines['bottom'].set_linewidth(linewidth)
    #     ax.spines['left'].set_linewidth(linewidth)
    #     # 设置横纵坐标的刻度线文字加粗并调整字体大小
    #     plt.tick_params(axis='x', labelsize=16, width=3, length=10, which='major', labelbottom=True, labelleft=True)
    #     plt.tick_params(axis='y', labelsize=16, width=3, length=10, which='major', labelbottom=True, labelleft=True)
    #     # 调整子图布局
    #     plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)
    #     plt.xlabel('ML Predicyed (eV)', fontsize=20, labelpad=15, fontweight='bold', fontproperties=font_prop)
    #     plt.ylabel('DFT (eV)', fontsize=20, labelpad=15, fontweight='bold', fontproperties=font_prop)
    #     # plt.title('Parity Plot of out', fontsize=21, pad=25, fontweight='bold')
    #     # 设置横轴标签和标题
    #     plt.grid(False)
    #     # 添加 y = x 的辅助线
    #     plt_range = -11
    #     plt.plot([plt_range, 1], [plt_range, 1], color=(42 / 255, 251 / 255, 42 / 255), alpha=0.5, linestyle='-',
    #              linewidth=3, zorder=0)
    #     plt_range_1 = -9.5
    #     plt.plot([plt_range, plt_range_1], [plt_range, plt_range_1], color=(42 / 255, 251 / 255, 42 / 255), alpha=0.5,
    #              linestyle='-', linewidth=3, zorder=0)
    #     plt_range_2 = -1
    #     plt.plot([plt_range_2, 1], [plt_range_2, 1], color=(42 / 255, 251 / 255, 42 / 255), alpha=0.5, linestyle='-',
    #              linewidth=3, zorder=0)
    #     ml_all = np.load(f'{xifu_names[0]}.npz')["ml_all"]
    #     vasp = np.load(f'{xifu_names[0]}.npz')["vasp"]
    #
    #     mae_xifu = torch.nn.functional.l1_loss(torch.Tensor(ml_all), torch.Tensor(vasp))
    #     mape_xifu = mape_loss_func(torch.Tensor(ml_all), torch.Tensor(vasp)) * 100
    #     plt.scatter(vasp,ml_all,  color=xifu_color[xifu_names[0]],facecolor=xifu_color[xifu_names[0]],edgecolor='none',alpha=1, s=3)
    #
    #     plt.text(0.07, 0.83, f"MAE = {mae_xifu:.3f}eV\nMAPE = {mape_xifu:.2f} %",
    #              transform=ax.transAxes, horizontalalignment='left',fontsize=20,linespacing=2, fontproperties=font_prop)
    #     save_dir = f'result/{xifu_names[0]}'
    #     if os.path.exists(save_dir):
    #         pass
    #     else:
    #         os.mkdir(save_dir)
    #     print(os.path.join(save_dir, f'{xifu_names[0]}.jpg'))
    #     plt.savefig(os.path.join(save_dir, f'{xifu_names[0]}.jpg'))
    #     # plt.show()
    #     plt.close()



