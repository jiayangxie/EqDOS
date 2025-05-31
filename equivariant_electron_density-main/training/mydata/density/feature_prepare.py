import torch
import math
# import scipy.special as sp
import torch.nn.functional as F
from e3nn import o3
import itertools
import re
from ase.symbols import symbols2numbers
import sys
import time

# def fea2xyz(feature,charge_num,abc):
#     #进行patch后取patch的顶点，其他点位忽略
#     #第feature个点位->[x,y,z]
#     xyz = np.array([math.ceil(feature % (charge_num[1] * charge_num[0]) % charge_num[0]),
#                     math.ceil((feature  % (charge_num[1] * charge_num[0]) + 1) / charge_num[0])-1,
#                     math.ceil((feature + 1) / (charge_num[1] * charge_num[0]))-1])
#     xyz = np.dot(xyz / charge_num, abc)  ##分子坐标转笛卡尔坐标
#     return xyz

# def fd(r,r_cut):
#     return (0.5*(torch.cos(math.pi*torch.sqrt(r)/r_cut)+1)).unsqueeze(2).repeat(1,1,3)

# def gto_norm(n, a):
#     # normalization factor of function r^n e^{-a r^2}
#     s = 2**(2*n+3) * torch.exp(torch.lgamma(n + 1)) * (2*a)**(n+1.5) \
#             / (torch.exp(torch.lgamma(n + 1)) * torch.sqrt(torch.tensor(math.pi)))
#     return torch.sqrt(s)



#定义径函数
def Rad_Dis(pos,aerfa,r_cut):
    beta = aerfa[:,:,1]
    aerfa = aerfa[:,:,0]
    # norm = gto_norm(ls,aerfa)
    fd=0.5*(torch.cos(math.pi*pos[:,3]/r_cut)+1) * pos[:,3]
    R_func = torch.exp(-aerfa * torch.pow(pos[:,3],2).unsqueeze(1).repeat(1,aerfa.shape[1])) * fd.unsqueeze(1).repeat(1,aerfa.shape[1])
    # R_func = R_func * norm
    R_func[torch.where(aerfa==-1)] = 0.0
    return R_func

def pos_prepare(R,fea,offset,r_cut):
    num_centers = R.size(0)
    n_input = fea.size(0)
    A = R.repeat(n_input, 1, 1)  ##每个原子的坐标   [batch,atom_num,3]
    B = fea.view(n_input, -1).unsqueeze(1).repeat(1, num_centers,1)  ##batch个k点的坐标 [batch,1,3]repeat->[batch,atom_num,3]
    pos = B - A
    pos_prepare = pos.unsqueeze(2).repeat(1,1,27,1)+offset
    r = torch.sqrt(torch.sum(pos_prepare*pos_prepare,dim=3))
    _,pos_id = torch.sort(r,dim = 2)
    pos_prepare = torch.cat((pos_prepare,r.unsqueeze(3)),dim=3)

    pos_id1 = pos_id[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    pos1 = torch.gather(input=pos_prepare, dim=2, index=pos_id1.repeat(1,1,1,4)).squeeze(2)

    indices1 = torch.where(pos1[:,:,3] < r_cut)
    pos1 = pos1[indices1]
    pos_id2 = pos_id[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    pos2 = torch.gather(input=pos_prepare, dim=2, index=pos_id2.repeat(1, 1, 1, 4)).squeeze(2)
    if (offset[0][13]<r_cut).any():
        indices2 = torch.where(pos2[:, :, 3] < r_cut)
    else:
        indices2 = (torch.empty(0).long(),torch.empty(0).long())
    pos2 = pos2[indices2]
    return pos1,indices1,pos2,indices2


# def Y_func_ml(m, l, phi, theta):
#     Y_func = sp.sph_harm(m, l, phi, theta)
#     Y_func_conj = torch.conj(Y_func)
#     Y_func = torch.mul(Y_func, Y_func_conj).real
#     return torch.sqrt(Y_func)


#定义球谐函数
def Y_Fuc(n_indices,pos,num_l):
    Y_func_l_m = torch.Tensor()
    for i in range(num_l):
        Y_func = o3.spherical_harmonics(i, pos[:, :3], True)
        Y_func_l_m = torch.cat((Y_func_l_m,Y_func),dim=1)

    # x = pos[:, 0]
    # y = pos[:, 1]
    # z = pos[:, 2]
    # r = pos[:, 3]
    # x = torch.where(x == 0, torch.tensor(1e-8,dtype=torch.float32), x)  # 如果z为0则把其设为1e-8
    # z = torch.where(z == 0, torch.tensor(1e-8,dtype=torch.float32), z)  # 如果z为0则把其设为1e-8
    # theta = torch.acos(z/r)
    # phi = torch.atan2(y,x)
    # Y_func_l_m = [[] for j in range(num_l * num_l)]
    # for i in range(num_l):
    #     l = i
    #     m = 0
    #     Y_func_l_m[i * i] = Y_func_ml(m, l, phi, theta)
    #     for j in range(1, i + 1):
    #         Y_func_l_m[i * i + 2*j-1] = Y_func_ml(j, l, phi, theta)
    #         Y_func_l_m[i * i + 2*j] = Y_func_ml(-j, l, phi, theta)
    # Y_func_l_m = torch.stack(Y_func_l_m).transpose(0, 1)
    # print(Y_func_l_m[0])

    # 最后一行补0，方便后续筛选def-universal-jfit-decontrct基组为 0 的部分
    Y_func_l_m = F.pad(input=Y_func_l_m, pad=(0, 1), mode='constant', value=0)
    # 开始将 num_l*num_l 转换成 def-universal-jfit-decontrct 基组，为 0 的部分选 Y_func_l_m 最后一行
    Y_func_l_m_all = Y_func_l_m[(torch.arange(n_indices.shape[0]).unsqueeze(1).repeat(1,n_indices.shape[1]), n_indices.long())]
    return Y_func_l_m_all

# 获取基组参数
def c_get(dir_name,c_name):
    import re
    c = {}
    l_list = {'S':0,'P':1,'D':2,'F':3,'G':4,'H':5,'I':6}
    c_path_file = dir_name+c_name
    print(c_path_file)
    with open(c_path_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        values=[]
        cc=0
        max_len = 0
        for i, line in enumerate(lines):
            if line.endswith('     0'):
                key = str(symbols2numbers(re.split(r'\s+',line)[0])[0])
                values=[]
                continue
            if line.endswith('    1   1.00'):
                value = [l_list[re.split(r'\s+',line)[0]]]
                cc = 1
                continue
            if cc==1:
                value.append(float(re.split(r'\s+',line)[1]))
                value.append(float(re.split(r'\s+', line)[2]))
                for m in range(2*value[0]+1):
                    values.append(value)
                cc=0
            if i == len(lines)-1:
                c[key] = torch.Tensor(values)
                sub_len = len(values)
                if sub_len > max_len:
                    max_len = sub_len
                    max_ls = c[key][:,0]
                break
            if (lines[i + 1] == '****'):
                c[key] = torch.Tensor(values)
                sub_len = len(values)
                if sub_len > max_len:
                    max_len = sub_len
                    max_ls = c[key][:,0]
                continue
        max_count_dict = {}
        for num in max_ls:
            if num.item() in max_count_dict:
                max_count_dict[num.item()] += 1
            else:
                max_count_dict[num.item()] = 1
        for key in c:
            if c[key].shape[1]!=max_len:
                sub_tensor = c[key]
                sub_ls = sub_tensor[:, 0]
                count_dict = {}
                for num in sub_ls:
                    if num.item() in count_dict:
                        count_dict[num.item()] += 1
                    else:
                        count_dict[num.item()] = 1
                padding_tensor = torch.zeros(max_len, sub_tensor.shape[1]) - 1  # 用-1填充
                n = 0
                m = 0
                for (l, l_num) in count_dict.items():
                    padding_tensor[n:n+l_num,:] = sub_tensor[m:m+l_num,:]
                    n = int(n + max_count_dict[l])
                    m = int(m + count_dict[l])
                c[key] = padding_tensor
        return c,max_count_dict

# 获取每个原子应该有多少个原子
def indices_get(l_dict,c_list):
    # l = l.long()
    m_dict = {}
    for (key,value) in c_list.items():
        l = value[:,0]
        unique_l, inverse_indices = torch.unique(l, return_inverse=True)
        counts = torch.bincount(inverse_indices)
        # 输出每个数字出现的次数
        """S:1 P:3 D:5 G:7"""
        m = torch.zeros(l.shape) - 1
        # Y_func = torch.zeros(Y_func_l_m.shape[0], l.shape[0])
        j = 0
        for i in range(m.shape[0]):
            if i != j:
                # 仍然在spdf
                continue
            if l[j] == -1:
                continue
            m_num_ = j + counts[torch.where(unique_l == l[j])].item()
            m[j:m_num_] = torch.cat([l[j] * (l[j] + 1) + torch.arange(-l[j], l[j] + 1, 1)] * int(counts[torch.where(unique_l == l[j])].item() / (2 * l[j] + 1)))
            # Y_func[:, j:m_num_] = Y_func_l_m[:, m[j:m_num_].long()]
            j = j + l_dict[l[j].item()]
        m_dict[key] = m
    return m_dict

def get_B_func(n,n_indices,R,fea,aerfa,abc,r_cut,num_l,num_ml,use_sparse):
    offset = torch.tensor(list(itertools.product([0, 1, -1], repeat=3)))
    offset = torch.sum(abc * offset.unsqueeze(2).repeat(1, 1, 1, 3), 2)
    pos1,indices1,pos2,indices2 = pos_prepare(R,fea,offset,r_cut)
    # n_indices 用于转换：原本球谐函数格式【(m=0,l=0)...】-torch.size[25](备注：要补零，所以末尾增加一行)    到     def2-universal-jfit-decontract.bas格式-torch.size[107]
    # n_indices1 再增加一个维度表示 feature 个数
    n_indices1 = n_indices.squeeze(0).repeat(fea.shape[0], 1,1)[indices1]
    aerfa1 = aerfa.squeeze(0).repeat(fea.shape[0],1,1,1)[indices1]
    R_func1 = Rad_Dis(pos1,aerfa1,r_cut)
    Y_func1 = Y_Fuc(n_indices1,pos1,num_l)
    B_func1 = Y_func1 * R_func1
    B_func1_ = torch.zeros(fea.shape[0], n.shape[0], num_ml)
    B_func1_[indices1] = B_func1.to(torch.float32)
    if use_sparse:
        B_func1_ = torch.sparse_coo_tensor(torch.stack(indices1, dim=1).transpose(0, 1), B_func1,[fea.shape[0], n.shape[0], n_indices.shape[1]])
    else:
        B_func1_ = torch.zeros(fea.shape[0], n.shape[0], num_ml)
        B_func1_[indices1] = B_func1.to(torch.float32)

    n_indices2 = n_indices.squeeze(0).repeat(fea.shape[0], 1, 1)[indices2]
    aerfa2 = aerfa.squeeze(0).repeat(fea.shape[0], 1, 1,1)[indices2]
    R_func2 = Rad_Dis(pos2, aerfa2, r_cut)
    Y_func2 = Y_Fuc(n_indices2, pos2, num_l)
    B_func2 = Y_func2 * R_func2
    if use_sparse:
        B_func2_ = torch.sparse_coo_tensor(torch.stack(indices2, dim=1).transpose(0, 1), B_func2,[fea.shape[0], n.shape[0], n_indices.shape[1]])
    else:
        B_func2_ = torch.zeros(fea.shape[0], n.shape[0], num_ml)
        B_func2_[indices2] = B_func2.to(torch.float32)

    B_func_ = torch.cat((B_func1_, B_func2_), dim=2)

    return B_func_