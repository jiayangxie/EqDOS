from tqdm import tqdm
import argparse
from scipy.stats import rankdata
import time

from ..tools.Dataset import Dataset
from .feature_filtering import feature_filtering
from .feature_prepare import get_B_func,c_get,indices_get
from .poscar_deal import *
from .chg_deal import *
from collections import defaultdict


##Selects edges with distance threshold and limited number of neighbors
def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr

##Slightly edited version from pytorch geometric to create edge from gaussian basis
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

def density_get(i,patch_num,POSCAR_only,dir_name,c_list,typedict,n_num,pbar,over,max_size,r_cut,num_l,num_ml,indices_dict,test,spin,feature_filter,patch_offset,feature_all,atoms_dict,use_sparse):
    if atoms_dict == None:
        POS_data_file = dir_name + f'{i}/POSCAR'
        abc,xyz,n,atom_conb,chemical_symbols,edge_index, shifts, unit_shifts = POSCAR_read(POS_data_file,r_cut)
        weighted_onehot, typedict ,val_onehot= n_onehot(n, typedict, n_num)
        ## 为球谐函数做准备
        # 用于转换：原本球谐函数格式【(m=0,l=0)...】-torch.size[25](备注：要补零，所以末尾增加一行)    到     def2-universal-jfit-decontract.bas格式-torch.size[107]
        n_indices = torch.zeros(n.shape[0], num_ml)
        for key, value in indices_dict.items():
            n_indices[torch.where(n == int(key))] = value
        # def2-universal-jfit-decontract.bas里计算径函数的aerfa参数
        aerfa = torch.zeros(n.shape[0], num_ml,2)
        for key, value in c_list.items():
            aerfa[torch.where(n == int(key))] = value[:,1:]
        # beta = torch.zeros(n.shape[0], num_ml)
        # for key, value in c_list.items():
        #     beta[torch.where(n == int(key))] = value[:,2]
        if test:
            atoms_dict = {"xyz": xyz,
                          "n": n,
                          "abc": abc,
                          "atom_conb": atom_conb,
                          "chemical_symbols": chemical_symbols,
                          "n_indices": n_indices,
                          "edge_index": edge_index,
                          "shifts": shifts,
                          "unit_shifts": unit_shifts,
                          "aerfa":aerfa,
                          "weighted_onehot":weighted_onehot,
                          "val_onehot":val_onehot,
                          }
            # torch.save(atoms_dict, atoms_file)
    else:
        xyz,n,abc,atom_conb,chemical_symbols,n_indices,edge_index,shifts,unit_shifts,aerfa,weighted_onehot,val_onehot = atoms_dict.values()
    xyz_num, Z_dict, val_ele, type = atom_deal(abc,xyz,chemical_symbols)
    # edge_veg = None
    edge_num = [edge_index.shape[1]]
    cell_size = torch.Tensor([np.dot(np.cross(abc[0], abc[1]), abc[2])]).unsqueeze(0)
    # cell_size = torch.cat((abc.reshape(-1).unsqueeze(0),cell_size),1)
    # cell_size = torch.Tensor([val_ele]).unsqueeze(0)

    num_atom = [int(len(n))]

    if POSCAR_only == False:
        CHG_data_file = dir_name + f'{i}/CHG'
        if test==False and os.path.exists(dir_name + f'{i}/density_tot.pth'):
            over = -1
        charge_num, copy_line, density,feature = CHG_read(CHG_data_file,xyz_num,patch_num,patch_offset,spin,over,test)
        if use_sparse and test==False:
            non_zero_density = torch.where(torch.abs(torch.sum(density,1)) >= 1e-1)
            density = density[non_zero_density]
            feature = feature[non_zero_density]
        # 点位总数，用于处理loss
        charge_num_all = charge_num[0] * charge_num[1] * charge_num[2] / (patch_num ** 3)
        charge_num_all = [charge_num_all * val_ele]

        if max_size!=None:
            if over == 0:
                pbar = tqdm(total=feature.shape[0] // max_size)
            if feature.shape[0] <= max_size * (over + 1):
                density = density[(max_size * over):min(max_size * (over + 1), density.shape[0])]
                feature = feature[(max_size * over):min(max_size * (over + 1), feature.shape[0])]
                pbar.update(1)
                if over != 0:
                    copy_line = '0'
                over = -1
            else:
                density = density[(max_size * over):min(max_size * (over + 1), density.shape[0])]
                feature = feature[(max_size * over):min(max_size * (over + 1), feature.shape[0])]
                pbar.update(1)
                if feature.shape[0] < max_size:
                    over = 0
                    copy_line = '0'
                else:
                    if over != 0:
                        # 0 表示不需要进行前面结构的写入
                        copy_line = '0'
                    over = over + 1
        # bond_dec,bond_center = bond_get(R,bond_center,key_length)
        d_idxs = [feature.size(0)]
        feature = get_B_func(n,n_indices,xyz,torch.matmul(feature / torch.tensor([charge_num[0],charge_num[1],charge_num[2]]), abc), aerfa,abc,r_cut,num_l,num_ml,use_sparse)
        # print(torch.sum(feature[:,-1,:feature.shape[2]//2],0))
        return n,xyz,abc, torch.Tensor(weighted_onehot),torch.Tensor(val_onehot),num_atom,cell_size,charge_num_all,feature,copy_line,d_idxs,density, edge_num, edge_index, shifts,unit_shifts, (atom_conb, type),over ,typedict,pbar,None,None
    else:
        charge_num = (torch.ceil(torch.sqrt(torch.sum(abc*abc,dim=1))*14.2/4)*4).int().tolist()
        charge_num = [cn // patch_num for cn in charge_num]
        # 点位总数
        charge_num_all = charge_num[0] * charge_num[1] * charge_num[2]
        copy_line = '\n' + str(charge_num[0]) + ' ' + str(charge_num[1]) + ' ' + str(charge_num[2]) + '\n'

        if feature_all == None:
            X, Y, Z = torch.meshgrid(torch.arange(charge_num[2]), torch.arange(charge_num[1]), torch.arange(charge_num[0]), indexing='ij')
            feature_all = torch.stack((Z, Y, X), dim=-1).reshape(-1,3)

        if max_size!=None:
            if over == 0:
                pbar = tqdm(total=charge_num_all // max_size)
            # pbar.update(1)
            if charge_num_all <= max_size * (over + 1):
                # feature = np.arange(max_size)
                # feature = feature[:min(max_size, charge_num_all % max_size)]+ (max_size * over)
                feature = feature_all[(max_size * over):min(max_size * (over + 1), feature_all.shape[0])]
                pbar.update(1)
                if over != 0:
                    copy_line = '0'
                over = -1
            else:
                # feature = np.arange(max_size)
                # feature = feature[:max_size]+ (max_size * over)

                feature = feature_all[(max_size * over):min(max_size * (over + 1), feature_all.shape[0])]
                pbar.update(1)
                if feature.shape[0] < max_size:
                    over = 0
                    copy_line = '0'
                else:
                    if over != 0:
                        # 0 表示不需要进行前面结构的写入
                        copy_line = '0'
                    over = over + 1

        charge_num_all = [charge_num_all * val_ele]

        # 把点位转换成绝对位置
        feature = torch.matmul(feature / torch.tensor([charge_num[0], charge_num[1], charge_num[2]]), abc)

        if feature_filter and test and (max_size!=None):
            features_indices, = feature_filtering(feature,abc)
            if features_indices.shape[0] != 0:
                features = torch.zeros(max_size, num_atom[0], num_ml * 2)
                # 把点位转换成与原子位置相关的 径向基函数 * 球谐函数
                features[features_indices]  = get_B_func(n, n_indices, xyz, feature[features_indices], aerfa, abc, r_cut, num_l, num_ml,use_sparse)
            else:
                features = None
        else:
            # 把点位转换成与原子位置相关的 径向基函数 * 球谐函数
            features = get_B_func(n,n_indices,xyz, feature, aerfa,abc,r_cut,num_l,num_ml,use_sparse)
        d_idxs = [feature.shape[0]]
        return n,xyz, abc, weighted_onehot,num_atom, cell_size, charge_num_all, features, copy_line, d_idxs, None, edge_num, edge_index, shifts,unit_shifts, (atom_conb, type), over, typedict, pbar,feature_all,atoms_dict


def density_prepare(sys_name,density_num,patch_num,n_num,r_cut,data_start = 0,spin=False,save_pth=True,POSCAR_only=False,test=False,over=0,max_size=None,pbar = None,feature_filter=False,patch_offset=torch.IntTensor([0,0,0]),feature_all = None,atoms_dict=None,use_sparse=False):
    if test==False:
        dir_name = 'mydata/'+'density/'+sys_name+'/POSCAR_ALL/'
        # dir_name = 'density/' + sys_name + '/POSCAR_ALL/'
    else:
        dir_name = 'data_to_be_predicted/'+'density/'+sys_name+'/POSCAR_ALL/'
    if test:
        c_save = torch.load(os.path.join('../result/save_model',sys_name,'c.pth'))
        c_list = c_save['c_list']
        l_dict = c_save['l_dict']
    else:
        c_list, l_dict = c_get('mydata/' + 'density/' + sys_name + '/POSCAR_ALL/', 'def2-universal-jfit-decontract.bas')
        # c_list, l_dict = c_get('mydata/' + 'density/' + sys_name + '/POSCAR_ALL/', 'def2-universal-jkfit.bas')
        # c_list, l_dict = c_get('mydata/' + 'density/' + sys_name + '/POSCAR_ALL/', 'pcSseg-2.bas')
        # c_list, l_dict = c_get('mydata/' + 'density/' + sys_name + '/POSCAR_ALL/', 'pcSseg-3.bas')
        # c_list, l_dict = c_get('mydata/' + 'density/' + sys_name + '/POSCAR_ALL/', 'pcSseg-4.bas')
        c_save = {'c_list':c_list,'l_dict':l_dict}
        torch.save(c_save,os.path.join('../result/save_model',sys_name,'c.pth'))
        torch.save(c_save, os.path.join('mydata/density/', sys_name, 'data_analysis/c.pth'))
    num_l = len(l_dict.keys())
    num_ml = sum(list(l_dict.values()))
    indices_dict = indices_get(l_dict,c_list)
    if test:
        i = density_num
        density_num = 1
    else:
        i = data_start
    j = 0
    xyzs = [[] for i in range(density_num)]
    abc = [[] for i in range(density_num)]
    weighted_onehots = [[] for i in range(density_num)]
    val_onehots = [[] for i in range(density_num)]
    ns = [[] for i in range(density_num)]
    num_a = [[] for i in range(density_num)]
    features = [[] for i in range(density_num)]
    copy_lines = [[] for i in range(density_num)]
    d_idxs = [[] for i in range(density_num)]
    cell_fin_sizes = [[] for i in range(density_num)]
    charge_nums = [[] for i in range(density_num)]
    densitys = [[] for i in range(density_num)]
    edge_num = [[] for i in range(density_num)]
    edge_index = [[] for i in range(density_num)]
    # edge_veg = [[] for i in range(density_num)]
    shifts = [[] for i in range(density_num)]
    unit_shifts = [[] for i in range(density_num)]
    cell_and_atom_type_dict = {}
    if test==False:
        pbar = tqdm(total=density_num)
        typedict = {}
        if sys_name == 'TiOH':
            typedict = typeread(sys_name)
    else:
        typedict = typeread(sys_name)
    while True:
        if os.path.exists(f'{dir_name+str(i)}'):
            ns[j], xyzs[j], abc[j],weighted_onehots[j],val_onehots[j], num_a[j], cell_fin_sizes[j], charge_nums[j], features[j],copy_lines[j], d_idxs[j],densitys[j], edge_num[j],edge_index[j],shifts[j],unit_shifts[j],cell_and_atom,over,typedict,pbar,feature_all,atoms_dict = density_get(i, patch_num, POSCAR_only, dir_name,c_list,typedict,n_num,pbar,over,max_size,r_cut,num_l,num_ml,indices_dict,test,spin,feature_filter,patch_offset,feature_all,atoms_dict,use_sparse)
            cell_and_atom_type_dict[cell_and_atom] = cell_and_atom_type_dict.get(cell_and_atom, 0) + 1
            i = i + 1
            if xyzs[j] != None:
                if j == density_num - 1:
                    break
                j = j + 1
                if test==False:
                    pbar.update(1)
            else:
                continue
        else:
            del xyzs[j - density_num:]
            del abc[j - density_num:]
            del weighted_onehots[j - density_num:]
            del val_onehots[j - density_num:]
            del ns[j - density_num:]
            del num_a[j - density_num:]
            del cell_fin_sizes[j - density_num:]
            del charge_nums[j - density_num:]
            del features[j - density_num:]
            del copy_lines[j - density_num:]
            del d_idxs[j - density_num:]
            del densitys[j - density_num:]
            del edge_num[j-density_num:]
            del edge_index[j-density_num:]
            # del edge_veg[j-density_num:]
            del shifts[j-density_num:]
            del unit_shifts[j-density_num:]

            print(f'预计包装{density_num}bulk个结构，但文件夹里只有{j}bulk个结构')
            density_num = j
            break
    dataset = {
        ##s输入数据
        'pos': xyzs,
        'abc': abc,
        'x': weighted_onehots,
        'val_x':val_onehots,
        'z': ns,
        'num_a': num_a,
        'cell_fin_size': cell_fin_sizes,
        'feature': features,
        'copy_line': copy_lines,
        'd_idx': d_idxs,
        'charge_num': charge_nums,
        'density': densitys,
        'edge_index': edge_index,
        # 'edge_vec': edge_veg,
        'edge_num': edge_num,
        'shift':shifts,
        'unit_shift':unit_shifts
    }
    if save_pth:
        if os.path.exists('mydata/density/' + sys_name + '/data_analysis/') == False:
            os.mkdir('mydata/density/' + sys_name + '/data_analysis/')
        with open('mydata/density/' + sys_name + '/data_analysis/traindata_analysis.txt', "w+",
                  encoding="utf-8") as f:
            f.write(f"目前数据集用到的结构总类有：\n")
            for atom_conb, value in cell_and_atom_type_dict.items():
                f.write(f'{(atom_conb)}:{value}\n')
    if test:
        pass
    else:
        typedict_save_file = os.path.join('../result/save_model/', sys_name, 'typedict.txt')
        if os.path.exists(typedict_save_file):
            os.remove(typedict_save_file)
        with open(typedict_save_file, 'a') as f:
            for key, value in typedict.items():
                f.write(f'{key} {value}\n')
    if test:
        return dataset,over,pbar,typedict,num_ml,feature_all,atoms_dict
    else:
        return dataset,typedict,num_ml

def dataset_get(sys_name,density_num,patch_num,n_num,r_cut,data_start = 0,spin=False,save_pth=True,POSCAR_only=False,test=False,over=0,max_size=None,pbar = None,feature_filter=False,patch_offsets=torch.IntTensor([[0,0,0]]),feature_all = None,atoms_dict=None,use_sparse=False):
    dataset = defaultdict(list)
    for patch_offset in patch_offsets:
        print(f'-----------正在准备{patch_offset}的数据--------')
        props, z_table, m_l = density_prepare(sys_name=sys_name, density_num=density_num,
                                              patch_num=patch_num, n_num=n_num, r_cut=r_cut,
                                              data_start=data_start, spin=spin, patch_offset=patch_offset,
                                              use_sparse=use_sparse)
        if patch_offsets.shape[0] > 1:
            for key, value in props.items():
                for v in value:
                    dataset[key].append(v)
        else:
            dataset = props
        if save_pth == True:
            if spin:
                torch.save(dataset, 'mydata/density/' + sys_name + '/data_analysis/density_spin.pth.tar')
            else:
                torch.save(dataset, 'mydata/density/' + sys_name + '/data_analysis/density.pth.tar')
    return dataset,z_table, m_l

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--sys_name', type=str)
    parser.add_argument('--density_num',type=int)
    parser.add_argument('--patch_num', type=int)
    parser.add_argument('--save_pth',type=str,default='True')
    parser.add_argument('--POSCAR_only',type=str,default='False')
    parser.add_argument('--test', type=str, default='False')
    parser.add_argument('--max_size',type=int,default=100000)
    args = parser.parse_args()
    if args.save_pth.lower()=='true':
        save_pth = True
    else:
        save_pth = False
    if args.POSCAR_only.lower()=='true':
        POSCAR_only = True
    else:
        POSCAR_only = False
    if args.test.lower()=='true':
        test = True
    else:
        test = False
    n_num = sum(1 for char in args.sys_name if char.isupper())
    density_prepare(sys_name=args.sys_name,density_num=args.density_num, patch_num=args.patch_num,n_num=n_num,save_pth=save_pth,POSCAR_only=POSCAR_only,test=False,over=0,max_size=None)
