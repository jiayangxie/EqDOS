import ase
from .neighborhood import get_neighborhood
import torch
from collections import Counter
import os
import numpy as np

def POSCAR_read(POS_data_file,r_cut):
    atoms = ase.io.read(POS_data_file, format='vasp')
    abc = torch.tensor(np.array(atoms.cell), dtype=torch.float32)
    xyz = torch.tensor(atoms.positions, dtype=torch.float32)
    n = torch.tensor(atoms.get_atomic_numbers(),dtype=torch.float32)
    atom_conb = atoms.get_chemical_formula()
    chemical_symbols = atoms.get_atomic_numbers()
    # edge_index, shifts, unit_shifts = get_neighborhood(positions=atoms.positions, cutoff=5, pbc=atoms.pbc,cell=atoms.cell)
    edge_index, shifts, unit_shifts = get_neighborhood(positions=atoms.positions, cutoff=r_cut, pbc=(True, True, True),cell=atoms.cell, true_self_interaction=True)
    return abc,xyz,n,atom_conb,chemical_symbols,edge_index, shifts, unit_shifts

def get_type(abc,xyz):
    if (abc[0][1] == abc[0][2] ==0
    ) and (abc[1][0] == abc[1][2] == 0
    ) and (abc[2][0] == abc[2][1] == 0
    )and abc[0][0] >= 10:
        return "cluster"
    else:
        # not enough, need to place the cell in center
        max_coord = torch.max(xyz,dim=0)
        min_coord = torch.min(xyz,dim=0)
        coord_space = max_coord[0] - min_coord[0]
        is_vacc = coord_space-0.5 <= 0
        sum_vac_dim = torch.sum(is_vacc)
        if sum_vac_dim >= 3:
            return "cluster"
        elif sum_vac_dim >= 1:
            return "layer"
        else:
            return "bulk"

def atom_deal(abc,xyz,chemical_symbols):
    val = {'8':6,'22':4,'6':4,'1':1,'7':5,'9':7}
    val_pure = {'Ti':6}
    xyz_num = xyz.shape[0]
    type = get_type(abc,xyz)
    element_counts = Counter(chemical_symbols)
    Z_dict = {}
    for element, count in element_counts.items():
        Z_dict[str(element)]=count
    if len(Z_dict.keys())==1:
        key, value = next(iter(Z_dict.items()))  # 获取并解包第一个键值对
        val_ele = val_pure[key] * int(value)
    else:
        val_ele = 0
        # 累加val_ele
        for key, value in Z_dict.items():
            val_ele += val[key] * int(value)
    return xyz_num,Z_dict,val_ele,type

def n_onehot(n,typedict,n_num):
    # atom_dictionary = get_dictionary(
    #     os.path.join(
    #         os.path.dirname(os.path.realpath(__file__)), "dictionary_default.json"
    #     )
    # )
    # onehot = np.vstack([atom_dictionary[str(atoms.get_atomic_numbers()[i])]for i in range(len(atoms))]).astype(float)
    val = {'8':6,'22':4,'6':4,'1':1,'7':5,'9':7}
    onehot = torch.zeros((n.shape[0],n_num))
    onehot_val = torch.zeros((n.shape[0]))
    for i, num in enumerate(n):
        if num.item() not in typedict:
            counter  = len(typedict)
            typedict[num.item()] = counter
        onehot[i, typedict[num.item()]] = num
        onehot_val[i] = val[str(int(num.item()))]
    onehot[onehot > 0.001] = 1
    return onehot,typedict,onehot_val

def typeread(sys_name):
    with open(os.path.join('../result/save_model/', sys_name, 'typedict.txt'), "r", encoding="utf-8") as f:
        dataline = f.read().splitlines()
    typedict={}
    for data in dataline:
        typedict[float(data.split(' ')[0])]=int(data.split(' ')[1])
    return typedict