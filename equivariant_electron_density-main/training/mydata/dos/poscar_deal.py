import torch
import numpy as np
from scipy.stats import rankdata
from ase.io import read
from .neighborhood import get_neighborhood
from torch_geometric.utils import dense_to_sparse, degree,add_self_loops
import json

# def get_dictionary(dictionary_file):
#     with open(dictionary_file) as f:
#         atom_dictionary = json.load(f)
#     return atom_dictionary


# ##Get min/max ranges for normalized edges
# def GetRanges(dataset, descriptor_label):
#     mean = 0.0
#     std = 0.0
#     for index in range(0, len(dataset)):
#         if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
#             if index == 0:
#                 feature_max = dataset[index].edge_descriptor[descriptor_label].max()
#                 feature_min = dataset[index].edge_descriptor[descriptor_label].min()
#             mean += dataset[index].edge_descriptor[descriptor_label].mean()
#             std += dataset[index].edge_descriptor[descriptor_label].std()
#             if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
#                 feature_max = dataset[index].edge_descriptor[descriptor_label].max()
#             if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
#                 feature_min = dataset[index].edge_descriptor[descriptor_label].min()
#
#     mean = mean / len(dataset)
#     std = std / len(dataset)
#     return mean, std, feature_min, feature_max

# def NormalizeEdge(dataset, descriptor_label):
#     mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)
#     for data in dataset:
#         data.edge_descriptor[descriptor_label] = (
#             data.edge_descriptor[descriptor_label] - feature_min
#         ) / (feature_max - feature_min)

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

def n_onehot(n,typedict,n_num):
    # atom_dictionary = get_dictionary(
    #     os.path.join(
    #         os.path.dirname(os.path.realpath(__file__)), "dictionary_default.json"
    #     )
    # )
    # onehot = np.vstack([atom_dictionary[str(atoms.get_atomic_numbers()[i])]for i in range(len(atoms))]).astype(float)
    onehot = torch.zeros((n.shape[0],n_num))
    for i, num in enumerate(n):
        if num.item() not in typedict:
            counter  = len(typedict)
            # if counter>16:
            #     return None,None,n,typedict
            # if len(n.unique())==1:
            #     return None,None,n,typedict
            typedict[num.item()] = counter
        # else:
        #     return None,None,n,typedict
        onehot[i, typedict[num.item()]] = num
    onehot[onehot > 0.001] = 1

    # if model_name == 'CNN':
    #     idx = edge_index[0]
    #     deg = degree(idx, xyz.shape[0], dtype=torch.long)
    #     deg = F.one_hot(deg, num_classes=13 + 1).to(torch.float)
    #     onehot = torch.cat([onehot, deg.to(onehot.dtype)], dim=-1)
    # onehot = np.concatenate((n.unsqueeze(1),onehot),axis=1)
    return torch.Tensor(onehot),typedict

def get_type(abc,xyz):
    if (abc[0][1] == abc[0][2] ==0
    ) and (abc[1][0] == abc[1][2] == 0
    ) and (abc[2][0] == abc[2][1] == 0
    )and abc[0][0] >= 10:
        return "cluster"
    else:
        # not enough, need to place the cell in center
        max_coord = torch.max(xyz, dim=0)
        min_coord = torch.min(xyz, dim=0)
        coord_space = max_coord[0] - min_coord[0]
        is_vacc = coord_space-0.5 <= 0
        sum_vac_dim = torch.sum(is_vacc)
        if sum_vac_dim >= 3:
            return "cluster"
        elif sum_vac_dim >= 1:
            return "layer"
        else:
            return "bulk"

def poscar_read(POS_data_file):
    # 读取 POSCAR 文件
    atoms = read(POS_data_file, format='vasp')
    xyz_num = len(atoms)
    abc = atoms.cell
    n = torch.Tensor(atoms.get_atomic_numbers()).unsqueeze(1)
    xyz =  torch.Tensor(atoms.positions)
    return  atoms,xyz, n,xyz_num,abc

def atom_deal(n,abc,xyz,atoms,model_name, typedict, n_num):
    if model_name == "mace":
        edge_index, shifts, unit_shifts = get_neighborhood(positions=atoms.positions, cutoff=10, pbc=(True,True,True),
                                                           cell=atoms.cell,true_self_interaction=True)

        # # 添加自己对自己的影响
        # edge_length = torch.zeros(edge_index.shape[1])
        # edge_index, edge_length = add_self_loops(
        #     edge_index, edge_length, num_nodes=len(atoms), fill_value=0
        # )
        # unit_shifts = torch.cat((unit_shifts,torch.zeros(len(atoms),3)),0)
        # shifts = torch.cat((shifts,torch.zeros(len(atoms),3)),0)

        edge_veg = None
    elif model_name == 'CNN':
        ##计算边信息
        ##Obtain distance matrix with ase
        distance_matrix = atoms.get_all_distances(mic=True)
        ##Create sparse graph from distance matrix
        distance_matrix_trimmed = threshold_sort(
            distance_matrix,
            8,
            12,
            adj=False,
        )
        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_length = out[1]
        edge_index, edge_length = add_self_loops(
            edge_index, edge_length, num_nodes=len(atoms), fill_value=0
        )


        max, min = 7.5002, 0
        edge_length = (edge_length - min) / (max - min)
        distance_gaussian = GaussianSmearing(
            0, 1, 50, 0.2
        )
        edge_veg = distance_gaussian(edge_length).to(torch.float)
        # print(edge_veg.shape)
        shifts = None
        unit_shifts = None
    edge_num = [edge_index.shape[1]]
    weighted_onehot, typedict = n_onehot(n, typedict, n_num)
    atom_conb = atoms.get_chemical_formula()
    type = get_type(abc,xyz)
    return weighted_onehot,edge_num,edge_index,edge_veg,shifts,unit_shifts,(atom_conb, type),typedict