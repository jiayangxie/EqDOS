###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
from e3nn import o3
from e3nn.util.jit import compile_mode
import e3nn
import re
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from typing import Union,Tuple
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch import Tensor
import torch
import torch.multiprocessing as mp

from mace.data import AtomicData
from mace.tools.scatter import scatter_sum

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)

class GC_block(MessagePassing):
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0, aggr: str = 'mean', **kwargs):
        super(GC_block, self).__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        if isinstance(channels, int):
            channels = (channels, channels)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(sum(channels) + dim, channels[1]),
                              torch.nn.PReLU(),
                              )
        self.mlp2 = torch.nn.Sequential(torch.nn.Linear(dim, dim),
                               torch.nn.PReLU(),
                               )
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out += x[1]
        return out
    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        z = torch.cat([x_i, x_j, self.mlp2(edge_attr)], dim=-1)
        z = self.mlp(z)
        return z

#两层卷积层，后面接一个全连接层
class dos_spd(torch.nn.Module):
    def __init__(self,input_dim):
        self.input_dim = input_dim
        super(dos_spd, self).__init__()
        self.split_dos_linear = torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.input_dim+2),
                                               torch.nn.PReLU(),
                                               )
        self.model1 = torch.nn.Sequential(
        	#输入通道一定为1，输出通道为卷积核的个数，2为卷积核的大小（实际为一个[1,2]大小的卷积核）
            torch.nn.Conv1d(1, 16, 2),
            torch.nn.ReLU(),
            # torch.nn.MaxPool1d(2),  # 输出大小：torch.Size([batch_size, 16, 5])
            torch.nn.Conv1d(16, 32, 2),
            torch.nn.ReLU(),
            # torch.nn.MaxPool1d(4),  # 输出大小：torch.Size([batch_size, 32, 1])
            torch.nn.Flatten(),  # 输出大小：torch.Size([batch_size, 32])
        )
    def forward(self, input):
        input = self.split_dos_linear(input)
        x = self.model1(input.unsqueeze(1))
        return x


class dos_split(torch.nn.Module):
    def __init__(self,input_dim,output_dim,max_spd):
        super(dos_split, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_spd = max_spd
        self.scaling_mlp_spd = torch.nn.Sequential(torch.nn.Linear(self.input_dim, 100),
                                                   torch.nn.PReLU(),
                                                   torch.nn.Linear(100, self.max_spd),
                                                   torch.nn.ReLU(),
                                                   )
        # self.split_dos = NonLinearDipoleReadoutBlock(o3.Irreps("20x0e + 10x1o + 10x2e + 10x3o"), o3.Irreps("1000x0e + 1000x1o + 1000x2e + 1000x3o"), torch.nn.Tanh())
        self.split_dos = torch.nn.Sequential(
                                             torch.nn.Linear(self.input_dim,self.output_dim),
                                             torch.nn.PReLU(),
                                             torch.nn.Linear(self.output_dim, self.output_dim*self.max_spd),
                                             torch.nn.Tanh(),
                                             torch.nn.Dropout(p=0.001),
                                               )
    def forward(self, node_feat,spd):
        dos_ml = ((self.split_dos(node_feat)+1)/2).view(-1,self.output_dim,self.max_spd)
        scaling = self.scaling_mlp_spd(node_feat) * spd
        return dos_ml,scaling


# # 定义每个 GPU 的计算任务
# def compute_on_gpu(rank, cell_size, feature_split, x_i, results):
#     device_i = torch.device(f"cuda:{rank}")
#     with torch.cuda.device(device_i):
#         # 将数据移动到当前 GPU
#         f_split = feature_split.detach().to(device_i)
#         x_i_split = x_i.detach().to(device_i)
#         cell_size_split = cell_size.detach().to(device_i)
#         # 在当前 GPU 上计算
#         output_split = (cell_size_split * torch.sum(torch.sum(f_split * x_i_split, dim=2), dim=1)).squeeze().unsqueeze(1)
#         f_split = None
#         x_i_split = None
#         cell_size_split = None
#         # 将结果存储到共享列表中
#         results[rank] = output_split.to(cell_size.device)
#
# def parallel_feature(cell_size, feature_i_up, x_i):
#     """
#     手动实现多卡并行
#     """
#     # 获取可用 GPU 数量
#     num_gpus = torch.cuda.device_count()
#
#     # 分割输入张量
#     batch_size = feature_i_up.size(0)
#     chunk_size = (batch_size + num_gpus - 1) // num_gpus  # 每个 GPU 的数据量，向上取整
#     feature_splits = [feature_i_up[i:i + chunk_size] for i in range(0, batch_size, chunk_size)]
#
#     # 使用多进程并行计算
#     results = mp.Manager().list([None] * num_gpus)  # 创建一个共享列表存储结果
#     processes = []
#     for i in range(num_gpus):
#         p = mp.Process(target=compute_on_gpu, args=(i, cell_size, feature_splits[i], x_i, results))
#         p.start()
#         processes.append(p)
#
#     # 等待所有进程完成
#     for p in processes:
#         p.join()
#
#     # 汇总结果
#     output_tensor = torch.cat(results, dim=0)
#     return output_tensor

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


def convert_irreps_to_dict(irreps_str):
    """
    将 e3nn 的输出格式（如 "32x0e + 32x1o + 32x2e + 32x3o"）转换为字典。
    参数:
        irreps_str (str): e3nn 的输出格式字符串。
    返回:
        dict: 键是不可约表示（irreps），值是对应的特征数量。
    """
    # 分割字符串，获取每个不可约表示的项
    total_features = 0
    terms = irreps_str.split('+')
    for term in terms:
        dim, coeff = term.split('x')
        # 使用正则表达式提取数字部分
        match = re.search(r'\d+', coeff)
        if match:
            coeff = int(match.group(0))  # 将提取的数字部分转换为整数
        total_features += int(dim) * (2*coeff+1)  # 累加特征数量
    return total_features

@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        correlation: int,
        gate: Optional[Callable],
        dropout_rate: float,
        # conv_num: int,
        required_variable: str,
        windows: int,
        m_l: int,
        max_spd: int,
    ):
        super().__init__()
        # self.register_buffer(
        #     "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        # )
        # self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        # self.register_buffer(
        #     "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        # )
        # Embedding\
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )


        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])
        self.windows = windows
        self.m_l = m_l
        self.max_spd = max_spd
        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])
        self.readouts = torch.nn.ModuleList()
        self.readouts.append(NonLinearDipoleReadoutBlock(hidden_irreps,hidden_irreps,gate))


        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            self.readouts.append(NonLinearDipoleReadoutBlock(o3.Irreps(hidden_irreps_out),o3.Irreps(hidden_irreps_out),gate))
        # ##Set up GNN layers
        # self.distance_gaussian = GaussianSmearing(0, 1, 16, 0.2)
        # self.pre_lin_list = torch.nn.Sequential(torch.nn.Linear(40, 40), torch.nn.PReLU())
        # self.conv_list = torch.nn.ModuleList()
        # self.bn_list = torch.nn.ModuleList()
        # for i in range(conv_num):
        #     conv = GC_block(320, 10, aggr="mean")
        #     self.conv_list.append(conv)
            # bn = torch.nn.BatchNorm1d(285, track_running_stats=True, affine=True)
            # self.bn_list.append(bn)
        input_size = convert_irreps_to_dict(str(hidden_irreps)) * (num_interactions - 1) + hidden_irreps.count(
            o3.Irrep(0, 1))

        if required_variable=='dos':
            self.dos_mlp = torch.nn.Sequential(torch.nn.Linear(input_size, 500, bias=True),
                                               torch.nn.PReLU(),
                                               torch.nn.Linear(500, self.windows, bias=True),
                                               torch.nn.Tanh(),
                                               torch.nn.Dropout(p=dropout_rate),
                                               )
            self.scaling_mlp = torch.nn.Sequential(torch.nn.Linear(input_size, 100,bias=True),
                                                   torch.nn.PReLU(),
                                                   torch.nn.Linear(100, 1,bias=True),
                                                   torch.nn.ReLU(),
                                                   )

        #用于dos分离
        elif required_variable=='dos_split':
            self.dos_split = dos_split(input_dim=input_size,output_dim=windows,max_spd=self.max_spd)


        #用于电荷密度学习
        elif required_variable=='density' or required_variable=='density_spin':
            self.den_mlp = torch.nn.Sequential(torch.nn.Linear(input_size,320, bias=True),
                                               torch.nn.Tanh(),
                                               torch.nn.Linear(320, self.m_l, bias=True),
                                               torch.nn.ReLU(),
                                               # torch.nn.Tanh(),
                                               # torch.nn.ReLU(),
                                                 # torch.nn.Dropout(p=dropout_rate),
                                                 )
            self.fea_mlp = torch.nn.Sequential(torch.nn.Linear(self.m_l * 2, self.m_l * 2, bias=False),
                                               torch.nn.ReLU(),
                                                 # torch.nn.Dropout(p=dropout_rate),
                                                 )
            self.fea_mlp2 = torch.nn.Sequential(torch.nn.Linear(self.m_l * 2, self.m_l * 2, bias=False),
                                                torch.nn.ReLU(),
                                                # torch.nn.Linear(32 * 2, 32 * 2, bias=False),
                                                # torch.nn.ReLU(),
                                                # torch.nn.Dropout(p=dropout_rate),
                                                 )

            self.fea_mlp_up = torch.nn.Sequential(
                                                  torch.nn.Linear(self.m_l * 2, self.m_l * 2, bias=False),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(self.m_l * 2, self.m_l * 2, bias=False),
                                                  torch.nn.ReLU(),
                                                      # torch.nn.Dropout(p=dropout_rate),
                                                      )
            # self.fea_mlp_up = torch.nn.Linear(32 * 2, 32 * 2, bias=False)
            self.fea_mlp_down = torch.nn.Sequential(
                                                    torch.nn.Linear(self.m_l * 2, self.m_l * 2, bias=False),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(self.m_l * 2, self.m_l * 2, bias=False),
                                                    torch.nn.ReLU(),
                                                        # torch.nn.Dropout(p=dropout_rate),
                                                        )



    def forward(
        self,
        data: Dict[str, torch.Tensor],
        required_variable: str,
        spin:bool=False,
        split:bool=False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["x"].requires_grad_(True)
        data["pos"].requires_grad_(True)
        num_graphs = len(data["ptr"])
        (
            data["pos"],
            data["shift"],
            displacement,
        ) = get_symmetric_displacement(
            positions=data['pos'],
            unit_shifts=data['unit_shift'],
            cell=data["abc"],
            edge_index=data["edge_index"],
            num_graphs=num_graphs,
            batch=data["batch"],
        )
        node_feats = self.node_embedding(data["x"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["pos"],
            edge_index=data["edge_index"],
            shifts=data["shift"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        node_feats_list = []

        for interaction, product, readout in zip(
            self.interactions, self.products,self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["x"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["x"],
            )
            node_feats = readout(node_feats)
            node_feats_list.append(node_feats)

        # Concatenate node features
        node_feats = torch.cat(node_feats_list, dim=-1)
        if required_variable=='density' or required_variable=='density_spin':
            # for i in range(0, len(self.conv_list)):
            #     node_feats = self.conv_list[i](node_feats,  data["edge_index"], edge_attrs)
            #     node_feats = self.bn_list[i](node_feats)
            num_atoms = data['num_a']
            num_batch = len(num_atoms)
            batch = torch.cat([torch.ones(int(num_atoms)) * i for i, num_atoms in enumerate(num_atoms)]).to(dtype=torch.int64, device=data["x"].device)

            # feature 代表基函数，node_feats 代表原子特征——参数c
            feature = self.fea_mlp(data['feature'])
            if spin:
                feature = self.fea_mlp2(feature)
                feature_up = 0.5 * self.fea_mlp_up(feature)
                feature_down = 0.5 * self.fea_mlp_down(feature)
            # print(self.fea_mlp_up.weight)
            cell_size = data['cell_fin_size'].unsqueeze(1)
            # cell_size = self.cell_linear(cell_size)

            num_feas = data['d_idx']
            fea_seg = torch.cat([torch.ones(int(num_feas)) * i for i, num_feas in enumerate(num_feas)])

            node_feats = self.den_mlp(node_feats)


            # if spin:
                # node_feats = 2 * self.den_mlp2(node_feats)
                # node_feats_up = self.den_mlp_up(node_feats)
                # node_feats_down = self.den_mlp_down(node_feats)

            node_feats = node_feats * data["val_x"].unsqueeze(1)

            fea_shape = feature.shape[1]
            for batch_i in range(num_batch):
                x_i = node_feats[torch.nonzero(batch == batch_i).squeeze()]
                #feature.shape[1]需要整个batch统一大小，要不然合并不到一起去，所以把x_i转换大小，让后续能够*
                x_i = torch.cat([x_i, torch.zeros(fea_shape - x_i.shape[0], x_i.shape[1]).to(x_i.device)], 0)
                x_i = x_i.repeat(1, 1, 2)
                # feature[batch_size,atom_num,107*2]
                if spin==False:
                    feature_i = feature[torch.nonzero(fea_seg == batch_i).squeeze()]
                    den = (cell_size[batch_i] * torch.sum(torch.sum(feature_i * x_i, dim=2),dim=1)).squeeze()
                else:
                    # x_i_up = node_feats_up[torch.nonzero(batch == batch_i).squeeze()]
                    # # feature.shape[1]需要整个batch统一大小，要不然合并不到一起去，所以把x_i转换大小，让后续能够*
                    # x_i_up = torch.cat([x_i_up, torch.zeros(fea_shape - x_i_up.shape[0], x_i_up.shape[1]).to(x_i_up.device)], 0)
                    # x_i_down = node_feats_down[torch.nonzero(batch == batch_i).squeeze()]
                    # # feature.shape[1]需要整个batch统一大小，要不然合并不到一起去，所以把x_i转换大小，让后续能够*
                    # x_i_down = torch.cat([x_i_down, torch.zeros(fea_shape - x_i_down.shape[0], x_i_down.shape[1]).to(x_i_down.device)], 0)
                    # x_i = torch.broadcast_to(x_i.repeat(1,2),(feature_i_up.shape[0], -1, -1)).to_sparse()
                    # x_i = x_i.unsqueeze(0).repeat(1,1,2).expand(feature_i_up.shape[0], -1, -1).to_sparse()
                    # x_i_up = x_i_up.repeat(1, 2)
                    # x_i_down = x_i_down.repeat(1, 2)

                    #后续不需要feature，释放内存
                    # torch.cuda.empty_cache()
                    feature_i_up = feature_up[torch.nonzero(fea_seg == batch_i).squeeze()]
                    # den_up = (cell_size[batch_i] * torch.sparse.sum(feature_i_up.to_sparse() * x_i,[1,2]).to_dense()).squeeze().unsqueeze(1)
                    den_up = (cell_size[batch_i] * torch.sum(torch.sum(feature_i_up * x_i, dim=2), dim=1)).squeeze().unsqueeze(1)
                    feature_i_down = feature_down[torch.nonzero(fea_seg == batch_i).squeeze()]
                    # den_down = (cell_size[batch_i] * torch.sparse.sum(feature_i_down.to_sparse() * x_i,[1,2]).to_dense()).squeeze().unsqueeze(1)
                    den_down = (cell_size[batch_i] * torch.sum(torch.sum(feature_i_down * x_i, dim=2), dim=1)).squeeze().unsqueeze(1)
                    den = torch.cat((den_up,den_down),1)

                if batch_i == 0:
                    target_ml = den
                else:
                    target_ml = torch.cat([target_ml, den], dim=0)
        elif required_variable=='dos' or required_variable=='dos_split':
            # print(data["abc"][:3])
            # print(data["x"][0])
            # print(data["edge_index"][:,0])
            # print(data["edge_vec"][0])
            # print(len(self.conv_list))
            # print(node_feats.shape)
            # node_feats = self.pre_lin_list(data["x"])
            # lengths = (data["edge_vec"] - 0) / (7.5002 - 0)
            # edge_veg = self.distance_gaussian(lengths).to(torch.float)
            # for i in range(0, len(self.conv_list)):
            #     print(node_feats.shape)
            #     print(edge_feats.shape)
            #     node_feats = self.conv_list[i](node_feats,  data["edge_index"], edge_feats)
                # node_feats = self.bn_list[i](node_feats)
            if split:
                dos_ml,scaling = self.dos_split(node_feats,data['spd'])
                target_ml = {'dos_ml': dos_ml,
                             'spd': data['spd'],
                             'scaling': scaling.squeeze()}
            else:
                # print(torch.sum(node_feats,1))
                # print(torch.sum(node_feats*self.node_embedding2(self.node_embedding1(data["x"])),1))
                # scaling = self.scaling_mlp(node_feats*self.node_embedding2(self.node_embedding1(data["x"])))
                scaling = self.scaling_mlp(node_feats)
                # 保证输出dos_ml在0-1之间
                dos_ml = (self.dos_mlp(node_feats)+1)/2

                # # 找到每一行的最大值
                # max_values = torch.max(dos_ml, dim=1, keepdim=True).values
                # dos_ml = dos_ml / max_values
                # dos_ml = torch.nn.Sigmoid(dos_ml)
                target_ml = {'dos_ml': dos_ml,
                             'scaling': scaling.squeeze()}
        return target_ml