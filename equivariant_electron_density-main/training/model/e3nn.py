"""model with self-interactions and gates

Exact equivariance to :math:`E(3)`

version of january 2021
"""
from typing import Tuple
import torch
from torch_scatter import scatter
import re

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import Convolution,tp_path_exists,Compose,smooth_cutoff
from e3nn.util.jit import compile_mode
###########################################################################################
# Neighborhood construction
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

class split(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(split, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = torch.nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = torch.nn.Conv2d(ch_in, ch_out, kernel_size=1)
        self.relu = torch.nn.ReLU()
        self.split_dos = torch.nn.Sequential(torch.nn.Linear(1, 3),
                                               torch.nn.PReLU(),
                                               )


        self.split_scaling = torch.nn.Sequential(torch.nn.Linear(1, 3),
                                               torch.nn.PReLU(),
                                               )

    def forward(self, target_ml):
        dos_out = self.split_dos(target_ml['dos_ml'].unsqueeze(2)).permute(0,2,1)
        # dos_out = self.depth_conv(dos_out.unsqueeze(3))
        # dos_out = self.point_conv(dos_out)
        # dos_out = self.relu(dos_out)

        scaling = self.split_scaling(target_ml['scaling'].unsqueeze(1))
        target_ml = {'dos_ml': dos_out.squeeze().permute(0,2,1),
                     'scaling': scaling.squeeze()}
        return target_ml

def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths

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

@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
        # from e3nn.nn import Activation
        # self.activation = Activation(irreps_out, acts=[torch.relu])  # 使用等变激活函数
        # self.linear2 = o3.Linear(irreps_in=irreps_out, irreps_out=irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        # return self.linear2(self.activation(self.linear(node_attrs)))
        return self.linear(node_attrs)


class e3nn(torch.nn.Module):
    r"""equivariant neural network

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features

    irreps_hidden : `e3nn.o3.Irreps`
        representation of the hidden features

    irreps_out : `e3nn.o3.Irreps`
        representation of the output features

    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials

    layers : int
        number of gates (non linearities)

    max_radius : float
        maximum radius for the convolution

    number_of_basis : int
        number of basis on which the edge length are projected

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes at a distance ``max_radius``

    num_nodes : float
        typical number of nodes in a graph
    """
    def __init__(
        self,
        in_num,
        out_num,
        irreps_in,
        irreps_hidden,
        irreps_out,
        # irreps_fea_out,
        irreps_node_attr,
        irreps_edge_attr,
        layers,
        max_radius,
        number_of_basis,
        radial_layers,
        radial_neurons,
        num_neighbors,
        num_nodes,
        # num_spd,
        required_variable: str,
        dropout_rate: float,
        windows: int,
        m_l: int,
        max_spd: int,
        reduce_output=False,
    ) -> None:
        super().__init__()

        self.pre_lin_list = torch.nn.ModuleList()
        pre_fc_count = 1
        for i in range(pre_fc_count):
            if i == 0:
                lin = torch.nn.Sequential(torch.nn.Linear(in_num, 64), torch.nn.PReLU())
                self.pre_lin_list.append(lin)
            else:
                lin = torch.nn.Sequential(torch.nn.Linear(64, 40), torch.nn.PReLU())
                self.pre_lin_list.append(lin)

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        # self.num_spd = num_spd
        self.reduce_output = reduce_output

        self.in_num = in_num
        self.out_num = out_num

        self.windows = windows
        self.m_l = m_l
        self.max_spd = max_spd

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.input_has_node_in = (irreps_in is not None)
        self.input_has_node_attr = (irreps_node_attr is not None)

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.cell_linear = torch.nn.Linear(1,1,bias=True)
        # self.irreps_fea_in = o3.Irreps(irreps_fea_out)
        # self.irreps_fea_out = o3.Irreps(irreps_fea_out)
        # self.fea_linear_1 = o3.Linear(self.irreps_fea_in,self.irreps_fea_out)
        self.fea_linear = torch.nn.Linear(self.out_num*2, self.out_num*2, bias=False)
        self.emmending = torch.nn.Embedding(self.in_num, self.out_num)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.dos_mlp = torch.nn.Sequential(torch.nn.Linear(self.out_num, 400),
                                  torch.nn.PReLU(),
                                  torch.nn.Linear(400, 400),
                                  torch.nn.PReLU(),
                                  )
        self.real_imag = torch.nn.Sequential(torch.nn.Linear(1,2),
                                  torch.nn.PReLU(),
                                  )



        for _ in range(layers):
            irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)])
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
            irreps = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors
            )
        )
        node_attr_irreps = o3.Irreps([(self.in_num, (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=self.irreps_in
        )
        input_size = convert_irreps_to_dict(str(self.irreps_out))
        if required_variable == 'dos':
            self.dos_mlp = torch.nn.Sequential(torch.nn.Linear(input_size, 500, bias=True),
                                               torch.nn.PReLU(),
                                               torch.nn.Linear(500, self.windows, bias=True),
                                               torch.nn.Tanh(),
                                               torch.nn.Dropout(p=dropout_rate),
                                               )
            self.scaling_mlp = torch.nn.Sequential(torch.nn.Linear(input_size, 100, bias=True),
                                                   torch.nn.PReLU(),
                                                   torch.nn.Linear(100, 1, bias=True),
                                                   torch.nn.ReLU(),
                                                   )

        # 用于dos分离
        elif required_variable == 'dos_split':
            self.dos_split = dos_split(input_dim=input_size, output_dim=windows, max_spd=self.max_spd)


        # 用于电荷密度学习
        elif required_variable == 'density' or required_variable == 'density_spin':
            self.den_mlp = torch.nn.Sequential(torch.nn.Linear(input_size, 320, bias=True),
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

            # self.emmending = torch.nn.Embedding(2, 107)
            # self.relu1 = torch.nn.ReLU()
            # self.relu2 = torch.nn.ReLU()
            # self.relu3 = torch.nn.ReLU()
            # self.layers = torch.nn.ModuleList()
    def forward(self, data:dict,required_variable:str,spin:bool) -> torch.Tensor:
        """evaluate the network

        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        R = data['pos']
        x = data['x']
        Z = data['z']
        # ##Pre-GNN dense layers
        # for i in range(0, len(self.pre_lin_list)):
        #     if i == 0:
        #         out = self.pre_lin_list[i](x)
        #     else:
        #         out = self.pre_lin_list[i](out)

        num_atoms = data['num_a']
        num_batch = len(num_atoms)
        batch = torch.cat([torch.ones(int(num_atoms)) * i for i, num_atoms in enumerate(num_atoms)]).to(dtype=torch.int64, device=R.device)
        if required_variable=="density":
            edge_index = data['edge_index'].long()
        elif (required_variable == 'dos') | (required_variable == 'dos_split'):
            edge_index = data['edge_index'].transpose(0,1).long()
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["pos"],
            edge_index=data["edge_index"],
            shifts=data["shift"],
        )
        lengths = vectors.norm(dim = 1)
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, vectors, True, normalization='component')
        edge_length_embedded = soft_one_hot_linspace(
            x=lengths,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='gaussian',
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(lengths / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and x!=None:
            assert self.irreps_in is not None
            # Embeddings
            x = self.node_embedding(data["x"])
            # x = data['x']
        else:
            assert self.irreps_in is None
            x = R.new_ones((R.shape[0], 1))

        if self.input_has_node_attr and 'z' in data:
            z = Z
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = R.new_ones((R.shape[0], 1))
        for lay in self.layers:
            x = lay(x, x, edge_src, edge_dst, edge_attr, edge_length_embedded)

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

            node_feats = self.den_mlp(x)

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
            #     node_feats = self.conv_list[i](node_feats,  data["edge_index"], edge_feats)
                # node_feats = self.bn_list[i](node_feats)
            if split:
                dos_ml,scaling = self.dos_split(x,data['spd'])
                target_ml = {'dos_ml': dos_ml,
                             'spd': data['spd'],
                             'scaling': scaling.squeeze()}
            else:
                # print(torch.sum(node_feats,1))
                # print(torch.sum(node_feats*self.node_embedding2(self.node_embedding1(data["x"])),1))
                # scaling = self.scaling_mlp(node_feats*self.node_embedding2(self.node_embedding1(data["x"])))
                scaling = self.scaling_mlp(x)
                # 保证输出dos_ml在0-1之间
                dos_ml = (self.dos_mlp(x)+1)/2

                # # 找到每一行的最大值
                # max_values = torch.max(dos_ml, dim=1, keepdim=True).values
                # dos_ml = dos_ml / max_values
                # dos_ml = torch.nn.Sigmoid(dos_ml)
                target_ml = {'dos_ml': dos_ml,
                             'scaling': scaling.squeeze()}
        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes**0.5)
        else:
            return target_ml


