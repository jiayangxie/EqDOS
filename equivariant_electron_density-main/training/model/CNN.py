#####################################################
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from typing import Union,Tuple
import torch

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

class split(torch.nn.Module):
    def __init__(self,input_dim):
        super(split, self).__init__()
        self.input_dim = input_dim
        self.split_scaling = torch.nn.Sequential(torch.nn.Linear(32, 32),
                                               torch.nn.PReLU(),
                                               )
        self.split_dos = torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.input_dim*2),
                                             torch.nn.PReLU(),
                                             torch.nn.Dropout(p=0.001),
                                             torch.nn.Linear(self.input_dim*2, self.input_dim*32),
                                             torch.nn.PReLU(),
                                             torch.nn.Dropout(p=0.001)
                                               )
        # self.split_dos = dos_spd(self.input_dim)
    def forward(self, target_ml):
        target_ml['dos_ml'] = self.split_dos(target_ml['dos_ml']).view(-1,self.input_dim,32)
        target_ml['scaling'] = target_ml['scaling'] * target_ml['spd']
        return target_ml


class CNN(torch.nn.Module):
    def __init__(
            self,
            data,
            windows: int,
            required_variable: str,
            dim1=128,
            dim2=64,
            pre_fc_count=1,
            gc_count=3,
            batch_norm="True",
            batch_track_stats="True",
            dropout_rate=0.0,
            **kwargs
    ):
        super(CNN, self).__init__()
        if batch_track_stats == "False":
            self.batch_track_stats = False
        else:
            self.batch_track_stats = True
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            self.gc_dim = data.num_features
        else:
            self.gc_dim = dim1
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1
        ##Determine output dimension length
        if data[0]["dos"].ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0]["dos"][0])

        ##Set up pre-GNN dense layers
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = torch.nn.Sequential(torch.nn.Linear(dim1, dim1), torch.nn.PReLU())
                    self.pre_lin_list.append(lin)
                else:
                    lin = torch.nn.Sequential(torch.nn.Linear(dim1, dim1), torch.nn.PReLU())
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        ##Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()

        for i in range(gc_count):
            conv = GC_block(self.gc_dim, 50, aggr="mean")
            # conv = CGConv(self.gc_dim, data.num_edge_features, aggr="mean", batch_norm=False)
            self.conv_list.append(conv)
            if self.batch_norm == "True":
                bn = torch.nn.BatchNorm1d(self.gc_dim, track_running_stats=self.batch_track_stats, affine=True)
                self.bn_list.append(bn)

        self.dos_mlp = torch.nn.Sequential(torch.nn.Linear(post_fc_dim, dim2),
                                  torch.nn.PReLU(),
                                  torch.nn.Linear(dim2, output_dim),
                                  torch.nn.PReLU(),
                                  torch.nn.Dropout(p=dropout_rate),
                                  )
        if required_variable=='dos':

            self.scaling_mlp = torch.nn.Sequential(torch.nn.Linear(post_fc_dim, 100),
                                                   torch.nn.PReLU(),
                                                   torch.nn.Linear(100, 1),
                                                   )
        #用于dos分离
        elif required_variable=='dos_split':
            self.scaling_mlp_spd = torch.nn.Sequential(torch.nn.Linear(post_fc_dim, 100),
                                                       torch.nn.PReLU(),
                                                       torch.nn.Linear(100, 32),
                                                       )
            self.split = split(input_dim=windows)
    def forward(self, data,required_variable,split=False):
        # print(data["abc"][:3])
        # print(data["x"][0])·
        # print(data["edge_index"][:,0])
        # print(data["edge_vec"][0])
        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data["x"])
            else:
                out = self.pre_lin_list[i](out)
        ##GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    out = self.conv_list[i](data["x"],  data["edge_index"], data["edge_vec"])
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](data["x"],  data["edge_index"], data["edge_vec"])
            else:
                if self.batch_norm == "True":
                    # print(data["x"].shape)
                    # print(data["edge_vec"].shape)
                    out = self.conv_list[i](out,  data["edge_index"], data["edge_vec"])
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](out,  data["edge_index"], data["edge_vec"])

        # out = torch.nn.functional.dropout(out, p=self.dropout_rate, training=self.training)
        if required_variable=='dos':
            ##Post-GNN dense layers
            dos_out = self.dos_mlp(out)
            scaling = self.scaling_mlp(out)
            target_ml = {'dos_ml': dos_out,
                         'scaling': scaling.squeeze()}
        elif required_variable=='dos_split':
            ##Post-GNN dense layers
            dos_out = self.dos_mlp(out)
            scaling = self.scaling_mlp_spd(out)
            target_ml = {'dos_ml': dos_out,
                         'spd': data['spd'],
                         'scaling': scaling.squeeze()}
            target_ml = self.split(target_ml)
        return target_ml