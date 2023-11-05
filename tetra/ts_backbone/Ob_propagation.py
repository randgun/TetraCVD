
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset
import torch.nn as nn
import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, SiLU
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import gather_csr, scatter, segment_csr


class Forth_progation(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels: int,
                 n_nodes: int, ob_dim: int, d_pe: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        # in_channels = out_channels = max_len*self.d_ob
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.d_pe = d_pe
        self.ob_dim = ob_dim
        self.index = None
        self.n_steps = out_channels // ob_dim

        self.hidden_size = 384

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.n_nodes = n_nodes
        self.nodewise_weights = Parameter(torch.Tensor(self.n_nodes, self.hidden_size))

        self.increase_dim = Linear(in_channels[1],  (heads * 16 + d_pe)*self.n_steps)
        self.linear_1 = Linear(out_channels,  self.hidden_size)
        self.linear_2 = Linear(self.hidden_size, out_channels)
        self.relu = nn.ReLU()
        #self.increase_dim = Linear(in_channels[1],  heads * 16 + d_pe)
        self.map_weights = Parameter(torch.Tensor(self.n_nodes, heads * 16))
        self.lin_value = Linear(in_channels[0], heads * out_channels)

        self.increase_dim.reset_parameters()
        self.lin_value.reset_parameters()
        self.linear_1.reset_parameters()
        self.linear_2.reset_parameters()
        glorot(self.nodewise_weights)
        glorot(self.map_weights)

        

    def forward(self, x: Union[Tensor, PairTensor], p_t: Tensor, edge_index: Adj, edge_weights=None, use_beta=True,
                edge_attr: OptTensor = None, return_attention_weights=None):

        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        """Here, the edge_attr is not edge weights, but edge features!
        If we want to the calculation contains edge weights, change the calculation of alpha"""

        self.edge_index = edge_index
        self.p_t = p_t
        self.use_beta = use_beta

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weights=edge_weights, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None
        edge_index = self.edge_index

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        use_beta = self.use_beta
        if use_beta == True:
            n_step = self.p_t.shape[0]
            n_edges = x_i.shape[0]
            # print("x_i: {}".format(x_i.shape))
            # max_len*self.d_ob
            # h_W: [n_edges, heads * out_channels / 4, 32]
            h_W = self.increase_dim(x_i)
            h_W = h_W.view(-1, n_step, self.heads * 16 + self.d_pe)
            # self.map_weights: [self.n_nodes, heads * 16]
            # w_v: [n_edges, 1, heads * 16]
            w_v = self.map_weights[self.edge_index[1]].unsqueeze(1)

            p_emb = self.p_t.unsqueeze(0)
            
            # aa: [n_edges, n_steps, head * 16 + d_pe(16)]
            aa = torch.cat([w_v.repeat(1, n_step, 1,), p_emb.repeat(n_edges, 1, 1)], dim=-1)
            # beta: [n_edges, n_steps]
            beta = torch.mean(h_W * aa, dim=-1)

        if edge_weights is not None:
            if use_beta == True:
                gamma = beta*(edge_weights.unsqueeze(-1))
                gamma = torch.repeat_interleave(gamma, self.ob_dim, dim=-1)

                # edge prune, prune out half of edges
                all_edge_weights = torch.mean(gamma, dim=1)
                # TODO
                K = int(gamma.shape[0] * 0.5)
                index_top_edges = torch.argsort(all_edge_weights, descending=True)[:K]
                # gamma: [top_edges, self.ob_dim * n_steps]
                gamma = gamma[index_top_edges]
                self.edge_index = self.edge_index[:, index_top_edges]
                index = self.edge_index[0]
                x_i = x_i[index_top_edges]
            else:
                gamma = edge_weights.unsqueeze(-1)

        self.index = index
        if use_beta == True:
            self._alpha = torch.mean(gamma, dim=-1)
        else:
            self._alpha = gamma
        
        gamma = softmax(gamma, index, ptr, size_i)
        gamma = F.dropout(gamma, p=self.dropout, training=self.training)

        decompose = True
        if decompose == False:
            out = F.relu(self.lin_value(x_i)).view(-1, self.heads, self.out_channels)
        else:
            source_nodes = self.edge_index[0]
            target_nodes = self.edge_index[1]
            # [self.n_nodes, heads * out_channels]
            w1 = self.nodewise_weights[source_nodes].unsqueeze(-1)
            w2 = self.nodewise_weights[target_nodes].unsqueeze(1)
            x_i = x_i.view(-1, self.heads, self.out_channels)
            hh = self.linear_1(x_i)
            # 论文公式2
            out = torch.bmm(hh, torch.bmm(w1, w2))
            
            out = self.linear_2(out)
            # out = self.relu(out)

        if use_beta == True:
            out = out * gamma.view(-1, self.heads, out.shape[-1])
        else:
            out = out * gamma.view(-1, self.heads, 1)
        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        # index有待商榷
        index = self.index
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class New_progation(MessagePassing):
    _alpha: OptTensor

    def __init__(self, 
                 n_nodes: int, 
                 d_model: int,
                 heads: int = 1, 
                 concat: bool = True,
                 dropout: float = 0.1, 
                 edge_dim: Optional[int] = None,
                 d_pe: int = 16, 
                 **kwargs
                 ):
        
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.index = None
        self.d_model = d_model
        self.d_pe = d_pe
        self.n_nodes = n_nodes

        # self.nodewise_weights = Parameter(torch.Tensor(self.n_nodes, d_model))
        self.nodewise_weights = torch.rand((self.n_nodes, d_model)).cuda()
        self.activate = SiLU()
        # self.map_weights = Parameter(torch.Tensor(self.n_nodes, self.d_model-self.d_pe))
        self.map_weights = torch.rand((self.n_nodes, self.d_model-self.d_pe)).cuda()
        self.sigmoid = nn.Sigmoid()


    def forward(self, 
                x: Union[Tensor, PairTensor], 
                p_t: Tensor, edge_index: Adj, 
                edge_weights=None, 
                use_beta=True,
                edge_attr: OptTensor = None, 
                return_attention_weights=None
                ):

        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        """Here, the edge_attr is not edge weights, but edge features!
        If we want to the calculation contains edge weights, change the calculation of alpha"""

        self.edge_index = edge_index
        self.p_t = p_t
        self.use_beta = use_beta

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weights=edge_weights, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None
        edge_index = self.edge_index

        # if self.concat:
        #     out = out.view(-1, self.heads * self.out_channels)
        # else:
        #     out = out.mean(dim=1)
        # print(out.shape)
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        if torch.sum(torch.isnan(x_i)==True) != 0:
            print('----x_i_1-----')
            raise
        use_beta = self.use_beta
        n_step = self.p_t.shape[0]
        if use_beta == True:
            n_edges = x_i.shape[0]
            # print("x_i: {}".format(x_i.shape))
            # w_v: [n_edges, 1, d_model-d_pe]
            w_v = self.map_weights[self.edge_index[1]].unsqueeze(1)
            p_emb = self.p_t.unsqueeze(0)
            
            # aa: [n_edges, n_steps, d_model]
            aa = torch.cat([w_v.repeat(1, n_step, 1,), p_emb.repeat(n_edges, 1, 1)], dim=-1)
            # beta: [n_edges, n_steps] 论文中的alpha
            beta = torch.mean(x_i * aa, dim=-1)
            beta = self.sigmoid(beta)

        if edge_weights is not None:
            if use_beta == True:
                # gamma: [n_edges, n_steps]
                gamma = beta*(edge_weights.unsqueeze(-1))

                # edge prune, prune out half of edges
                # [n_edges, 1]
                all_edge_weights = torch.mean(gamma, dim=1)
                K = int(gamma.shape[0] * 0.5)
                index_top_edges = torch.argsort(all_edge_weights, descending=True)[:K]
                # gamma: [K, n_steps*self.ob_dim]
                gamma = gamma[index_top_edges]
                self.edge_index = self.edge_index[:, index_top_edges]
                index = self.edge_index[0]
                x_i = x_i[index_top_edges]
            else:
                gamma = edge_weights.unsqueeze(-1)
            

        self.index = index
        if use_beta == True:
            self._alpha = torch.mean(gamma, dim=-1)
        else:
            self._alpha = gamma
        
        
        gamma = softmax(gamma, index, ptr, size_i)
        gamma = F.dropout(gamma, p=self.dropout, training=self.training)
        # print('--------------------------------------------------')
        # print(gamma.shape)

        
        source_nodes = self.edge_index[0]
        target_nodes = self.edge_index[1]
        # [self.n_edges, heads * out_channels]
        w1 = self.nodewise_weights[source_nodes].unsqueeze(-1)
        w2 = self.nodewise_weights[target_nodes].unsqueeze(1)
        
        # print(w1.shape)
        # print(w2.shape)
        # 论文公式2
        W = torch.bmm(w1, w2)
        if torch.sum(torch.isnan(x_i)==True) != 0:
            print('----x_i_2-----')
            raise
        if torch.sum(torch.isnan(W)==True) != 0:
            print('----W-----')
            raise
        # print(x_i.shape, W.shape)
        # print(n_step, self.d_model)
        out = torch.bmm(x_i.view(-1, n_step, self.d_model), W)
        if torch.sum(torch.isnan(out)==True) != 0:
            print('----out-----')
            raise

        
        out = out * gamma.unsqueeze(-1)
        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        # index有待商榷
        index = self.index
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class Observation_progation(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels: int,
                 n_nodes: int, ob_dim: int,
                 heads: int = 1, concat: bool = True, beta: bool = False,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.weight = Parameter(torch.Tensor(in_channels[1], heads * out_channels))
        self.bias = Parameter(torch.Tensor(heads * out_channels))

        self.n_nodes = n_nodes
        self.nodewise_weights = Parameter(torch.Tensor(self.n_nodes, heads * out_channels))

        self.increase_dim = Linear(in_channels[1],  heads * out_channels*8)
        self.map_weights = Parameter(torch.Tensor(self.n_nodes, heads * 16))

        self.ob_dim = ob_dim
        self.index = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()
        glorot(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        glorot(self.nodewise_weights)
        glorot(self.map_weights)
        self.increase_dim.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], p_t: Tensor, edge_index: Adj, edge_weights=None, use_beta=False,
                edge_attr: OptTensor = None, return_attention_weights=None):

        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        """Here, the edge_attr is not edge weights, but edge features!
        If we want to the calculation contains edge weights, change the calculation of alpha"""

        self.edge_index = edge_index
        self.p_t = p_t
        self.use_beta = use_beta

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weights=edge_weights, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None
        edge_index = self.edge_index

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message_selfattention(self, x_i: Tensor, x_j: Tensor,edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        if edge_weights is not None:
            alpha = edge_weights.unsqueeze(-1)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        out *= alpha.view(-1, self.heads, 1)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weights: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        use_beta = self.use_beta
        if use_beta == True:
            n_step = self.p_t.shape[0]
            n_edges = x_i.shape[0]

            h_W = self.increase_dim(x_i).view(-1, n_step, 32)
            w_v = self.map_weights[self.edge_index[1]].unsqueeze(1)

            p_emb = self.p_t.unsqueeze(0)

            aa = torch.cat([w_v.repeat(1, n_step, 1,), p_emb.repeat(n_edges, 1, 1)], dim=-1)
            beta = torch.mean(h_W * aa, dim=-1)

        if edge_weights is not None:
            if use_beta == True:
                gamma = beta*(edge_weights.unsqueeze(-1))
                gamma = torch.repeat_interleave(gamma, self.ob_dim, dim=-1)

                # edge prune, prune out half of edges
                all_edge_weights = torch.mean(gamma, dim=1)
                K = int(gamma.shape[0] * 0.5)
                index_top_edges = torch.argsort(all_edge_weights, descending=True)[:K]
                gamma = gamma[index_top_edges]
                self.edge_index = self.edge_index[:, index_top_edges]
                index = self.edge_index[0]
                x_i = x_i[index_top_edges]
            else:
                gamma = edge_weights.unsqueeze(-1)

        self.index = index
        if use_beta == True:
            self._alpha = torch.mean(gamma, dim=-1)
        else:
            self._alpha = gamma

        gamma = softmax(gamma, index, ptr, size_i)
        gamma = F.dropout(gamma, p=self.dropout, training=self.training)

        decompose = True
        if decompose == False:
            out = F.relu(self.lin_value(x_i)).view(-1, self.heads, self.out_channels)
        else:
            print("+++++++++++++++++++++++++++++")
            # 炸显存
            source_nodes = self.edge_index[0]
            target_nodes = self.edge_index[1]
            w1 = self.nodewise_weights[source_nodes].unsqueeze(-1)
            w2 = self.nodewise_weights[target_nodes].unsqueeze(1)
            out = torch.bmm(x_i.view(-1, self.heads, self.out_channels), torch.bmm(w1, w2))
        if use_beta == True:
            out = out * gamma.view(-1, self.heads, out.shape[-1])
        else:
            out = out * gamma.view(-1, self.heads, 1)
        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.
        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.
        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        index = self.index
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
