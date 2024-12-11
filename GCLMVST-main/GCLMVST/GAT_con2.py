from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, is_torch_sparse_tensor
from torch_geometric.utils.sparse import set_sparse_value


class CustomGATConv2(MessagePassing):
    """
    Custom GAT convolution layer with optional edge attributes and improved self-loop weighting.

    Parameters:
        in_channels (int or Tuple[int, int]): Input channels.
        out_channels (int): Output channels.
        heads (int): Number of attention heads.
        concat (bool): Whether to concatenate outputs of each head.
        negative_slope (float): LeakyReLU angle of the negative slope.
        dropout (float): Dropout rate.
        add_self_loops (bool): If True, will add self-loops to the input graph.
        edge_dim (int, optional): Edge feature dimensionality.
        fill_value (float or Tensor or str): Fill value for added self-loops.
        improved (bool): If True, uses an improved self-loop weight.
        bias (bool): If True, adds a learnable bias.
    """

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            improved: bool = False,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.improved = improved

        # Initialize linear layers for source and destination nodes
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False, weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False, weight_initializer='glorot')

        # Attention parameters for source, destination, and optional edge attributes
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        # Bias term
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        if self.att_edge is not None:
            glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):

        # Precompute linear transformations
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:
            x_src, x_dst = x
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            edge_index, edge_attr = self.handle_self_loops(edge_index, edge_attr, size)

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, edge_weight=edge_weight)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        # Concatenate heads or take mean
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        # Return attention weights if specified
        if isinstance(return_attention_weights, bool):
            return self.handle_attention_weights(out, edge_index, alpha)
        else:
            return out

    def handle_self_loops(self, edge_index: Adj, edge_attr: OptTensor, size: Size):
        """Handles adding self-loops to the graph."""
        loop_value = 2.0 if self.improved else self.fill_value

        if isinstance(edge_index, Tensor):
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=loop_value,
                                                   num_nodes=size[0] if size else None)
        elif isinstance(edge_index, SparseTensor):
            if self.edge_dim is None:
                edge_index = set_sparse_value(edge_index)
            else:
                raise NotImplementedError("Self-loops with edge attributes for SparseTensor not supported.")

        return edge_index, edge_attr

    def handle_attention_weights(self, out, edge_index, alpha):
        """Handles the attention weights for output."""
        if isinstance(edge_index, Tensor):
            if is_torch_sparse_tensor(edge_index):
                adj = set_sparse_value(edge_index, alpha)
                return out, (adj, alpha)
            else:
                return out, (edge_index, alpha)
        elif isinstance(edge_index, SparseTensor):
            return out, edge_index.set_value(alpha, layout='coo')

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, edge_weight: OptTensor,
                    index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        # If edge attributes are present, incorporate them into attention calculation
        if edge_attr is not None and self.lin_edge is not None:
            edge_attr = edge_attr.view(-1, 1) if edge_attr.dim() == 1 else edge_attr
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha += (edge_attr * self.att_edge).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        if edge_weight is not None:
            alpha *= edge_weight.view(-1, 1)

        # Apply softmax normalization
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads}, improved={self.improved})'
