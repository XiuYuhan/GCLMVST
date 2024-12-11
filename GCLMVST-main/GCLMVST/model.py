import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  BatchNorm
from torch_geometric.utils import shuffle_node, mask_feature
from torch_geometric.typing import OptTensor

from GCLMVST.GAT_con2 import CustomGATConv2


class Model(nn.Module):
    def __init__(self, in_dim, arg):
        super().__init__()
        self.arg = arg
        self.encoder = CustomGATConv2(in_dim, arg.latent_dim, flow=arg.flow)
        self.decoder = CustomGATConv2(arg.latent_dim, in_dim, flow=arg.flow)


        self.norm = BatchNorm(arg.latent_dim)
        self.act = nn.ELU()
        self.act1 = nn.ReLU()
        self.info_nce = InfoNCE()


    def forward(self, x, g_s, g_f: OptTensor = None, g_h: OptTensor = None, w_h: OptTensor = None, w_h_a: OptTensor = None):


        x_n = shuffle_node(x)[0]
        x_p = mask_feature(x, p=self.arg.drop_feat_p)[0]


        if self.arg.mode_his == 'his':
            hi = self.encoder(x, g_h, edge_weight=w_h)
            h = self.decoder(hi, g_h, edge_weight=w_h)
        else:
            if self.arg.cut_corr == None:
                hi = self.encoder(x, g_s, edge_weight=w_h)
                h = self.decoder(hi, g_s, edge_weight=w_h)
            else:
                hi = self.encoder(x, g_s, edge_weight=w_h, edge_attr=w_h_a)
                h = self.decoder(hi, g_s, edge_weight=w_h, edge_attr=w_h_a)

        h0 = self.act(self.norm(hi))

        h1 = self.encoder(x_p, g_f, edge_weight=w_h_a)
        h1 = self.act(self.norm(h1))

        h2 = self.encoder(x_n, g_s, edge_weight=w_h)
        h2 = self.act(self.norm(h2))

        loss = self.compute_loss(x, h, h0, h1, h2)

        return hi, h, loss

    def compute_loss(self, x, y, p, p1, p2):
        loss_rec = F.mse_loss(x, y)

        loss_ctr = self.info_nce(p, p1, p2, temperature=self.arg.temp)

        return loss_rec + 0.2 * loss_ctr

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(max=20)

        return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def KL_Loss(self, mu, logvar):
        logvar = logvar.clamp(max=20)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        return KLD
class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

    计算 InfoNCE 损失，用于自监督学习中的对比任务。
    此损失函数确保相似（正）样本的嵌入向量靠近，不同（负）样本的嵌入向量远离。
    查询嵌入与一个正样本键和一个或多个负样本键进行比较。
    """

    def __init__(self, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys, temperature):
        return info_nce(query, positive_key, negative_keys,
                        temperature=temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=1., reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        # print(logits)
        # mu_p = torch.mean(positive_logit)
        # var_p = torch.var(positive_logit)
        # # print(mu_p, var_p)
        # mu_all = torch.mean(logits)
        # var_all = torch.var(logits)
        # # print(mu_all, var_all)
        # print((mu_p-mu_all)/mu_p)
        # temperature = (var_p.pow(2)-var_all.pow(2))/(-(mu_p-mu_all)+torch.sqrt((mu_p-mu_all).pow(2)+2*(var_p.pow(2)-var_all.pow(2))*np.log(len(positive_logit)*(len(positive_logit)+1)/(2*len(positive_logit)))))
        # print(temperature)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.utils import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch
from torch import Tensor
from typing import Optional, Tuple

def permute_node(x: Tensor, batch: Optional[Tensor] = None, training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Randomly permute the feature matrix :obj:`x` along the first dimension.

    The method returns (1) the permuted :obj:`x`, (2) the permutation
    indicating the orders of original nodes after permutation.

    Args:
        x (FloatTensor): The feature matrix.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`FloatTensor`, :class:`LongTensor`)

    Example:
        >>> x = torch.tensor([[0, 1, 2],
        ...                   [3, 4, 5],
        ...                   [6, 7, 8],
        ...                   [9, 10, 11]], dtype=torch.float)
        >>> x, node_perm = permute_node(x)
        >>> x
        tensor([[ 6.,  7.,  8.],
                [ 3.,  4.,  5.],
                [ 0.,  1.,  2.],
                [ 9., 10., 11.]])
        >>> node_perm
        tensor([2, 1, 0, 3])

        >>> # For batched graphs as inputs
        >>> batch = torch.tensor([0, 0, 1, 1])
        >>> x, node_perm = permute_node(x, batch)
        >>> x
        tensor([[ 6.,  7.,  8.],
                [ 0.,  1.,  2.],
                [ 3.,  4.,  5.],
                [ 9., 10., 11.]])
        >>> node_perm
        tensor([2, 0, 1, 3])
    """
    if not training:
        perm = torch.arange(x.size(0), device=x.device)
        return x, perm

    if batch is None:
        perm = torch.randperm(x.size(0), device=x.device)
        return x[perm], perm

    num_nodes = torch.bincount(batch)  # 计算每个批次中的节点数量
    cumsum = torch.cat([torch.zeros(1, device=x.device), num_nodes.cumsum(dim=0)])  # 累加求和
    perm = torch.cat([
        torch.randperm(n, device=x.device) + offset
        for offset, n in zip(cumsum[:-1], num_nodes)
    ])
    return x[perm], perm
