import torch
import torch.nn as nn 
from torch.nn.functional import normalize as l2
import math
import torch.nn.functional as F
import numpy as np


class DotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys):

        return torch.bmm(queries, keys.transpose(1, 2))

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys):

        return torch.bmm(queries, keys.transpose(1, 2)) / (queries.size(2)**0.5)

class GeneralDotProduct(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        torch.nn.init.orthogonal_(self.W)

    def forward(self, queries, keys):

        return torch.bmm(queries @ self.W, keys.transpose(1,2))

class ConcatDotProduct(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        raise 

    def forward(self, queries, keys):

        return

class Additive(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.U = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.T = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.W = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim,1))
        torch.nn.init.orthogonal_(self.U)
        torch.nn.init.orthogonal_(self.T)

    def forward(self, queries, keys):

        return self.W(queries.unsqueeze(1)@self.U + keys.unsqueeze(2)@self.T + self.b).squeeze(-1).transpose(1,2)


class Attention(nn.Module):
    def __init__(self, similarity, hidden_dim=1024):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)
        self.attention_maps = []

        assert similarity in ['dot', 'scaled_dot', 'general_dot', 'concat_dot', 'additive']
        if similarity == 'dot':
            self.similarity = DotProduct()
        elif similarity == 'scaled_dot':
            self.similarity = ScaledDotProduct()
        elif similarity == 'general_dot':
            self.similarity = GeneralDotProduct(hidden_dim)
        elif similarity == 'concat_dot':
            self.similarity = ConcatDotProduct(hidden_dim)
        elif similarity == 'additive':
            self.similarity = Additive(hidden_dim)
        else:
            raise

    def forward(self, queries, keys, qmasks=None, kmasks=None):
        if torch.is_tensor(qmasks) and not torch.is_tensor(kmasks):
            dim0, dim1 = qmasks.size(0), keys.size(1)
            kmasks = torch.ones(dim0,dim1).cuda()

        elif not torch.is_tensor(qmasks) and torch.is_tensor(kmasks):
            dim0, dim1 = kmasks.size(0), queries.size(1)
            qmasks = torch.ones(dim0,dim1).cuda()
        else:
            pass

        attention = self.similarity(queries, keys)
        if torch.is_tensor(qmasks) and torch.is_tensor(kmasks):
            qmasks = qmasks.repeat(queries.size(0)//qmasks.size(0),1).unsqueeze(2)
            kmasks = kmasks.repeat(keys.size(0)//kmasks.size(0),1).unsqueeze(2)
            attnmasks = torch.bmm(qmasks, kmasks.transpose(1, 2))
            attention = torch.clip(attention, min=-10, max=10)
            attention = attention.exp()
            attention = attention * attnmasks
            attention = attention / (attention.sum(2).unsqueeze(2) + 1e-5)
        else:
            attention = self.softmax(attention)

        return attention

@torch.no_grad()
def save_attention_maps(self, input, output):
    
    self.attention_maps.append(output.data.detach().cpu().numpy())

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, similarity='dot', same_linear=False, analysis=False):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"{hidden_dim} dimension, {num_heads} heads"
        self.num_heads   = num_heads
        partition_size   = hidden_dim // num_heads
        
        self.same_linear = same_linear
        if self.same_linear:
            self.project_shared  = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.project_queries = nn.Linear(hidden_dim, hidden_dim)
            self.project_keys    = nn.Linear(hidden_dim, hidden_dim)
            self.project_values  = nn.Linear(hidden_dim, hidden_dim)
        
        self.concatenation  = nn.Linear(hidden_dim, hidden_dim)
        self.attention      = Attention(similarity, partition_size)

        if analysis:
            self.attention.register_forward_hook(save_attention_maps)

    def forward(self, queries, keys, values, qmasks=None, kmasks=None):
        h = self.num_heads
        b, n, d = queries.size()
        _, m, _ = keys.size()
        p = d // h

        if self.same_linear:
            queries = self.project_shared(queries)
            keys    = self.project_shared(keys)
            values  = self.project_shared(values)
        else:
            queries = self.project_queries(queries)  # shape [b, n, d]
            keys    = self.project_keys(keys)  # shape [b, m, d]
            values  = self.project_values(values)  # shape [b, m, d]

        queries = queries.view(b, n, h, p)
        keys = keys.view(b, m, h, p)
        values = values.view(b, m, h, p)

        queries = queries.permute(2, 0, 1, 3).contiguous().view(h * b, n, p)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)
        values = values.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)

        attn_w = self.attention(queries, keys, qmasks, kmasks)  # shape [h * b, n, p]
        output = torch.bmm(attn_w, values)
        output = output.view(h, b, n, p)
        output = output.permute(1, 2, 0, 3).contiguous().view(b, n, d)
        output = self.concatenation(output)  # shape [b, n, d]

        return output, attn_w

class EmptyModule(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, x):
        return 0.

class RFF(nn.Module):
    def __init__(self, h, d=0.1):
        super().__init__()
        self.rff = nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())

    def forward(self, x):

        return self.rff(x)

class NullNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):

        return input

class SetNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.eps           = eps
        self.weights_gamma = nn.Parameter(torch.ones(1,1,hidden_dim))
        self.bias_beta     = nn.Parameter(torch.zeros(1,1,hidden_dim))
        if elementwise_affine:
            self.weights_gamma.requires_grad = True
            self.bias_beta.requires_grad     = True

    def forward(self, input, masks):
        # input : a (B, N, D)-sized Tensor
        # masks : a (B, N)-sized Tensor
        if torch.is_tensor(masks):
            dim   = input.size(2)
            mu    = (input * masks.unsqueeze(2)).sum(2).sum(1) / (masks.sum(1) * dim) # (B)-sized Tensor
            sigma = (input - mu.view(-1,1,1)) * masks.unsqueeze(2)
            sigma = sigma**2
            sigma = sigma.sum(2).sum(1) / (masks.sum(1) * dim)                        # (B)-sized Tensor
        else:
            dim   = input.size(2)
            mu    = input.sum(2).sum(1) / (input.size(1) * dim) # (B)-sized Tensor
            sigma = (input - mu.view(-1,1,1)) ** 2
            sigma = sigma.sum(2).sum(1) / (input.size(1) * dim)                       # (B)-sized Tensor
        mu, sigma  = mu.view(-1,1,1), sigma.view(-1,1,1)                              # (B, 1, 1)-sized Tensors
        normalized = (input - mu) / (sigma + self.eps)                                # (B, N, D)-sized Tensor

        return normalized * self.weights_gamma + self.bias_beta


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_heads, 
                 rff_module, 
                 similarity='dot', 
                 same_linear=False,
                 norm_method=None,
                 elementwise_affine=True,
                 clean_path=False,
                 analysis=False):
        super().__init__()
        # 1. Initialize the Multihead Attention Block
        self.multihead = MultiheadAttention(hidden_dim, num_heads, similarity, same_linear, analysis) 

        # 2. Set the Normalization Method
        self.norm_method = norm_method
        if self.norm_method == 'layer_norm':
            self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=elementwise_affine)
            self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=elementwise_affine)
        elif self.norm_method == 'set_norm':
            self.norm1 = SetNorm(hidden_dim, elementwise_affine=elementwise_affine)
            self.norm2 = SetNorm(hidden_dim, elementwise_affine=elementwise_affine)
        else:
            print("There is no specified normalization method in this MAB")
            self.norm1 = NullNorm()
            self.norm2 = NullNorm()

        # 3. Set the Clean-Path Equivariant Residual Connection Method
        self.clean_path = clean_path

        # 4. Load the Row-wise Feedforward Layer
        self.rff = rff_module

    def forward(self, X, Y, Xm=None, Ym=None):
        if not self.clean_path:
            if self.norm_method != 'set_norm':
                H, A = self.multihead(X, Y, Y, Xm, Ym)
                X    = X + H
                X    = self.norm1(X) 
                return self.norm2(X) + self.rff(X), A
            else:
                H, A = self.multihead(X, Y, Y, Xm, Ym)
                X    = X + H
                X    = self.norm1(X, Xm) 
                return self.norm2(X, Xm) + self.rff(X), A
        else:
            if self.norm_method != 'set_norm':
                H, A = self.multihead(self.norm1(X), self.norm1(X), self.norm1(Y), Xm, Ym)
                X    = X + H
                return X + self.rff(self.norm2(X)), A
            else:
                H, A = self.multihead(self.norm1(X, Xm), self.norm1(Y, Ym), self.norm1(Y, Ym), Xm, Ym)
                X    = X + H
                return X + self.rff(self.norm2(X, Xm)), A


class SetAttentionBlock(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_heads, 
                 rff_module, 
                 similarity='dot', 
                 same_linear=False,
                 norm_method=None,
                 elementwise_affine=True,
                 clean_path=False,
                 analysis=False):
        super().__init__()
        mab_args = (hidden_dim, num_heads, rff_module, similarity, same_linear, norm_method, elementwise_affine, clean_path, analysis)
        self.mab = MultiheadAttentionBlock(*mab_args)

    def forward(self, X, Xm=None):

        return self.mab(X, X, Xm, Xm)

class InducedSetAttentionBlock(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_heads, 
                 rff_module, 
                 similarity='dot', 
                 same_linear=False,
                 norm_method=None,
                 elementwise_affine=True,
                 clean_path=False,
                 analysis=False,
                 num_inducing_points=4):
        super().__init__()
        mab_args = (hidden_dim, num_heads, rff_module, similarity, same_linear, norm_method, elementwise_affine, clean_path, analysis)
        self.mab1 = MultiheadAttentionBlock(*mab_args)
        self.mab2 = MultiheadAttentionBlock(*mab_args)
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, hidden_dim))

    def forward(self, X, Xm=None):
        b = x.size(0)
        I = self.inducing_points
        I = I.repeat([b, 1, 1])  # shape [b, m, d]
        H = self.mab1(I, X, None, Xm)  # shape [b, m, d]

        return self.mab2(X, H, Xm, None)

class PoolingMultiheadAttention(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_heads, 
                 rff_module, 
                 similarity='dot', 
                 same_linear=False,
                 norm_method=None,
                 elementwise_affine=True,
                 clean_path=False,
                 analysis=False,
                 num_seed_vectors=2):
        super().__init__()
        mab_args = (hidden_dim, num_heads, rff_module, similarity, same_linear, norm_method, elementwise_affine, clean_path, analysis)
        self.mab = MultiheadAttentionBlock(*mab_args)
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seed_vectors, hidden_dim))
        # torch.nn.init.xavier_uniform_(self.seed_vectors)
        torch.nn.init.orthogonal_(self.seed_vectors)

    def forward(self, X, Xm=None):
        b = X.size(0)
        S = self.seed_vectors
        S = S.repeat([b, 1, 1])  # random seed vector: shape [b, k, d]

        return self.mab(S, X, None, Xm)

class PoolingMultiheadCrossAttention(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_heads, 
                 rff_module, 
                 similarity='dot', 
                 same_linear=False,
                 norm_method=None,
                 elementwise_affine=True,
                 clean_path=False,
                 analysis=False):
        super().__init__()
        mab_args = (hidden_dim, num_heads, rff_module, similarity, same_linear, norm_method, elementwise_affine, clean_path, analysis)
        self.mab = MultiheadAttentionBlock(*mab_args)
         
    def forward(self, X, Y, Xm=None, Ym=None):
        
        return self.mab(X, Y, Xm, Ym)

class QueryProposal(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_heads, 
                 similarity='dot', 
                 same_linear=False,
                 analysis=False):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"{hidden_dim} dimension, {num_heads} heads"
        self.num_heads = num_heads
        partition_size   = hidden_dim // num_heads
        
        self.same_linear = same_linear
        if self.same_linear:
            self.project_shared  = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.project_queries = nn.Linear(hidden_dim, hidden_dim)
            self.project_keys    = nn.Linear(hidden_dim, hidden_dim)    
        self.attention      = Attention(similarity, partition_size)

        if analysis:
            self.attention.register_forward_hook(save_attention_maps)

    def forward(self, queries, keys, qmasks=None, kmasks=None):
        h = self.num_heads
        b, n, d = queries.size()
        _, m, _ = keys.size()
        p = d // h

        if self.same_linear:
            queries = self.project_shared(queries)
            keys    = self.project_shared(keys)
        else:
            queries = self.project_queries(queries)  # shape [b, n, d]
            keys    = self.project_keys(keys)  # shape [b, m, d]
            
        queries = queries.view(b, n, h, p)
        keys = keys.view(b, m, h, p)
        
        queries = queries.permute(2, 0, 1, 3).contiguous().view(h * b, n, p)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)
        
        attn_w  = self.attention(queries, keys, qmasks, kmasks)  # shape [h * b, n, p]
        x, y, z = attn_w.size()

        return attn_w.view(x//b,b,y,z).mean(0)[:,:,:-1].sum(2).unsqueeze(2)