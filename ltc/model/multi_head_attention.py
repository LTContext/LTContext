# some parts of this code inspired by the multi_head_dispatch class in xformers
from functools import partial
from typing import Optional
from dataclasses import dataclass


import torch
from torch import nn
from torch.nn.init import constant_
from einops import repeat, rearrange

from ltc.model.attention_utils import ScaledDotProduct

import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


@dataclass
class AttentionParams:
    num_heads: int
    type: str = 'full'
    bias: bool = True
    dropout: float = 0.0
    use_separate_proj_weight: bool = True
    requires_input_projection: bool = True


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 dim_model: int,
                 attention_params: AttentionParams,
                 use_conv1d_proj: bool = False,
                 rel_pos_encoder: nn.Module = None,
                 dim_key: Optional[int] = None,
                 dim_value: Optional[int] = None,
                 out_proj: Optional[nn.Module] = None,
                 ):
        super().__init__()

        # Popular default is that all latent dimensions are the same
        dim_key, dim_value = map(lambda x: x if x else dim_model, (dim_key, dim_value))

        self.rel_pos_encoder = rel_pos_encoder
        self.num_heads = attention_params.num_heads
        self.dim_k = dim_key // self.num_heads
        self.dim_value = dim_value
        self.dim_model = dim_model
        self.attention = build_attention(attention_params)
        self.requires_input_projection = attention_params.requires_input_projection
        self.use_conv1d_proj = use_conv1d_proj

        if use_conv1d_proj:
            LinearProj = partial(nn.Conv1d, bias=attention_params.bias, kernel_size=1)
        else:
            LinearProj = partial(nn.Linear, bias=attention_params.bias)

        if attention_params.use_separate_proj_weight:
            self.proj_q = LinearProj(dim_model, dim_key)
            self.proj_k = LinearProj(dim_model, dim_key)
            self.proj_v = LinearProj(dim_model, dim_value)
        else:
            # sharing weights
            assert dim_key == dim_value, "To share qkv projection " \
                                         "weights dimension of q, k, v should be the same"
            self.proj_q = LinearProj(dim_model, dim_key)
            self.proj_v, self.proj_k = self.proj_q, self.proj_q

        self.dropout = nn.Dropout(0.1, inplace=False)
        self.proj = out_proj if out_proj else LinearProj(dim_value, dim_value)
        if isinstance(self.proj, nn.Linear) and self.proj.bias is not None:
            constant_(self.proj.bias, 0.0)

    def _check(self, t, name):
        if self.use_conv1d_proj:
            d = t.shape[1]
        else:
            d = t.shape[2]
        assert (
            d % self.dim_k == 0
        ), f"the {name} embeddings need to be divisible by the number of heads"

    def _split_heads(self, tensor):
        """

        :param tensor: [batch_size, seq_len, embed_dim]
        :return:
            [batch_size * num_heads, seq_len, embed_dim // num_heads]
        """
        assert len(tensor.shape) == 3, "Invalid shape for splitting heads"

        batch_size, seq_len = tensor.shape[0], tensor.shape[1]
        embed_dim = tensor.shape[2]

        new_embed_dim = embed_dim // self.num_heads

        tensor = torch.reshape(tensor, (batch_size, seq_len, self.num_heads, new_embed_dim))

        # Transpose the matrix and flatten it so the outer-dimension will be the batch-size times number of heads
        tensor = torch.transpose(tensor, 1, 2).flatten(start_dim=0, end_dim=1).contiguous()
        return tensor

    def _combine_heads(self, tensor, batch_size):
        """

        :param tensor:  [batch_size * num_heads, seq_len, embed_dim // num_heads]
        :param batch_size:
        :return:
            [batch_size, seq_len, embed_dim]
        """
        assert len(tensor.shape) == 3, "Invalid shape to combine heads"

        tensor = tensor.unflatten(0, (batch_size, self.num_heads))
        tensor = torch.transpose(tensor, 1, 2)  # -> [batch_size, seq_len, num_heads, embed_dim // num_heads]
        seq_len = tensor.shape[1]
        embed_dim = tensor.shape[-1]

        # the new feature size, if we combine all the heads
        new_embed_dim = self.num_heads * embed_dim
        # Reshape the Tensor to remove the heads dimension and come back to a Rank-3 tensor
        tensor = torch.reshape(tensor, (batch_size, seq_len, new_embed_dim)).contiguous()
        return tensor

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                att_mask: Optional[torch.Tensor] = None):
        """

        :param query: tensor w/ shape [batch_size, channels, n_queries]
        :param key: tensor w/ shape [batch_size, channels, n_keyval]
        :param value: tensor w/ shape [batch_size, channels, n_keyval] ,
                       number of key and values are always the same
        :param att_mask: tensor w/ shape [batch_size, 1, n_keyval]
        :return:
        """
        if key is None:
            key = query
        if value is None:
            value = query

        # Check the dimensions properly
        self._check(query, "query")
        self._check(value, "value")
        self._check(key, "key")

        bs, _, q_len = query.size()  # Batch x Sequence x Embedding (latent)
        _, _, k_len = key.size()  # K, Q's sequence length could differ

        # Calculate query, key, values for all heads in batch
        if self.requires_input_projection:
            q, k, v = self.proj_q(query),  self.proj_k(key), self.proj_v(value)
        else:
            k, q, v = key, query, value

        if self.use_conv1d_proj:
            q = rearrange(q, 'b d l -> b l d')
            k = rearrange(k, 'b d l -> b l d')
            v = rearrange(v, 'b d l -> b l d')

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        position_bias = None
        if isinstance(self.rel_pos_encoder, nn.Module):
            position_bias = self.rel_pos_encoder(q, k)

        if att_mask is not None:
            att_mask = repeat(att_mask, "b 1 k_len -> b q_len k_len", q_len=q_len, k_len=k_len)
            att_mask = repeat(att_mask, 'b l1 l2 -> (b n_heads) l1 l2', n_heads=self.num_heads)

        z, attn_matrix = self.attention(
            q=q,
            k=k,
            v=v,
            att_mask=att_mask,
            return_attn_matrix=True,
            position_bias=position_bias
        )

        z = self._combine_heads(z, bs)
        if self.use_conv1d_proj:
            z = rearrange(z, 'b l d -> b d l')

        output = self.dropout(self.proj(z))
        return output


def build_attention(attention_params):
    if attention_params.type == 'full':
        return ScaledDotProduct(dropout=attention_params.dropout)
    else:
        raise ModuleNotFoundError(f"Attention with name {attention_params.type} is not found!")
