import math
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    Cache, apply_rotary_pos_emb, 
    LlamaAttention, LlamaConfig,
)

from kernel.abx_rope import abx as recompute_k_gemv


class HeadwiseLowRankModule(nn.Module):
    """ Headwise Low-Rank module """
    def __init__(self, ranks, in_features, out_features, bias):
        super().__init__()

        self.ranks = ranks
        self.num_groups = len(ranks)
        self.in_features = in_features
        self.out_features = out_features
        self.group_dim = out_features // self.num_groups

        if (self.group_dim * self.num_groups) != self.out_features:
            raise ValueError(
                f"out_features must be divisible by num_groups (got `out_features`: {self.out_features}"
                f" and `num_groups`: {self.num_groups})."
            )

        self.VT = nn.Linear(in_features, sum(ranks), bias=False)
        nn.init.xavier_uniform_(self.VT.weight)

        # Create the list of linear layers first
        Us = []
        for r in ranks:
            linear_layer = nn.Linear(r, self.group_dim, bias=bias)
            nn.init.xavier_uniform_(linear_layer.weight)
            Us.append(linear_layer)

        self.U_list = nn.ModuleList(Us)

        # Initialize self.U with the same random weights
        self.U = nn.Parameter(torch.empty(self.group_dim, sum(ranks)))
        
        # Copy the initialized weights to each linear layer
        total_ranks = 0
        for i, r in enumerate(ranks):
            self.U.data[:, total_ranks:total_ranks + r] = self.U_list[i].weight.data.clone()
            total_ranks += r

        """"
        # Initialize both methods with the same random weights
        self.U = nn.Parameter(torch.empty(self.group_dim, sum(ranks)))
        nn.init.xavier_uniform_(self.U)
        
        Us = []
        total_ranks = 0
        for r in ranks:
            linear_layer = nn.Linear(r, self.group_dim, bias=bias)
            # Copy the initialized weights to make them the same
            linear_layer.weight = nn.Parameter(self.U[:, total_ranks:total_ranks + r].clone())
            total_ranks += r
            Us.append(linear_layer)

        self.U_list = nn.ModuleList(Us)
        """

    def forward(self, hidden_states: torch.Tensor):
        """ hidden_states: Tensor of shape (batch_size, seq_len, in_features) """
        assert hidden_states.dim() == 3, f"hidden_states should have 3 dimensions, got {hidden_states.dim()}"
        
        hidden_states = self.VT(hidden_states)

        # hidden_states: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        outputs = []
        total_ranks = 0
        for i in range(self.num_groups):
            outputs.append(self.U_list[i](hidden_states[:, :, total_ranks: total_ranks+self.ranks[i]]))
            total_ranks += self.ranks[i]

        return torch.cat(outputs, dim=-1)

    def project_to_latent(self, hidden_states: torch.Tensor):
        """ hidden_states: Tensor of shape (batch_size, seq_len, in_features) """
        assert hidden_states.dim() == 3, f"hidden_states should have 3 dimensions, got {hidden_states.dim()}"

        hidden_states = self.VT(hidden_states)

        return hidden_states
    
    def reconstruct(self, hidden_states: torch.Tensor):
        """ hidden_states: Tensor of shape (batch_size, seq_len, sum(ranks)) """
        assert hidden_states.dim() == 3, f"hidden_states should have 3 dimensions, got {hidden_states.dim()}"

        outputs = []
        total_ranks = 0
        for i in range(self.num_groups):
            outputs.append(F.linear(hidden_states[:, :, total_ranks : total_ranks+self.ranks[i]], 
                                    self.U[:, total_ranks : total_ranks+self.ranks[i]]))
            total_ranks += self.ranks[i]

        return torch.cat(outputs, dim=-1)

    @staticmethod
    def from_linear(
        old_module: nn.Linear,
        ranks: list,
    ):   
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features).float()

        wl = []
        wr = []
        for i in range(len(ranks)):
            l, s, r = torch.linalg.svd(w[i], full_matrices=False)
            l = l[:, 0:ranks[i]]
            s = s[0:ranks[i]]
            r = r[0:ranks[i], :]
            l = l.mul(s)

            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U_list[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U_list[i].weight.data.shape} != {wl[i].shape}")
            new_module.U_list[i].weight.data = wl[i].contiguous()

        # Initialize U Linear with the same weights
        new_module.U = nn.Parameter(torch.empty(new_module.group_dim, sum(ranks)))
        
        # Copy the initialized weights to each linear layer
        total_ranks = 0
        for i, r in enumerate(ranks):
            new_module.U.data[:, total_ranks:total_ranks + r] = new_module.U_list[i].weight.data.clone()
            total_ranks += r

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module

class LlamaPaluAttention(LlamaAttention):
    """
    Llama Attention with Low-Rank KV-Cache with Palu. This module inherits from
    `LlamaAttention` but change linear layer and add custom Triton kernel.
    """

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # self.rank_k = [128, 128, 128, 128, 128, 128, 128, 128]
        # self.rank_v = 1024
        # self.group_size = 4
        # self.num_groups = 8
        self.rank_k = config.rank_k
        self.rank_v = config.rank_v
        self.group_size = config.group_size
        self.num_groups = config.num_groups
        self.lr_head_dim = self.head_dim * self.rank_v // self.hidden_size
        self.fused_group_dim = self.rank_v // self.num_groups
        self.fused_hidden_dim = self.fused_group_dim * self.num_heads
        
        self.k_proj = HeadwiseLowRankModule(self.rank_k, self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.rank_v, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.fused_hidden_dim, self.hidden_size, bias=config.attention_bias)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        test_forward: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)
        key_h_states = self.k_proj.project_to_latent(hidden_states)
        value_h_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_h_states = key_h_states.view(bsz, q_len, self.num_groups, self.rank_k[0]).transpose(1, 2)
        value_h_states = value_h_states.view(bsz, q_len, self.num_groups, self.fused_group_dim).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2]
        kv_seq_len = key_h_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        # cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_h_states, value_h_states = past_key_value.update(key_h_states, value_h_states, self.layer_idx)


        if q_len > 1 or test_forward:
            # Prompting
            print("Normal forward")
            # Recompute the key states
            key_h_states = key_h_states.reshape(bsz, q_len, sum(self.rank_k))
            key_states = self.k_proj.reconstruct(key_h_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE after recomputing the key states
            cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        else:
            # Generating
            print('Kernel forward')
            # Apply our reconsturction kernel
            # A: (num_heads, 1, head_dim)
            # B: (num_heads, rank_per_groups, head_dim)
            # X: (num_head_groups, seq_len, rank_per_groups)
            A = query_states.squeeze(0)
            B = self.k_proj.U.view(self.num_heads, self.rank_k[0], self.head_dim)
            X = key_h_states.squeeze(0)
            attn_weights = recompute_k_gemv(A, B, X).unsqueeze(0)

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # attn_output = torch.matmul(attn_weights, value_states)

        attn_weights = attn_weights.reshape(bsz, self.num_groups, q_len * self.group_size, kv_seq_len)
        attn_output = torch.matmul(attn_weights, value_h_states)


        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
