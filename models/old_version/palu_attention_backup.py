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
        nn.init.normal_(self.VT.weight)

        # Create the list of linear layers first
        Us = []
        for r in ranks:
            linear_layer = nn.Linear(r, self.group_dim, bias=bias)
            nn.init.normal_(linear_layer.weight)
            Us.append(linear_layer)

        self.U_list = nn.ModuleList(Us)

        """
        # Copy the weights from U_list
        self.U = nn.Parameter(torch.empty(self.group_dim, sum(ranks)))
        total_ranks = 0
        for i, r in enumerate(ranks):
            self.U.data[:, total_ranks:total_ranks + r] = self.U_list[i].weight.data.clone()
            total_ranks += r

        # New Copy the weights from U_list
        self.U_new = nn.Parameter(torch.empty(sum(ranks), self.group_dim))
        total_ranks = 0
        for i, r in enumerate(ranks):
            self.U_new.data[total_ranks:total_ranks + r, :] = self.U_list[i].weight.data.T.clone()
            total_ranks += r
        """

        # New Head Copy the weights from U_list
        self.U_head_list = []
        dim_list = [0, 128, 256, 384]
        total_ranks = 0
        for i, r in enumerate(ranks):
            for dim in dim_list:
                self.U_head_list.append(self.U_list[i].weight.data[dim:dim+128, :].clone())
            total_ranks += r

        # Final b
        '''
        U_head_list_T = [self.U_head_list[i].T for i in range(len(self.U_head_list))]
        b = torch.stack(U_head_list_T)
        print(f'b: {b.size()}')
        b = b.reshape(self.num_groups, -1, self.ranks[i], 128)
        '''
        U_list_T = [x.weight.data.T for x in self.U_list]
        b = torch.stack(U_list_T)
        b = b.reshape(self.num_groups, self.ranks[0], 4, 128)
        b = b.transpose(1, 2)
        b = b.reshape(32, self.ranks[0], 128)
        self.b = nn.Parameter(b)
    
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

        # hidden_states: Tensor of shape (batch_size, seq_len, r1 + r2 + ... )
        outputs_1 = []
        outputs_2 = []
        total_ranks = 0
        for i in range(self.num_groups):
            outputs_1.append(self.U_list[i](hidden_states[:, :, total_ranks: total_ranks+self.ranks[i]]))
            # for j in range(4):
            #     outputs_2.append(F.linear(hidden_states[:, :, total_ranks: total_ranks+self.ranks[i]], 
            #                              self.U_head_list[i*4+j]))
            total_ranks += self.ranks[i]

        # x = hidden_states.reshape(-1, self.num_groups, self.ranks[0]).transpose(0, 1).unsqueeze(1)
        # U_head_list_T = [self.U_head_list[i].T for i in range(len(self.U_head_list))]
        # b = torch.stack(U_head_list_T)
        # b = b.reshape(self.num_groups, -1, self.ranks[i], 128)
        # torch.testing.assert_close(b, self.b, rtol=1e-3, atol=1e-3)
        # xb = (x @ self.b).reshape(32, -1, 128)

        # o = torch.cat(outputs_1, dim=-1).reshape(-1, 32, 128).transpose(0, 1)
        # torch.testing.assert_close(o, xb)
        # print('pass')


        return torch.cat(outputs_1, dim=-1)
    
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
        
        # New Head Copy the weights from U_list
        new_module.U_head_list = []
        dim_list = [0, 128, 256, 384]
        total_ranks = 0
        for i, r in enumerate(ranks):
            for dim in dim_list:
                new_module.U_head_list.append(new_module.U_list[i].weight.data[dim:dim+128, :].clone())
            total_ranks += r

        # Final b
        U_list_T = [x.weight.data.T for x in new_module.U_list]
        b = torch.stack(U_list_T)
        b = b.reshape(new_module.num_groups, new_module.ranks[0], 4, 128)
        b = b.transpose(1, 2)
        b = b.reshape(32, new_module.ranks[0], 128)
        new_module.b.data = b

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
        
        # self.total_rank_k = [128, 128, 128, 128, 128, 128, 128, 128]
        # self.total_rank_v = 1024
        # self.group_size = 4
        # self.num_groups = 8
        self.group_size = config.group_size
        self.num_groups = config.num_groups
        self.total_rank_k = config.total_rank_k
        self.total_rank_v = config.total_rank_v
        self.group_rank_k = self.total_rank_k // self.num_groups
        self.group_rank_v = self.total_rank_v // self.num_groups
        self.fused_hidden_dim_o = self.group_rank_v * self.num_heads
        
        self.rank_k_list = [self.group_rank_k for _ in range(self.num_groups)]
        self.rank_v_list = [self.group_rank_v for _ in range(self.num_groups)]
        self.k_proj = HeadwiseLowRankModule(self.rank_k_list, self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = HeadwiseLowRankModule(self.rank_v_list, self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # self.v_proj = nn.Linear(self.hidden_size, self.total_rank_v, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.fused_hidden_dim_o, self.hidden_size, bias=config.attention_bias)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        golden_kernel: bool = False,
        golden_fusion: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        key_h_states = self.k_proj.project_to_latent(hidden_states)

        if golden_fusion:
            value_states = self.v_proj(hidden_states)
        else:
            value_h_states = self.v_proj.project_to_latent(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_h_states = key_h_states.view(bsz, q_len, self.num_groups, self.group_rank_k).transpose(1, 2)
        if golden_fusion:
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        else:
            value_h_states = value_h_states.view(bsz, q_len, self.num_groups, self.group_rank_v).transpose(1, 2)

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


        if q_len > 1 or golden_kernel:
            print("Normal forward")
            # Prompting (Original implementation)
            # Recompute the key states
            X = key_h_states.squeeze(0)
            key_h_states = key_h_states.transpose(1, 2).reshape(bsz, kv_seq_len, self.total_rank_k)
            key_states = self.k_proj.reconstruct(key_h_states)
            key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE after recomputing the key states
            cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            attn_weights = torch.matmul(query_states[:, :, 0:1, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            # Kernel Golden
            """"""
            # key_h_states: (bsz, num_groups, seq_len, group_rank_k)
            A = query_states[:, :, 0:1, :].squeeze(0)
            B = self.k_proj.b

            x = X.unsqueeze(1)
            xb = (x @ self.k_proj.b.reshape(self.num_groups, self.group_size, self.group_rank_k, self.head_dim)).reshape(32, -1, 128).unsqueeze(0)
            # torch.testing.assert_close(key_states, xb)
            print('pass reconsturct')

            print(f'kv_seq_len: {kv_seq_len}')
            cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
            print(f'cos: {cos.size()}, sin: {sin.size()}')
            xb_rope, _ = apply_rotary_pos_emb(xb ,xb, cos, sin, position_ids)
            print(f'xb_rope: {xb_rope.size()}')
            axb = (query_states[:, :, 0:1, :] @ xb_rope.transpose(2, 3)) / math.sqrt(self.head_dim)
            torch.testing.assert_close(axb, attn_weights)
            print('oh yaaaa')

            print(f'A: {A.size()}, B: {B.size()}, X: {X.size()}')
            kernel_axb = recompute_k_gemv(A, B, X).unsqueeze(0) / math.sqrt(self.head_dim)
            print(f'axb: {axb.size()}')
            print(f'kernel_axb: {kernel_axb.size()}')

            for i in range(32):
                print(axb[0, i].data - kernel_axb[0, i].data)

            torch.testing.assert_close(attn_weights, kernel_axb, rtol=1e-3, atol=1e-3)
            print('kernel oh yaaaa')
            exit()

            return key_states, attn_weights, query_states
        else:
            print('Kernel forward')
            # Generating (Apply our reconsturction kernel)
            # A: (num_heads, 1, head_dim)
            # B: (num_heads, rank_per_groups, head_dim)
            # X: (num_head_groups, seq_len, rank_per_groups)
            A = query_states.squeeze(0)
            B = self.k_proj.U.T.reshape(self.num_heads, self.group_rank_k, self.head_dim)
            X = key_h_states.squeeze(0)
            attn_weights = recompute_k_gemv(A, B, X).unsqueeze(0) / math.sqrt(self.head_dim)

            return _, attn_weights, _

        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # return key_states, attn_weights, query_states

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
    
        if golden_fusion:
            print(golden_fusion)
            print('Original forward')
            # TODO: Original version
            # print(f'value_h_states: {value_h_states.size()}')
            # value_states = self.v_proj.reconstruct(value_h_states)
            # value_states = value_states.reshape(1, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            print(f'attn_weights: {attn_weights.size()}')
            print(f'value_states: {value_states.size()}')
            attn_output = torch.matmul(attn_weights, value_states)
            print(f'attn_output: {attn_output.size()}')
            # value_h_states = value_h_states.reshape(bsz, self.num_heads, q_len, self.head_dim)
        else:
            print('Fusion forward')
            # Fusion version
            # attn_weights: (bsz, num_groups, q_len * group_size, kv_seq_len)
            attn_h_weights = attn_weights.reshape(1, self.num_groups, q_len * self.group_size, q_len)

            attn_h_output = torch.matmul(attn_h_weights, value_h_states)

            # attn_h_output: (bsz, num_heads, q_len * group_size, group_rank)
            attn_output = attn_h_output.reshape(1, self.num_heads, q_len, self.group_rank_v)


        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    @staticmethod
    def from_attention(
        module: LlamaAttention,
        config: LlamaConfig,
        no_fusion: bool = False,
    ):
        new_module = LlamaPaluAttention(config, module.layer_idx)
        new_module.q_proj = module.q_proj
        # rank_k_list = [config.total_rank_k // config.num_groups for _ in range(config.num_groups)]
        new_module.k_proj = HeadwiseLowRankModule.from_linear(module.k_proj, new_module.rank_k_list)

        inputs = torch.randn(1, 64, config.hidden_size)
        new_output = new_module.k_proj.project_to_latent(inputs)
        new_output = new_module.k_proj.reconstruct(new_output)
        torch.testing.assert_close(module.k_proj(inputs), new_output, rtol=1e-4, atol=1e-4)

        # Decompose and fuse v_proj and o_proj
        rank_v_list = [config.total_rank_v // config.num_groups for _ in range(config.num_groups)]
        new_v_proj = HeadwiseLowRankModule.from_linear(module.v_proj, rank_v_list)

        # TODO: Make a no fusing version
        if no_fusion:
            new_module.v_proj = module.v_proj
            # new_module.v_proj = new_v_proj
            new_module.o_proj = module.o_proj
            return new_module

        # FIXME: Fusing version
        # new_module.v_proj = new_v_proj.VT
        new_module.v_proj = new_v_proj
        new_o_weight = torch.zeros(new_module.o_proj.weight.size())

        head_dim = module.head_dim
        num_groups = config.num_groups
        group_size = config.group_size
        group_rank = new_module.group_rank_v 

        total_dims_2, total_ranks, total_fused_dims = 0, 0, 0
        for _ in range(num_groups):
            total_dims = 0
            for _ in range(group_size):
                new_o_weight[:, total_fused_dims:total_fused_dims + group_rank] = \
                    module.o_proj.weight.data[:, total_dims_2:total_dims_2 + head_dim] @ \
                        new_v_proj.U.data[total_dims:total_dims + head_dim, total_ranks : total_ranks + group_rank]

                total_dims += head_dim
                total_dims_2 += head_dim
                total_fused_dims += group_rank

            total_ranks += group_rank

        with torch.no_grad():
            print(f'new_o_proj size: {new_module.o_proj.weight.size()}')
            new_module.o_proj.weight.copy_(new_o_weight)

        return new_module
