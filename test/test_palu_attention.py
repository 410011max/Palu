import torch
import torch.nn as nn
import pytest

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

import sys
sys.path.append("..")
from models.palu_attention import HeadwiseLowRankModule, LlamaPaluAttention


def test_init_equivalence():
    batch_size = 1
    seq_len = 5
    ranks = [2, 2]
    in_features = 6
    out_features = 6
    bias = False

    module = HeadwiseLowRankModule(ranks, in_features, out_features, bias)
    
    hidden_states = torch.randn(batch_size, seq_len, in_features)

    # 使用 forward 方法
    forward_output = module(hidden_states)
    
    # 使用 project_to_latent 和 reconstruct 方法
    latent_states = module.project_to_latent(hidden_states)
    reconstructed_output = module.reconstruct(latent_states)

    print(module.U.shape)
    print(module.U_list)

    # 檢查兩者是否相等
    assert torch.allclose(forward_output, reconstructed_output, atol=1e-4), \
        "Forward output and reconstructed output are not equal."

def test_SVD_equivalence():
    batch_size = 1
    seq_len = 5
    ranks = [3, 3]
    in_features = 6
    out_features = 6
    bias = False

    linear = nn.Linear(in_features, out_features, bias)
    svd_linear = HeadwiseLowRankModule.from_linear(linear, ranks)
    
    inputs = torch.randn(batch_size, seq_len, in_features)

    # 使用 linear
    linear_output = linear(inputs)
    
    # 使用 SVD linear
    svd_linear_output = svd_linear(inputs)

    # 檢查兩者是否相等
    assert torch.allclose(linear_output, svd_linear_output, atol=1e-4), \
        "Linear output and SVD Linear output are not equal."

def test_palu_attention_equivalence():
    batch_size = 1
    seq_len = 1
    config = LlamaConfig()
    config.group_size = 4
    config.num_groups = config.num_attention_heads // 4
    config.rank_k = [2048 // config.num_groups for _ in range(config.num_groups)]
    config.rank_v = 2048

    palu_attention = LlamaPaluAttention(config, 0).to('cuda:0')
    
    inputs = torch.randn(batch_size, seq_len, config.hidden_size).to('cuda:0')

    # 使用正常 forward
    normal_output, _, _ = palu_attention(inputs, test_forward=True)
    
    # 使用 recontruction kernel
    kernel_output, _, _ = palu_attention(inputs)

    print(normal_output[:10])
    print(kernel_output[:10])

    # 檢查兩者是否相等
    assert torch.allclose(normal_output, kernel_output, atol=1e-1), \
        "Linear output and kernel output are not equal."

if __name__ == "__main__":
    pytest.main()
    test_palu_attention_equivalence()