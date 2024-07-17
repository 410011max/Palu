import sys
import logging
from tqdm import tqdm
from functools import partial

import torch
import argparse
import os

import socket
from datetime import datetime
from torch.autograd.profiler import record_function

from transformers.models.llama.modeling_llama import LlamaConfig, DynamicCache, LlamaAttention
from models.llama_attention_palu import LlamaAttention_PALU, LlamaAttention_PALU_no_rope
from models.palu_attention import LlamaPaluAttention#, LlamaPaluAttentionNoRoPE

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile, file_postfix="prefilling", device="cuda:0"):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}_{file_postfix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}_{file_postfix}.html", device=device)

def build_attention(args):
    device = "cuda:0"
    dtype = torch.float16

    logging.info(f"Creating Attention, dtype: {dtype}, device: {device}")
    config = LlamaConfig()
    config.max_position_embeddings = 300000
    attention = LlamaAttention(config, layer_idx=0).to(dtype).to(device)
    
    return attention, config

def build_attention_palu(args):
    device = "cuda:0"
    dtype = torch.float16

    logging.info(f"Creating Attention_Palu, dtype: {dtype}, device: {device}")
    config = LlamaConfig()
    config.max_position_embeddings = 300000
    config.group_size = args.group_size
    config.num_groups = config.num_attention_heads // args.group_size
    config.rank_k = [args.rank_k // config.num_groups for _ in range(config.num_groups)]
    config.rank_v = args.rank_v
    logging.info(f"rank_k: {config.rank_k}, rank_v: {config.rank_v}, group_size: {config.group_size}, num_groups: {config.num_groups}")
    # attention_palu = LlamaAttention_PALU(config, layer_idx=0).to(dtype).to(device)
    attention_palu = LlamaPaluAttention(config, layer_idx=0).to(dtype).to(device)
    
    return attention_palu, config

def build_attention_palu_no_rope(args):
    device = "cuda:0"
    dtype = torch.float16

    logging.info(f"Creating Attention_Palu (w/o RoPE), dtype: {dtype}, device: {device}")
    config = LlamaConfig()
    config.max_position_embeddings = 300000
    config.group_size = args.group_size
    config.num_groups = config.num_attention_heads // args.group_size
    config.rank_k = args.rank_k
    config.rank_v = args.rank_v
    logging.info(f"rank_k: {config.rank_k}, rank_v: {config.rank_v}, group_size: {config.group_size}, num_groups: {config.num_groups}")
    attention_palu_no_rope = LlamaAttention_PALU_no_rope(config, layer_idx=0).to(dtype).to(device)
    # attention_palu_no_rope = LlamaPaluAttentionNoRoPE(config, layer_idx=0).to(dtype).to(device)
    
    return attention_palu_no_rope, config

def profile_tpot(model, cache_size_k, cache_size_v, cache_type=torch.float16, batch_size=1, prompt_len=1024, repeats=100,
                 cache_graph=False, torch_profile=False, outfile=""):
    logging.info(">>> Profiling TPOT (generation stage)")
    device = next(iter(model.parameters())).device
    
    cache_k = torch.randn(cache_size_k, dtype=cache_type, device=device)
    cache_v = torch.randn(cache_size_v, dtype=cache_type, device=device)
    past_key_value = DynamicCache()
    past_key_value.update(cache_k, cache_v, 0)
    
    position_ids = torch.arange(prompt_len, prompt_len+1)

    hidden_dim = 4096
    input_token = torch.randn((batch_size, 1, hidden_dim), dtype=torch.float16, device=device) # only input 1 token at a time

    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        with torch.cuda.stream(s):
            for _ in range(25):
                _ = model(input_token, past_key_value=past_key_value, position_ids=position_ids)
    torch.cuda.current_stream().wait_stream(s)

    if cache_graph:
        with torch.no_grad():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = model(input_token, past_key_value=past_key_value, position_ids=position_ids)
            
        def generate(new_input_token, past_key_value, position_ids):
            input_token.copy_(new_input_token)
            graph.replay()
            return out
    else:
        def generate(new_input_token, past_key_value, position_ids):
            out = model(new_input_token, past_key_value=past_key_value, position_ids=position_ids)
            return out

    new_input_token = torch.randn((batch_size, 1, hidden_dim), dtype=torch.float16, device=device) # only input 1 token at a time
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True) 
        end   = torch.cuda.Event(enable_timing=True) 
        start.record()
        for _ in range(repeats):
            generate(new_input_token, past_key_value=past_key_value, position_ids=position_ids)
        end.record()
        torch.cuda.synchronize()
    dur = start.elapsed_time(end)
    logging.info(f"Finished, prompt_len: {prompt_len}, latency: {dur/repeats:.2f} milliseconds (cache_graph={cache_graph})")

    if torch_profile:
        outfile_postfix = f"{outfile}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, file_postfix=outfile_postfix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=1, warmup=5, active=6) , repeat=1
                for _ in range(12):
                    generate(new_input_token, past_key_value, position_ids=position_ids)
                    prof.step()


def main(args):    
    
    if args.palu:
        attention, config = build_attention_palu(args)
        attention.eval()

        num_groups = config.num_groups
        # NOTE: Assuming uniform head_dim
        group_dim_k = config.rank_k[0]
        group_dim_v = config.rank_v // config.num_groups 
        # key_h_states = key_h_states.view(bsz, q_len, num_groups, rank_k[0]).transpose(1, 2)
        # value_h_states = value_h_states.view(bsz, q_len, num_groups, fused_group_dim).transpose(1, 2)
        cache_size_k = (args.batch_size, num_groups, args.prompt_len, group_dim_k)
        cache_size_v = (args.batch_size, num_groups, args.prompt_len, group_dim_v)
        profile_tpot(attention, cache_size_k, cache_size_v, torch.float16, args.batch_size, args.prompt_len, args.repeats, args.cache_graph, args.torch_profile, "tpot_palu_fp16")
    elif args.palu_no_rope:
        attention, config = build_attention_palu_no_rope(args)
        attention.eval()

        num_groups = config.num_groups
        # NOTE: Assuming uniform head_dim
        group_dim_k = config.rank_k // config.num_groups 
        group_dim_v = config.rank_v // config.num_groups 
        # key_h_states = key_h_states.view(bsz, q_len, num_groups, fused_group_dim).transpose(1, 2)
        # value_h_states = value_h_states.view(bsz, q_len, num_groups, fused_group_dim).transpose(1, 2)
        cache_size_k = (args.batch_size, num_groups, args.prompt_len, group_dim_k)
        cache_size_v = (args.batch_size, num_groups, args.prompt_len, group_dim_v)
        profile_tpot(attention, cache_size_k, cache_size_v, torch.float16, args.batch_size, args.prompt_len, args.repeats, args.cache_graph, args.torch_profile, "tpot_palu_fp16")
    else:
        attention, config = build_attention(args)
        attention.eval()

        # value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        cache_size = (args.batch_size, num_heads, args.prompt_len, head_dim)
        profile_tpot(attention, cache_size, cache_size, torch.float16, args.batch_size, args.prompt_len, args.repeats, args.cache_graph, args.torch_profile, "tpot_fp16")
    

if __name__ =='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--palu', action='store_true',
        help='Whether to use PALU attention.'
    )
    parser.add_argument(
        '--palu_no_rope', action='store_true',
        help='Whether to use PALU attention without ROPE.'
    )
    parser.add_argument(
        '--rank_k', type=int, default=1024,
        help='The rank of key matrix for PALU attention.'
    )
    parser.add_argument(
        '--rank_v', type=int, default=2048,
        help='The rank of value matrix for PALU attention.'
    )
    parser.add_argument(
        '--group_size', type=int, default=4,
        help='The group size for PALU attention.'
    )
    parser.add_argument(
        '--repeats', type=int, default=100,
        help='The number of profiling to repeat (default: 100)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='The input batch size to Mamba. (default: 1)'
    )
    parser.add_argument(
        '--prompt_len', type=int, default=1024,
        help='The number of input tokens to Mamba. (default: 1024)'
    )
    parser.add_argument(
        '--cache_graph', action='store_true', default=False,
        help='To enable CUDA graph cache, this only works for the generation stage (TPOT and TTLT)'
    )
    parser.add_argument(
        '--torch_profile', action='store_true',
        help='Whether to launch the pytorch profiler.'
    )
    
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)3d] %(message)s",
        datefmt="%d/%b/%Y %H:%M:%S",
        stream=sys.stdout)
    main(args)
