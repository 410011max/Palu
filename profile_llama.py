import sys
import logging
from functools import partial

import torch
import argparse

import socket
from datetime import datetime
from torch.autograd.profiler import record_function

from transformers.models.llama.modeling_llama import LlamaConfig, DynamicCache, LlamaForCausalLM, LlamaModel
from models.llama_palu import LlamaModel_PALU


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

def build_llama(args):
    device = "cuda:0"
    dtype = torch.float16

    logging.info(f"Creating llama model, dtype: {dtype}, device: {device}")
    config = LlamaConfig()
    config.max_position_embeddings = 300000
    config._attn_implementation = "eager"
    model = LlamaModel(config).to(device, dtype)
    print(model)

    return model, config

def build_llama_palu(args):
    device = "cuda:0"
    dtype = torch.float16

    logging.info(f"Creating llama_palu, dtype: {dtype}, device: {device}")
    config = LlamaConfig()
    config.max_position_embeddings = 300000
    config.group_size = args.group_size
    config.num_groups = config.num_attention_heads // args.group_size
    config.total_rank_k = args.rank_k
    config.total_rank_v = args.rank_v
    logging.info(f"rank_k: {config.total_rank_k}, rank_v: {config.total_rank_v}, group_size: {config.group_size}, num_groups: {config.num_groups}")
    model_palu = LlamaModel_PALU(config).to(device, dtype)
    
    return model_palu, config


def profile_ttft(model, batch_size=1, prompt_len=1024, repeats=100, torch_profile=False, outfile=""):
    logging.info(">>> Profiling TTFT (prefilling stage)")
    device = next(iter(model.parameters())).device
    
    logging.info(f"Testing (batch_size, prompt_len): ({batch_size}, {prompt_len})")
    
    prompt = torch.randint(100, size=(batch_size, prompt_len), device=device)

    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        with torch.cuda.stream(s):
            for _ in range(25):
                _ = model(prompt, use_cache=False)
    torch.cuda.current_stream().wait_stream(s)

    def generate(new_prompt):
        out = model(new_prompt, use_cache=False)
        return out
    
    new_prompt = torch.randint(100, size=(batch_size, prompt_len), device=device)


    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True) 
        end   = torch.cuda.Event(enable_timing=True) 
        start.record()
        for _ in range(repeats):
            generate(new_prompt)
        end.record()
        torch.cuda.synchronize()
    dur = start.elapsed_time(end)
    logging.info(f"Finished, prompt_len: {prompt_len}, latency: {dur/repeats:.2f} milliseconds")

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
                    with record_function("## forward ##"):
                        out = model(prompt)
                    prof.step()

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
                    with record_function("## forward ##"):
                        out = model(prompt)
                    prof.step()

def profile_tpot(model, cache_size_k, cache_size_v, cache_type=torch.float16, batch_size=1, prompt_len=1024, repeats=100,
                 cache_graph=False, torch_profile=False, outfile=""):
    logging.info(">>> Profiling TPOT (generation stage)")
    device = next(iter(model.parameters())).device
    
    cache_k = torch.randn(cache_size_k, dtype=cache_type, device=device)
    cache_v = torch.randn(cache_size_v, dtype=cache_type, device=device)
    past_key_values = DynamicCache()
    for i in range(32):
        past_key_values.update(cache_k, cache_v, i)

    # only input 1 token at a time
    input_token = torch.randint(100, size=(batch_size, 1), device=device)

    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        with torch.cuda.stream(s):
            for _ in range(25):
                _ = model(input_token, past_key_values=past_key_values)
    torch.cuda.current_stream().wait_stream(s)

    def generate(new_input_token, past_key_values):
        out = model(new_input_token, past_key_values=past_key_values)
        return out

    # only input 1 token at a time
    new_input_token = torch.randint(100, size=(batch_size, 1), device=device)

    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True) 
        end   = torch.cuda.Event(enable_timing=True) 
        start.record()
        for _ in range(repeats):
            generate(new_input_token, past_key_values)
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
                    generate(new_input_token, past_key_values)
                    prof.step()


def main(args):    
    
    if args.palu:
        model, config = build_llama_palu(args)
    else:
        model, config = build_llama(args)
    
    model.eval()
    
    if args.tpot:
        if args.palu:
            for prompt_len in args.prompt_len:
                num_groups = config.num_groups
                # NOTE: Assuming uniform head_dim
                group_dim_k = config.total_rank_k // config.num_groups 
                group_dim_v = config.total_rank_v // config.num_groups 
                cache_size_k = (args.batch_size, num_groups, prompt_len, group_dim_k)
                cache_size_v = (args.batch_size, num_groups, prompt_len, group_dim_v)
                profile_tpot(model, cache_size_k, cache_size_v, torch.float16, args.batch_size, prompt_len, args.repeats, args.cache_graph, args.torch_profile, "tpot_palu_fp16")
        else:
            for prompt_len in args.prompt_len:
                num_heads = config.num_attention_heads
                head_dim = config.hidden_size // num_heads
                cache_size = (args.batch_size, num_heads, prompt_len, head_dim)
                profile_tpot(model, cache_size, cache_size, torch.float16, args.batch_size, prompt_len, args.repeats, args.cache_graph, args.torch_profile, "tpot_fp16")
    
    if args.ttft:
        for prompt_len in args.prompt_len:
            if args.palu:
                profile_ttft(model, args.batch_size, prompt_len, args.repeats, args.torch_profile, "ttft_palu_fp16")
            else:
                profile_ttft(model, args.batch_size, prompt_len, args.repeats, args.torch_profile, "ttft_fp16")


if __name__ =='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--palu', action='store_true',
        help='Whether to use PALU.'
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
        '--prompt_len', nargs="+", type=int, default=[1024],
        help='The number of input tokens to Mamba. (default: 1024)'
    )
    parser.add_argument(
        '--ttft', action='store_true',
        help='Profile time to first token (TTFT, i.e. prefilling stage)'
    )
    parser.add_argument(
        '--tpot', action='store_true',
        help='Profile time per output token (TPOT) (TPOT, i.e. generation stage)'
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
