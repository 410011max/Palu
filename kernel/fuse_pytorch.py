import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import argparse


def torch_ax_repeat(a, x):
    x = x.repeat_interleave(a.shape[0]//x.shape[0], dim=0)
    ax = a @ x.to(torch.float16)
    return ax

def torch_ax(a, x):
    ax = a @ x.to(torch.float16)
    return ax


def run_benchmark(args):
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=args.target_seq_lens,
            line_arg="provider",
            line_vals=["q_proj", "WX", "torch_repeat", "torch"],
            line_names=["q_proj", "WX", "torch_repeat", "Torch"],
            styles=[("gray", "--"), ("green", "--"), ("blue", "-"), ("red", "-")],
            ylabel="us",
            plot_name=f"low-rank-fuse-rank-{args.total_rank}-group-{args.num_head_groups}",
            args={
                "dtype": torch.float16,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "total_rank": args.total_rank,
                "num_head_groups": args.num_head_groups, # number of head groups
            },
        ))

    @triton.testing.perf_report(configs)
    def bench_low_rank(num_heads, head_dim, total_rank, seq_len, num_head_groups, provider, dtype=torch.float16, device="cuda"):
        rank_per_groups = total_rank // num_head_groups
        num_group = num_heads // num_head_groups

        warmup = 25
        rep = 100
        A = torch.randn(num_heads, 1, seq_len, dtype=dtype, device=device)
        A_gs = A.reshape(num_head_groups, num_group, seq_len)
        X = torch.randn(num_head_groups, seq_len, rank_per_groups, dtype=dtype, device=device)
        org_A = torch.randn(num_heads, 1, seq_len, dtype=dtype, device=device)
        org_X = torch.randn(num_heads, head_dim, seq_len, dtype=dtype, device=device)
        
        q = torch.randn(1, 4096, dtype=dtype, device=device)
        q_w = torch.randn(4096 * (rank_per_groups // 128), 4096, dtype=dtype, device=device)
        
        quantiles = [0.5, 0.2, 0.8]
        
        if provider == "torch_repeat":
            def fn(): return torch_ax_repeat(A, X)
            ms, min_ms, max_ms = triton.testing.do_bench(
                fn, quantiles=quantiles, warmup=warmup, rep=rep)

        if provider == "torch":
            def fn(): return torch_ax(A_gs, X)
            ms, min_ms, max_ms = triton.testing.do_bench(
                fn, quantiles=quantiles, warmup=warmup, rep=rep)

        if provider == "WX":
            def fn(): return torch.matmul(org_A, org_X.transpose(-1, -2))
            ms, min_ms, max_ms = triton.testing.do_bench(
                fn, quantiles=quantiles, warmup=warmup, rep=rep)
        
        if provider == "q_proj":
            def fn(): return F.linear(q, q_w)
            ms, min_ms, max_ms = triton.testing.do_bench(
                fn, quantiles=quantiles, warmup=warmup, rep=rep)
        
        return ms*1000, min_ms*1000, max_ms*1000

    import os
    # create a directory to store the results
    os.makedirs('results', exist_ok=True)
    bench_low_rank.run(print_data=True, show_plots=True, save_path='results/')

def run_test(args):
    num_heads = args.num_heads
    total_rank = args.total_rank
    seq_len = 1024
    num_head_groups = args.num_head_groups
    num_group = num_heads // num_head_groups
    rank_per_groups = total_rank //  num_head_groups
    dtype = torch.float16
    device = "cuda"
    
    A = torch.randn(num_heads, 1, seq_len, dtype=dtype, device=device)
    A_gs = A.reshape(num_head_groups, num_group, seq_len)
    X = torch.randn(num_head_groups, seq_len, rank_per_groups, dtype=dtype, device=device)

    ax = torch_ax(A_gs, X)
    ax = ax.reshape(num_heads, 1, rank_per_groups)
    ax_repeat = torch_ax_repeat(A, X)
    
    print("Max diff: ", torch.max(torch.abs(ax - ax_repeat)))


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument("--total_rank", type=int, default=2048, help="Total rank")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of heads, default to 32 (llama)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension, default to 128 (llama)")
    parser.add_argument("--num_head_groups", type=int, default=8, help="Number of head groups")
    parser.add_argument("--target_seq_lens", nargs="+", type=int, 
                        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144], help="Target sequence lengths")
    parser.add_argument("--check", action="store_true", default=False, help="Check the correctness of the implementation")
    args = parser.parse_args()
    return args

def main(args):
    print("Start benchmarking fused low-rank KV Cache Kernels...")
    print("Total Rank: ", args.total_rank)
    print("Number of Heads: ", args.num_heads)
    print("Head Dimension: ", args.head_dim)
    print("Number of Head Groups: ", args.num_head_groups)
    print("Rank per Head Groups: ", args.total_rank // args.num_head_groups)
    if args.check:
        run_test(args)
    else:
        run_benchmark(args)

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    main(args)
    
