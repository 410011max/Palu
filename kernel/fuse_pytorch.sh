CUDA_VISIBLE_DEVICES=1 python fuse_pytorch.py --total_rank 2048 --num_head_groups 32
CUDA_VISIBLE_DEVICES=1 python fuse_pytorch.py --total_rank 2048 --num_head_groups 16
CUDA_VISIBLE_DEVICES=1 python fuse_pytorch.py --total_rank 2048 --num_head_groups 8
CUDA_VISIBLE_DEVICES=1 python fuse_pytorch.py --total_rank 2048 --num_head_groups 1
CUDA_VISIBLE_DEVICES=1 python fuse_pytorch.py --total_rank 1024 --num_head_groups 32
CUDA_VISIBLE_DEVICES=1 python fuse_pytorch.py --total_rank 1024 --num_head_groups 16
CUDA_VISIBLE_DEVICES=1 python fuse_pytorch.py --total_rank 1024 --num_head_groups 8
CUDA_VISIBLE_DEVICES=1 python fuse_pytorch.py --total_rank 1024 --num_head_groups 1