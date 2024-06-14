

CUDA_VISIBLE_DEVICES=2 python abx_rope.py --total_rank 1024  --num_head_groups 1
CUDA_VISIBLE_DEVICES=2 python abx_rope.py --total_rank 1024  --num_head_groups 8 
CUDA_VISIBLE_DEVICES=2 python abx_rope.py --total_rank 1024  --num_head_groups 16
CUDA_VISIBLE_DEVICES=2 python abx_rope.py --total_rank 1024  --num_head_groups 32

CUDA_VISIBLE_DEVICES=1 python abx_rope.py --total_rank 2048  --num_head_groups 8
CUDA_VISIBLE_DEVICES=2 python abx_rope.py --total_rank 2048  --num_head_groups 16
CUDA_VISIBLE_DEVICES=3 python abx_rope.py --total_rank 2048  --num_head_groups 32

CUDA_VISIBLE_DEVICES=1 python abx_rope_share_head.py --total_rank 1024  --num_head_groups 1
CUDA_VISIBLE_DEVICES=1 python abx_rope_share_head.py --total_rank 1024  --num_head_groups 8 
CUDA_VISIBLE_DEVICES=1 python abx_rope_share_head.py --total_rank 1024  --num_head_groups 32


CUDA_VISIBLE_DEVICES=1 python fuse_per_group.py --total_rank 1024 --num_head_groups 8
CUDA_VISIBLE_DEVICES=2 python fuse_per_group.py --total_rank 1024 --num_head_groups 16

CUDA_VISIBLE_DEVICES=1 python fuse_per_group.py --total_rank 2048 --num_head_groups 8
CUDA_VISIBLE_DEVICES=3 python fuse_per_group.py --total_rank 2048 --num_head_groups 16
CUDA_VISIBLE_DEVICES=3 python fuse_per_group.py --total_rank 2048 --num_head_groups 32

CUDA_VISIBLE_DEVICES=2 ncu --target-processes all --set detailed --import-source yes --section SchedulerStats --section WarpStateStats --section SpeedOfLight_RooflineChart --section SpeedOfLight_HierarchicalTensorRooflineChart \
    -f -o share_head python abx_rope_share_head.py --check

TRITON_PRINT_AUTOTUNING=1 CUDA_VISIBLE_DEVICES=1 python fuse_per_group.py --total_rank 1024 --num_head_groups 8

CUDA_VISIBLE_DEVICES=1 ncu --target-processes all --set detailed --import-source yes --section SchedulerStats --section WarpStateStats --section SpeedOfLight_RooflineChart --section SpeedOfLight_HierarchicalTensorRooflineChart \
    -f -o fuse_pytorch_gs32 python fuse_pytorch.py --total_rank 1024 --num_head_groups 1 --check

conda activate /home/scott306lr_l/env/pytorch/
