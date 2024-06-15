
CUDA_VISIBLE_DEVICES=0 python profile_attention.py \
    --rank_k 1024 --rank_v 1024 --group_size 4 \
    --prompt_len 65536 | tee -a logs/palu.log

CUDA_VISIBLE_DEVICES=1 python profile_attention.py \
    --rank_k 1024 --rank_v 1024 --group_size 4 \
    --prompt_len 65536 --palu | tee -a logs/palu.log


conda activate /home/max410011_l/.conda/envs/palu/