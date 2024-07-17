
CUDA_VISIBLE_DEVICES=0 python profile_attention.py \
    --rank_k 1024 --rank_v 1024 --group_size 4 \
    --prompt_len 65536 | tee -a logs/palu.log

CUDA_VISIBLE_DEVICES=1 python profile_attention.py \
    --rank_k 1024 --rank_v 3072 --group_size 4 \
    --prompt_len 65536 --palu | tee -a logs/palu.log

CUDA_VISIBLE_DEVICES=0 python profile_attention.py \
    --rank_k 1024 --rank_v 3072 --group_size 4 \
    --prompt_len 65536 --palu_no_rope | tee -a logs/palu_no_rope.log

conda activate /home/scott306lr_l/envs/pytorch/
conda activate /home/max410011_l/.conda/envs/palu/




CUDA_VISIBLE_DEVICES=0 python profile_attention.py \
    --rank_k 1024 --rank_v 3072 --group_size 4 \
    --prompt_len 65536 --palu

