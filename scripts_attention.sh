
gss=(1 4 32)
rank_ks=(1024)
rank_vs=(1024)
prompt_lens=(128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144)

for prompt_len in ${prompt_lens[@]}; do
    CUDA_VISIBLE_DEVICES=0 python profile_attention.py \
        --prompt_len $prompt_len | tee -a logs/original.log

    for gs in ${gss[@]}; do
        for rank_k in ${rank_ks[@]}; do
            for rank_v in ${rank_vs[@]}; do
                CUDA_VISIBLE_DEVICES=0 python profile_attention.py \
                    --rank_k $rank_k --rank_v $rank_v --group_size $gs \
                    --prompt_len $prompt_len --palu | tee -a logs/palu.log
            done
        done
    done
done