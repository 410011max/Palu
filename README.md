# Palu
Implementation for Palu.

## Evaluation end-to-end speed-up (attention module)
```
CUDA_VISIBLE_DEVICES=1 python profile_attention.py --prompt_len 65536
CUDA_VISIBLE_DEVICES=1 python profile_attention.py --prompt_len 65536 --palu
```

## Evaluation end-to-end speed-up (model)
To be done.