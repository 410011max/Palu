# Palu
Implementation for Palu.

## Evaluation end-to-end speed-up (attention module)
```
CUDA_VISIBLE_DEVICES=1 python profile_attention.py --prompt_len 65536
CUDA_VISIBLE_DEVICES=1 python profile_attention.py --prompt_len 65536 --palu
```

## Evaluation end-to-end speed-up (model)
To be done.

## Pending List
- [ ] Check correctness for speed-up experiment
- [ ] Integrate Palu code into this repo (including low-rank and quant)
- [ ] Evaluation for different tasks
    - [ ] zero-shot
    - [ ] perplexity
    - [ ] longbench