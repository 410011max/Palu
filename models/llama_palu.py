import torch
from torch import nn

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *

from .palu_attention import LlamaPaluAttention


class LlamaDecoderLayer_PALU(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        if not getattr(config, "use_flash", False):
            self.self_attn = LlamaPaluAttention(config=config, layer_idx=layer_idx)
        else:
            raise ValueError("Not support flash attention in PALU model.")

class LlamaModel_PALU(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_PALU(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

class LlamaForCausalLM_PALU(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_PALU(config)

