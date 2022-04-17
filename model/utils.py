from ast import Str
import torch
import torch.nn as nn
from transformers import (
    AdamW,
    AutoModel,
    AutoConfig,
)

def get_label_model(label_model: str, pretrained: bool):

    if pretrained:
        return AutoModel.from_pretrained(label_model)
    else:
        model_config = AutoConfig.from_pretrained(label_model)
        return AutoModel.from_config(model_config)
