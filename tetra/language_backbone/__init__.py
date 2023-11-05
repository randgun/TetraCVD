import registry
import torch.nn as nn
from collections import OrderedDict
import tetra
from .bert_model import BertCrossLayer, BertEncoder

@registry.LANGUAGE_BACKBONES.register("Bio_CliniacalBERT")
@registry.LANGUAGE_BACKBONES.register("Biobert-v1.1")
@registry.LANGUAGE_BACKBONES.register("ClinicalBERT")
def build_bert_backbone(cfg):
    body = tetra.VarEncoder(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model

def build_backbone(cfg):
    assert cfg.language_backbone.model_name in registry.LANGUAGE_BACKBONES, \
        "cfg.language_backbone.model_name: {} is not registered in registry".format(
            cfg.language_backbone.model_name
        )
    return registry.LANGUAGE_BACKBONES[cfg.language_backbone.model_name](cfg)