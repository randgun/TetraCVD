from tetra.language_backbone import build_backbone
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel, BertConfig
from tetra.language_backbone.bert_model import BertEncoder

class TierBert(nn.Module):
    def __init__(self, configs):
        super(TierBert, self).__init__()
        # self.backbone = build_backbone(configs)
        self.backbone = VarEncoder(configs)
        self.bert_dim = configs.language_backbone.bert_dim
        self.emb = nn.Linear(self.bert_dim, 1)
        self.softmax = nn.Softmax(dim=0)
        self.use_aggre = configs.language_backbone.use_aggre
        self.n_classes = configs.n_classes
        self.dropout = nn.Dropout(configs.language_backbone.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(self.bert_dim, self.bert_dim),
            # nn.LayerNorm(self.bert_dim),
            nn.ReLU(),
            # nn.SiLU(),
            nn.Linear(self.bert_dim, self.n_classes),
        )
        self.config = BertConfig(
            hidden_size=self.bert_dim,
            num_hidden_layers=configs.language_backbone.num_layers,
            num_attention_heads=configs.language_backbone.num_heads,
            intermediate_size=configs.language_backbone.bert_dim * 
                                configs.language_backbone.mlp_ratio,
        )
        self.encoder = BertEncoder(self.config)
        self.device = configs.device
        self.aggre_layers = configs.language_backbone.aggre_layers

    def forward(self, inputs, masks, seq_mask):
        batch_size, docu_len, seq_len = inputs.shape
        tensor_list = []
        mask_list = []

        for i in range(batch_size):
            input = inputs[i, :, :]
            mask = masks[i, :, :]
            
            output = self.backbone(input, mask)
            # print(output['aggregate'].shape)
            # print(output['embedded'].shape)
            if self.use_aggre == True:
                feature = output['aggregate']
            else:
                feature = output['embedded']
                feature = feature[:, 0, :].squeeze(1)
            
            '''
            # feature: [docu_len, self.bert_dim]
            embedded = self.emb(feature)
            
            # print(feature.shape)
            # print(embedded.shape)
            embedded = embedded.permute([1, 0])
            w = self.softmax(embedded)
            feature = torch.mm(w, feature)
            tensor_list.append(feature)
            # print(feature.shape)
            '''
            tensor_list.append(feature[None,...])
        
        docu_emb = torch.cat(tensor_list, dim=0)
        
        # print("-----------------------------------------------")
        # print(feature.shape)
        # print(docu_emb.shape)
        # print(seq_mask.shape)
        # TODO document level's mask
        extend_seq_masks = self.backbone.model.get_extended_attention_mask(seq_mask, seq_mask.size(), self.device)
        # print(extend_seq_masks.shape)
        outputs = self.encoder(docu_emb, attention_mask=extend_seq_masks)
    
        output = aggre(outputs, seq_mask, self.aggre_layers)

        if self.use_aggre == True:
                feature = output['aggregate']
        else:
            feature = output['embedded']
            feature = feature[:, 0, :].squeeze(1)
        
        probs = self.mlp(feature)

        ret = {
            "probs": probs,
            "embedded": feature,
            "hidden": outputs,
        }

        return ret
    
class VarEncoder(nn.Module):
    def __init__(self, configs):
        super(VarEncoder, self).__init__()
        self.configs = configs
        self.model_name = configs.language_backbone.model_name

        if self.model_name == "Bio_ClinicalBERT":
            self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.language_dim = 768

        elif self.model_name == "Biobert-v1.1":
            self.model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
            self.language_dim = 768

        elif self.model_name == "ClinicalBERT":
            self.model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        
        elif self.model_name == "Bio_Discharge_Summary_BERT":
            self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")

        else:
            raise NotImplementedError

        self.num_layers = configs.language_backbone.num_layers
        print(self.model.num_parameters())

    def forward(self, input, mask):

        outputs = self.model(
                input_ids=input,
                attention_mask=mask,
                output_hidden_states=True,
            )
        
        return aggre(outputs, mask, self.configs.language_backbone.aggre_layers)
    

def aggre(outputs, mask, aggre_layers=1):
    # outputs has 13 layers, 1 input layer and 12 hidden layers
    encoded_layers = outputs.hidden_states[1:]
    features = None
    # 提取后aggre_layers的特征
    features = torch.stack(encoded_layers[-aggre_layers:], 1).mean(1)

    # language embedding has shape [len(phrase), seq_len, language_dim]
    features = features / aggre_layers
    
    #[S, N, E] * [S, N, 1]
    # embedded[:, 0, :].squeeze(1) 提取每个句子的语义
    embedded = features * mask.unsqueeze(-1).float()
    aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float() + 1e-5)

    ret = {
        "aggregate": aggregate,
        "embedded": embedded,
        "masks": mask,
        "hidden": encoded_layers[-1]
    }

    return ret