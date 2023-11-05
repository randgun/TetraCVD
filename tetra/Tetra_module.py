import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig
from .language_backbone import BertCrossLayer, BertEncoder
from ts_backbone.models_rd import Raindrop_v4
from .language_backbone.tier_bert import TierBert, aggre
from utils.tools import init_weights, itc_loss
import os

class Tetra(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.device = configs.device
        self.structure = configs.tetra.structure
        self.itc = configs.tetra.itc

        bert_config = BertConfig(
                hidden_size=configs.tetra.hidden_size,
                num_hidden_layers=configs.tetra.num_layers,
                num_attention_heads=configs.tetra.num_heads,
                intermediate_size=configs.tetra.hidden_size * configs.tetra.mlp_ratio,
                max_position_embeddings=configs.tetra.docu_len,
                hidden_dropout_prob=configs.tetra.drop_rate,
                attention_probs_dropout_prob=configs.tetra.drop_rate,
            )
        
        self.cross_modal_text_transform = nn.Linear(configs.language_backbone.bert_dim, 
                                                    configs.tetra.hidden_size)
        self.cross_modal_text_transform.apply(init_weights)
        self.cross_modal_ts_transform = nn.Linear(configs.ts_backbone.d_ob+16, 
                                                     configs.tetra.hidden_size)
        self.cross_modal_ts_transform.apply(init_weights)

        self.token_type_embeddings = nn.Embedding(2, configs.tetra.hidden_size)
        self.token_type_embeddings.apply(init_weights)

        self.cross_modal_ts_layers = nn.ModuleList([BertCrossLayer(bert_config) 
                                                       for _ in range(configs.tetra.num_top_layer)])
        self.cross_modal_ts_layers.apply(init_weights)
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) 
                                                      for _ in range(configs.tetra.num_top_layer)])
        self.cross_modal_text_layers.apply(init_weights)


        self.cross_modal_ts_pooler = Pooler(configs.tetra.hidden_size)
        self.cross_modal_ts_pooler.apply(init_weights)
        self.cross_modal_text_pooler = Pooler(configs.tetra.hidden_size)
        self.cross_modal_text_pooler.apply(init_weights)

        self.text_model = TierBert(configs)
        self.ts_model = Raindrop_v4(configs)

        self.encoder = BertEncoder(bert_config)

        self.d_ob = configs.ts_backbone.d_ob
        self.d_inp = configs.ts_backbone.d_inp

        self.increase_dim = nn.Linear((self.d_ob+16)*self.d_inp, configs.tetra.hidden_size)
        self.linear = nn.Linear(configs.tetra.hidden_size, 1)

        if configs.is_training == 1:
            task_num = 0 if configs.task == 'multi-class' else 1
            self.text_model.load_state_dict(torch.load(os.path.join('./language_backbone/checkpoints/' + 
                                                                    configs.tetra.text_weights[task_num], 'checkpoint.pth')))
            self.ts_model.load_state_dict(torch.load(os.path.join('./ts_backbone/checkpoints/' + 
                                                                    configs.tetra.ts_weights[task_num], 'checkpoint.pth')))

        hs = configs.tetra.hidden_size
        if self.structure == 'double':
            self.classifier = nn.Sequential(
                    nn.Linear(hs * 2, hs * 2),
                    # nn.LayerNorm(hs * 2),
                    # nn.GELU(),
                    nn.ReLU(),
                    nn.Linear(hs * 2, configs.n_classes),
                )
        elif self.structure == 'single':
            self.classifier = nn.Sequential(
                    nn.Linear(hs, hs),
                    # nn.LayerNorm(hs),
                    # nn.GELU(),
                    nn.ReLU(),
                    nn.Linear(hs, configs.n_classes),
                )
        elif self.structure == 'concat':
            d = (self.d_ob+16) + hs
            self.classifier = nn.Sequential(
                    nn.Linear(d, d),
                    nn.ReLU(),
                    nn.Linear(d, configs.n_classes),
                )
        self.classifier.apply(init_weights)


    def forward(self, batch):

        ts_token_type_idx = 1
        text_ids, attention_mask, seq_mask, ts_P,  ts_Ptime, ts_Pstatic, lengths = batch
        
        text_modal = self.text_model(text_ids, attention_mask, seq_mask)
        text_embedding = text_modal['hidden'].hidden_states[-1]
        # print(text_embeds.shape)
        text_embeds = self.cross_modal_text_transform(text_embedding)
        input_shape = seq_mask.size()
        extend_text_masks = self.text_model.backbone.model.get_extended_attention_mask(seq_mask, input_shape, self.device)

        ts_modal = self.ts_model(ts_P, ts_Pstatic, ts_Ptime, lengths)
        # print(ts_embeds.shape)
        ts_embedding = ts_modal[-1]
        ts_embeds = self.cross_modal_ts_transform(ts_embedding)
        ts_masks = torch.ones((ts_embeds.size(0), ts_embeds.size(1)), dtype=torch.long, device=self.device)
        extend_ts_masks = self.text_model.backbone.model.get_extended_attention_mask(ts_masks, ts_masks.size(), self.device)
    
        text_embeds, ts_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(seq_mask)),
            ts_embeds + self.token_type_embeddings(torch.full_like(ts_masks, ts_token_type_idx))
        )
        
        loss = torch.tensor(0, dtype=torch.int32)
        if self.structure == 'double':
            x, y = text_embeds, ts_embeds
            for text_layer, ts_layer in zip(self.cross_modal_text_layers, self.cross_modal_ts_layers):
                x1 = text_layer(x, y, extend_text_masks, extend_ts_masks)
                y1 = ts_layer(y, x, extend_ts_masks, extend_text_masks)
                x, y = x1[0], y1[0]
            
            text_feats, ts_feats = x, y
            cls_feats_text = self.cross_modal_text_pooler(x)
            cls_feats_ts = torch.mean(y, dim=1)
            res = torch.cat((cls_feats_text, cls_feats_ts), dim=-1)
            if self.itc == True:
                loss = itc_loss(cls_feats_text, cls_feats_ts, torch.tensor(0.1))
            
        elif self.structure == 'single':
            txt_len = text_embeds.shape[1]
            embeds = torch.cat((text_embeds, ts_embeds), dim=1)
            masks = torch.cat((seq_mask, ts_masks), dim=-1)
            extend_masks = self.text_model.backbone.model.get_extended_attention_mask(masks, masks.size(), self.device)
            outputs = self.encoder(embeds, attention_mask=extend_masks)
            if self.itc == True:
                embedding = outputs.hidden_states[-1]
                txt_feature = embedding[:, :txt_len, :]
                cls_feats_text = torch.mean(txt_feature * seq_mask.unsqueeze(-1).float(), dim=1)
                ts_feature = embedding[:, txt_len:, :]
                cls_feats_ts = torch.mean(ts_feature, dim=1)
                loss = itc_loss(cls_feats_text, cls_feats_ts, torch.tensor(0.1))
            res = aggre(outputs, masks)['aggregate']
            
        elif self.structure == 'concat':
            # text_embedding = text_modal['hidden']
            # ts_embedding = ts_embedding.reshape(-1, (self.d_ob+16)*self.d_inp)
            ts_embedding = torch.mean(ts_embedding, dim=1)
            text_embedding = text_modal['embedded']
            res = torch.cat((text_embedding, ts_embedding), dim=-1)
            # h = self.linear(embedding).permute(0, 2, 1)
            # res = torch.bmm(h, embedding).squeeze(1)

        else:
            raise NotImplementedError

        probs = self.classifier(res)

        ret = {
                "cls_feats": res,
                "probs": probs,
                "itc_loss": loss
            }
        
       
        return ret


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
