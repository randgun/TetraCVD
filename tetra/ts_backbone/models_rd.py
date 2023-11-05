import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
import numbers
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import uniform, glorot, zeros, ones, reset
from tetra.ts_backbone.Ob_propagation import Observation_progation, New_progation, Forth_progation



class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = d_model // 2

    def getPE(self, P_time):
        B = P_time.shape[1]

        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)

        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)

        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.cuda()
        return pe


class Raindrop_v2(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        nlayers = number of graph layers
        dropout = dropout rate (default 0.3)
        max_len = maximum sequence length
        d_static = dimension of original static feature 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
        global_structure = wighted connection in graph
        sensor_wise_mask = 
    """

    def __init__(self, configs):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.d_inp = configs.ts_backbone.d_inp
        self.d_ob = configs.ts_backbone.d_ob
        self.static = configs.ts_backbone.static
        self.d_static = configs.ts_backbone.d_static
        self.d_pe = configs.ts_backbone.d_pe
        self.aggreg = configs.ts_backbone.aggreg
        self.global_structure = configs.ts_backbone.global_structure
        self.sensor_wise_mask = configs.ts_backbone.sensor_wise_mask

        nhead = configs.ts_backbone.nhead
        nhid = configs.ts_backbone.nhid
        nlayers = configs.ts_backbone.nlayers
        dropout = configs.ts_backbone.dropout
        max_len = configs.ts_backbone.max_len
        MAX = configs.ts_backbone.MAX
        n_classes = configs.n_classes
        d_model = configs.ts_backbone.d_model

        # print(d_static, d_inp)
        if self.static:
            self.emb = nn.Linear(self.d_static, self.d_inp)

        # self.d_ob = int(d_model/d_inp)

        # 忽略
        self.encoder = nn.Linear(self.d_inp*self.d_ob, self.d_inp*self.d_ob)

        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_inp*(self.d_ob+16), nhead, nhid, dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.adj = torch.ones([self.d_inp, self.d_inp]).cuda()

        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob)).cuda()

        self.ob_propagation = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=self.d_inp, ob_dim=self.d_ob)

        self.ob_propagation_layer2 = Observation_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                           n_nodes=self.d_inp, ob_dim=self.d_ob)

        if self.sensor_wise_mask == False:
            d_final = d_model + self.d_pe + self.d_inp
        else:
            d_final = self.d_inp*(self.d_ob + self.d_pe + 1)

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)
    
    def forward(self, src, static, times, lengths):
        """Input to the model:
        T, N, F
        src = P: [T, N, F + F] : np.concatenate([Pnorm_tensor, M], axis=2), 第三维是特征向量 + mask向量
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        maxlen, batch_size = src.shape[0], src.shape[1]
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :self.d_inp]
        n_sensor = self.d_inp
        src = torch.repeat_interleave(src, self.d_ob, dim=-1)

        # self.R_u: (1, self.d_inp*self.d_ob)
        h = F.relu(src*self.R_u)
        pe = self.pos_encoder(times)
        if static is not None:
            emb = self.emb(static)

        h = self.dropout(h)

        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).cuda()

        step1 = True
        x = h
        if step1 == False:
            output = x
            distance = 0
        elif step1 == True:
            adj = self.global_structure.cuda()
            adj[torch.eye(self.d_inp).byte()] = 1

            # 非0边权下标
            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]

            batch_size = src.shape[1]
            n_step = src.shape[0]
            output = torch.zeros([n_step, batch_size, self.d_inp*self.d_ob]).cuda()

            use_beta = True
            if use_beta == True:
                alpha_all = torch.zeros([int(edge_index.shape[1]/2), batch_size]).cuda()
            else:
                alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).cuda()
            for unit in range(0, batch_size):
                stepdata = x[:, unit, :]
                p_t = pe[:, unit, :]

                stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
                stepdata = stepdata.reshape(self.d_inp, n_step*self.d_ob)

                stepdata, attentionweights = self.ob_propagation(stepdata, p_t=p_t, edge_index=edge_index, edge_weights=edge_weights,
                                 use_beta=use_beta,  edge_attr=None, return_attention_weights=True)

                edge_index_layer2 = attentionweights[0]
                edge_weights_layer2 = attentionweights[1].squeeze(-1)

                stepdata, attentionweights = self.ob_propagation_layer2(stepdata, p_t=p_t, edge_index=edge_index_layer2, edge_weights=edge_weights_layer2,
                                 use_beta=False,  edge_attr=None, return_attention_weights=True)

                stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
                stepdata = stepdata.permute([1, 0, 2])
                # stepdata: [n_step, self.d_inp*self.d_ob]
                stepdata = stepdata.reshape([-1, self.d_inp*self.d_ob])

                output[:, unit, :] = stepdata
                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)

        if self.sensor_wise_mask == True:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.d_inp*(self.d_ob+16))
        else:
            output = torch.cat([output, pe], axis=2)

        step2 = True
        if step2 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step2 == False:
            r_out = output

        sensor_wise_mask = self.sensor_wise_mask

        masked_agg = True
        if masked_agg == True:
            lengths2 = lengths.unsqueeze(1)
            mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if sensor_wise_mask:
                output = torch.zeros([batch_size,self.d_inp, self.d_ob+16]).cuda()
                extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
                for se in range(self.d_inp):
                    r_out = r_out.view(-1, batch_size, self.d_inp, (self.d_ob+16))
                    out = r_out[:, :, se, :]
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.d_inp*(self.d_ob+16)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)

        embedding = output.view([-1, self.d_inp, self.d_ob+16])

        if static is not None:
            output = torch.cat([output, emb], dim=1)
        probs = self.mlp_static(output)
        

        return probs, distance, embedding


class Raindrop_v3(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        nlayers = number of graph layers
        dropout = dropout rate (default 0.3)
        max_len = maximum sequence length
        d_static = dimension of original static feature 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
        global_structure = wighted connection in graph
        sensor_wise_mask = 
    """
    # 重写版本
    # 效果不太好且显存占用较大

    def __init__(self, configs):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.d_inp = configs.ts_backbone.d_inp
        self.d_ob = configs.ts_backbone.d_ob
        self.d_model = configs.ts_backbone.d_model
        self.static = configs.ts_backbone.static
        self.d_static = configs.ts_backbone.d_static
        self.d_pe = configs.ts_backbone.d_pe
        self.aggreg = configs.ts_backbone.aggreg
        self.global_structure = configs.ts_backbone.global_structure
        self.sensor_wise_mask = configs.ts_backbone.sensor_wise_mask

        nhead = configs.ts_backbone.nhead
        nhid = configs.ts_backbone.nhid
        nlayers = configs.ts_backbone.nlayers
        dropout = configs.ts_backbone.dropout
        max_len = configs.ts_backbone.max_len
        MAX = configs.ts_backbone.MAX
        n_classes = configs.n_classes

        if self.static:
            self.emb = nn.Linear(self.d_static, self.d_inp)

        # 忽略
        self.encoder = nn.Linear(self.d_inp*self.d_ob, self.d_inp*self.d_ob)

        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_model+self.d_pe, nhead, nhid, dropout, batch_first=True)
        else:
            encoder_layers = TransformerEncoderLayer(self.d_model+16, nhead, nhid, dropout, batch_first=True)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.adj = torch.ones([self.d_inp, self.d_inp]).cuda()

        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob)).cuda()
       
        self.ob_propagation = New_progation(heads=1, n_nodes=self.d_inp, d_pe=self.d_pe, d_model=self.d_model)

        self.ob_propagation_layer2 = New_progation(heads=1, n_nodes=self.d_inp, d_pe=self.d_pe, d_model=self.d_model)
        # print(self.d_model, self.d_pe, self.d_inp, n_classes)
        if self.static == False:
            d_final = self.d_inp * (self.d_model + self.d_pe)
        else:
            d_final = self.d_inp * (self.d_model + self.d_pe) + self.d_inp

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, n_classes),
        )

        self.increase_dim = nn.Sequential(
            nn.Linear(self.d_ob, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-10
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, static, times, lengths):
        """Input to the model:
        T, N, F: time, batch_size, d_inp
        src = P: [T, N, F + F] : np.concatenate([Pnorm_tensor, M], axis=2), 第三维是特征向量 + mask向量
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        maxlen, batch_size = src.shape[0], src.shape[1]
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :self.d_inp]
        src = torch.repeat_interleave(src, self.d_ob, dim=-1)
        src = src.reshape([maxlen, batch_size, self.d_inp, self.d_ob])

        h = self.increase_dim(src)
        h = self.dropout(h) # [T, N, F, d_model]
        

        pe = self.pos_encoder(times) # [T, N, d_pe]
        if static is not None:
            emb = self.emb(static)

        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).cuda()

        x = h        
        adj = self.global_structure.cuda()
        adj[torch.eye(self.d_inp).byte()] = 1

        # 非0边权下标
        edge_index = torch.nonzero(adj).T
        edge_weights = adj[edge_index[0], edge_index[1]]

        batch_size = src.shape[1]
        n_step = src.shape[0]
        output = torch.zeros([n_step, batch_size, self.d_inp, self.d_model+self.d_pe]).cuda()

        use_beta = True
        if use_beta == True:
            alpha_all = torch.zeros([int(edge_index.shape[1]/2), batch_size]).cuda()
        else:
            alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).cuda()

        for unit in range(0, batch_size):
            stepdata = x[:, unit, :, :]
            
            p_t = pe[:, unit, :]
            stepdata = stepdata.permute(1, 0, 2) # [F, T, d_model]
            # print("stepdata.shape:{}".format(stepdata.shape))

            stepdata, attentionweights = self.ob_propagation(stepdata, p_t=p_t, edge_index=edge_index, 
                                                             edge_weights=edge_weights, use_beta=use_beta,  
                                                             edge_attr=None, return_attention_weights=True,)
            if torch.sum(torch.isnan(stepdata)==True) != 0:
                print('----1-----')
                raise
            edge_index_layer2 = attentionweights[0]
            edge_weights_layer2 = attentionweights[1].squeeze(-1)

            stepdata, attentionweights = self.ob_propagation_layer2(stepdata, p_t=p_t, edge_index=edge_index_layer2, edge_weights=edge_weights_layer2,
                                use_beta=False,  edge_attr=None, return_attention_weights=True)

            if torch.sum(torch.isnan(stepdata)==True) != 0:
                print('-----2-------')
                raise
            t_tmp = p_t.unsqueeze(1).repeat([1, self.d_inp, 1])
            t_tmp = t_tmp.permute(1, 0, 2)
            # print(stepdata.shape, t_tmp.shape)
            stepdata = torch.concat((stepdata, t_tmp), dim=-1)
            src_mask = ~mask[unit,:]
            src_mask = src_mask.repeat(stepdata.shape[0], 1)
            stepdata = self.transformer_encoder(stepdata, src_key_padding_mask=src_mask)

            # [T, F, d_model+d_pe]
            stepdata = stepdata.permute(1, 0, 2)

            output[:, unit, :, :] = stepdata
            alpha_all[:, unit] = attentionweights[1].squeeze(-1)

        distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
        distance = torch.mean(distance)

        sensor_wise_mask = True
        r_out = output

        masked_agg = True
        
        lengths2 = lengths.unsqueeze(1)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        if sensor_wise_mask:
            output = torch.zeros([batch_size, self.d_inp, self.d_model+self.d_pe]).cuda()
            # [T, N, F]
            extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
            for se in range(self.d_inp):
                r_out = r_out.view(-1, batch_size, self.d_inp, self.d_model+self.d_pe)
                # out: [T, N, 1, d_model + d_pe]
                out = r_out[:, :, se, :]
                len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                output[:, se, :] = out_sensor
            # print('********************')
            # print(output.shape)
            output = output.view([-1, self.d_inp*(self.d_model+self.d_pe)])
        elif self.aggreg == 'mean':
            output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        
        if static is not None:
            output = torch.cat([output, emb], dim=1)
        # print(output.shape)
        # print(output)
        if torch.sum(torch.isnan(output)==True) != 0:
            print('-----3-------')
            raise
        output = self.mlp_static(output)
        if torch.sum(torch.isnan(output)==True) != 0:
            print('-----4-------')
            raise
        return output, distance, None
    

class Raindrop_v4(nn.Module):
    """Implement the raindrop stratey one by one."""
    """ Transformer model with context embedding, aggregation, split dimension positional and element embedding
    Inputs:
        d_inp = number of input features
        d_model = number of expected model input features
        nhead = number of heads in multihead-attention
        nhid = dimension of feedforward network model
        nlayers = number of graph layers
        dropout = dropout rate (default 0.3)
        max_len = maximum sequence length
        d_static = dimension of original static feature 
        MAX  = positional encoder MAX parameter
        n_classes = number of classes 
        global_structure = wighted connection in graph
        sensor_wise_mask = 
    """

    def __init__(self, configs):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'

        self.d_inp = configs.ts_backbone.d_inp
        self.d_ob = configs.ts_backbone.d_ob
        self.static = configs.ts_backbone.static
        self.d_static = configs.ts_backbone.d_static
        self.d_pe = configs.ts_backbone.d_pe
        self.aggreg = configs.ts_backbone.aggreg
        self.global_structure = configs.ts_backbone.global_structure
        self.sensor_wise_mask = configs.ts_backbone.sensor_wise_mask

        nhead = configs.ts_backbone.nhead
        nhid = configs.ts_backbone.nhid
        self.nlayers = configs.ts_backbone.nlayers
        dropout = configs.ts_backbone.dropout
        max_len = configs.ts_backbone.max_len
        MAX = configs.ts_backbone.MAX
        n_classes = configs.n_classes
        d_model = configs.ts_backbone.d_model

        # print(d_static, d_inp)
        if self.static:
            self.emb = nn.Linear(self.d_static, self.d_inp)

        # self.d_ob = int(d_model/d_inp)

        # 忽略
        self.encoder = nn.Linear(self.d_inp*self.d_ob, self.d_inp*self.d_ob)

        self.pos_encoder = PositionalEncodingTF(self.d_pe, max_len, MAX)

        if self.sensor_wise_mask == True:
            encoder_layers = TransformerEncoderLayer(self.d_inp*(self.d_ob+16), nhead, nhid, dropout)
        else:
            encoder_layers = TransformerEncoderLayer(d_model+16, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)

        self.adj = torch.ones([self.d_inp, self.d_inp]).cuda()

        self.R_u = Parameter(torch.Tensor(1, self.d_inp*self.d_ob)).cuda()

        self.graph_layers = nn.ModuleList([Forth_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=self.d_inp, ob_dim=self.d_ob, d_pe=self.d_pe)
                                            for _ in range(self.nlayers)])

        self.ob_propagation = Forth_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                    n_nodes=self.d_inp, ob_dim=self.d_ob, d_pe=self.d_pe)

        self.ob_propagation_layer2 = Forth_progation(in_channels=max_len*self.d_ob, out_channels=max_len*self.d_ob, heads=1,
                                                           n_nodes=self.d_inp, ob_dim=self.d_ob, d_pe=self.d_pe)

        if self.sensor_wise_mask == False:
            d_final = d_model + self.d_pe + self.d_inp
        else:
            d_final = self.d_inp*(self.d_ob + self.d_pe + 1)

        self.mlp_static = nn.Sequential(
            nn.Linear(d_final, d_final),
            nn.ReLU(),
            nn.Linear(d_final, n_classes),
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 1e-3
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.static:
            self.emb.weight.data.uniform_(-initrange, initrange)
        glorot(self.R_u)

    def iter_int(self, x, num, rate):
        for _ in range(num):
            x = int(x * rate)
        return x
    
    def forward(self, src, static, times, lengths):
        """Input to the model:
        T, N, F
        src = P: [T, N, F + F] : np.concatenate([Pnorm_tensor, M], axis=2), 第三维是特征向量 + mask向量
        static = Pstatic: [128, 9]: this one doesn't matter; static features
        times = Ptime: [215, 128]: the timestamps
        lengths = lengths: [128]: the number of nonzero recordings.
        """
        maxlen, batch_size = src.shape[0], src.shape[1]
        missing_mask = src[:, :, self.d_inp:int(2*self.d_inp)]
        src = src[:, :, :self.d_inp]
        n_sensor = self.d_inp
        src = torch.repeat_interleave(src, self.d_ob, dim=-1)

        # self.R_u: (1, self.d_inp*self.d_ob)
        h = F.relu(src*self.R_u)
        pe = self.pos_encoder(times)
        if static is not None:
            emb = self.emb(static)

        h = self.dropout(h)

        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).cuda()

        step1 = True
        x = h
        if step1 == False:
            output = x
            distance = 0
        elif step1 == True:
            adj = self.global_structure.cuda()
            adj[torch.eye(self.d_inp).byte()] = 1

            # 非0边权下标
            edge_index = torch.nonzero(adj).T
            edge_weights = adj[edge_index[0], edge_index[1]]

            batch_size = src.shape[1]
            n_step = src.shape[0]
            output = torch.zeros([n_step, batch_size, self.d_inp*self.d_ob]).cuda()

            use_beta = True
            if use_beta == True:
                # alpha_all = torch.zeros([int(edge_index.shape[1]/2), batch_size]).cuda()
                # TODO
                alpha_all = torch.zeros([self.iter_int(edge_index.shape[1], self.nlayers - 1, 0.5), batch_size]).cuda()
            else:
                alpha_all = torch.zeros([edge_index.shape[1],  batch_size]).cuda()

            for unit in range(0, batch_size):
                stepdata = x[:, unit, :]
                p_t = pe[:, unit, :]
        
                stepdata = stepdata.reshape([n_step, self.d_inp, self.d_ob]).permute(1, 0, 2)
                stepdata = stepdata.reshape(self.d_inp, n_step*self.d_ob)
                
                for i in range(self.nlayers):
                    if i == self.nlayers - 1:
                        use_beta = False
                    stepdata, attentionweights = self.graph_layers[i](stepdata, p_t=p_t, edge_index=edge_index, 
                                                            edge_weights=edge_weights, use_beta=use_beta, 
                                                            edge_attr=None, return_attention_weights=True)
                    stepdata = self.dropout(stepdata)
                    edge_index = attentionweights[0]
                    edge_weights = attentionweights[1].squeeze(-1)

                stepdata = stepdata.view([self.d_inp, n_step, self.d_ob])
                stepdata = stepdata.permute([1, 0, 2])
                # stepdata: [n_step, self.d_inp*self.d_ob]
                stepdata = stepdata.reshape([-1, self.d_inp*self.d_ob])

                output[:, unit, :] = stepdata
                alpha_all[:, unit] = attentionweights[1].squeeze(-1)

            distance = torch.cdist(alpha_all.T, alpha_all.T, p=2)
            distance = torch.mean(distance)

        if self.sensor_wise_mask == True:
            extend_output = output.view(-1, batch_size, self.d_inp, self.d_ob)
            extended_pe = pe.unsqueeze(2).repeat([1, 1, self.d_inp, 1])
            output = torch.cat([extend_output, extended_pe], dim=-1)
            output = output.view(-1, batch_size, self.d_inp*(self.d_ob+16))
        else:
            output = torch.cat([output, pe], axis=2)

        step2 = True
        if step2 == True:
            r_out = self.transformer_encoder(output, src_key_padding_mask=mask)
        elif step2 == False:
            r_out = output

        sensor_wise_mask = self.sensor_wise_mask

        masked_agg = True
        if masked_agg == True:
            lengths2 = lengths.unsqueeze(1)
            mask2 = mask.permute(1, 0).unsqueeze(2).long()
            if sensor_wise_mask:
                output = torch.zeros([batch_size, self.d_inp, self.d_ob+16]).cuda()
                extended_missing_mask = missing_mask.view(-1, batch_size, self.d_inp)
                for se in range(self.d_inp):
                    r_out = r_out.view(-1, batch_size, self.d_inp, (self.d_ob+16))
                    out = r_out[:, :, se, :]
                    len = torch.sum(extended_missing_mask[:, :, se], dim=0).unsqueeze(1)
                    out_sensor = torch.sum(out * (1 - extended_missing_mask[:, :, se].unsqueeze(-1)), dim=0) / (len + 1)
                    output[:, se, :] = out_sensor
                output = output.view([-1, self.d_inp*(self.d_ob+16)])
            elif self.aggreg == 'mean':
                output = torch.sum(r_out * (1 - mask2), dim=0) / (lengths2 + 1)
        elif masked_agg == False:
            output = r_out[-1, :, :].squeeze(0)

        embedding = output.view([-1, self.d_inp, self.d_ob+16])

        if static is not None:
            output = torch.cat([output, emb], dim=1)
        probs = self.mlp_static(output)
        

        return probs, distance, embedding