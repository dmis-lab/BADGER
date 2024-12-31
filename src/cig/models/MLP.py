import torch
import torch.nn as nn

from .BADGER_modules.set_transformer_modules import * 
from .BADGER_modules.encoder_sub_modules import * 

class DiffDecoder(nn.Module):
    def __init__(self, conf):
        super(DiffDecoder, self).__init__()
        h = conf.model_params.hidden_dim 
        d = conf.model_params.dropout_rate

        self.decoder = nn.Sequential(nn.Linear(h, h//2),
                                     nn.LeakyReLU(),
                                     nn.Dropout(d),
                                     nn.Linear(h//2, 1))

    def forward(self, genes):
        return self.decoder(genes)


class MLPLayers(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=[256, 512], output_dim=978, dropout_rate=0.1):
        super(MLPLayers, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # 은닉층 구성
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # BatchNorm1d 대신 LayerNorm 사용
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # 출력층
        layers.extend([
            nn.Linear(current_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        self.mlp_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        
        # Apply MLP
        x = self.mlp_layers(x)
        
        
        return x
    
    
class Net(nn.Module):
    def __init__(self, 
                 conf, 
                 hidden_dims=[128, 256, 512, 978],  # 은닉층 차원
                 ):
        super(Net, self).__init__()
        
        H = conf.model_params.hidden_dim
        D = conf.model_params.dropout_rate
        
        if conf.model_params.cell_embed == 'similarity': C = 114
        elif conf.model_params.cell_embed == 'scgpt':    C = 512
        else:                                            C = H
        
        self.module = nn.ModuleDict()
        self.module['comp_encoder']        = load_drug_encoder(conf)
        self.module['cell_encoder']        = nn.Sequential(nn.Linear(C, H),
                                                           nn.ReLU(),
                                                           nn.Dropout(D))
        self.module['drugcell_fusion']     = nn.Sequential(nn.Linear(2*H,H),
                                                           nn.ReLU(),
                                                           nn.Dropout(D))
        self.module['diff_decoder']     = DiffDecoder(conf)

        self.label_type = conf.dataprep.label_type
        self.drug_pooled = conf.model_params.badger.drug_pooled
        # 은닉층 구성
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(D)
            ])
        
        self.module['mlp'] = MLPLayers(input_dim=128, hidden_dims=[256, 512], output_dim=978)
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, batch):
        return_batch = dict()
        X_drug, X_cell, X_smi, X_frp, m_frp, X_frg, X_pcp, m_pcp, y_dge, y_pw, sig_ids = batch
        # Drug Encoder
        X_drug = self.module['comp_encoder'](mol_batch=X_drug, smi_batch=X_smi, 
                                             pcp_batch=X_pcp, pcp_masks=m_pcp,
                                             frp_batch=X_frp, frp_masks=m_frp,
                                             frg_batch=X_frg)
        if isinstance(self.module['comp_encoder'], SmilesChemBERTa):
            if self.drug_pooled:
                return_batch['dump/chemberta'], _, X_drug = X_drug
                X_mask, X_drug = None, X_drug
            else:
                X_drug, X_mask, return_batch['dump/chemberta'] = X_drug
        else:
            X_drug, X_mask = X_drug

        # Cell Encoder
        X_cell = self.module['cell_encoder'](X_cell)

        # Drug-Cell Fusion!
        if X_drug.dim() == 3:
            X_cell = X_cell.unsqueeze(1).repeat(1,X_drug.size(1),1) * X_mask.unsqueeze(2)
        X_fuse = self.module['drugcell_fusion'](torch.cat([X_drug,X_cell],-1))
        if X_fuse.dim() == 3:
            X_fuse = X_fuse * X_mask.unsqueeze(2)

        # MLP를 통한 예측
        decoded = self.module['mlp'](X_fuse)

        return_batch = dict()
        return_batch['meta/id'] = sig_ids
        if self.label_type == 'binary' or self.label_type == 'binary_reverse':
            return_batch['task/dge_pred'] = self.sigmoid(decoded)
            return_batch['task/dge_true'] = y_dge
        elif self.label_type == 'real' or self.label_type == 'real_reverse':
            return_batch['task/dge_pred'] = decoded
            return_batch['task/dge_true'] = y_dge
        else:
            raise ValueError('Unknown label_type: %s' % self.label_type)
        
        return return_batch