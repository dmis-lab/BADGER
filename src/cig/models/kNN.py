import torch
import torch.nn as nn

from .BADGER_modules.set_transformer_modules import * 
from .BADGER_modules.encoder_sub_modules import * 

import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor
import os
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor
import os
import pickle
from tqdm import tqdm
import numpy as np

# Global class definition
class KNNWithProgress(KNeighborsRegressor):
    def fit(self, X, y):
        n_samples = X.shape[0]
        # Don't store progress bar as instance attribute
        with tqdm(total=n_samples, desc="Building KNN index") as pbar:
            # Call parent's fit method
            result = super().fit(X, y)
            # Update progress bar
            pbar.update(n_samples)
        return result
    
    def predict(self, X):
        predictions = []
        batch_size = 1000  # Adjust based on your memory constraints
        
        with tqdm(total=len(X), desc="Predicting") as pbar:
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_pred = super().predict(batch)
                predictions.append(batch_pred)
                pbar.update(len(batch))
        
        return np.concatenate(predictions)

class KNNLayer(nn.Module):
    def __init__(self, n_neighbors=5, output_dim=978):
        super(KNNLayer, self).__init__()
        self.n_neighbors = n_neighbors
        self.output_dim = output_dim
        self.knn = None
        self.fitted = False
    
    def fit(self, X, y):
        # Convert to numpy for sklearn compatibility
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        print("Fitting KNN model...")
        self.knn = KNNWithProgress(n_neighbors=self.n_neighbors)
        self.knn.fit(X, y)
        self.fitted = True
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("KNN model has not been fitted yet!")
            
        # Convert to numpy if it's a torch tensor
        device = X.device if isinstance(X, torch.Tensor) else None
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            
        predictions = self.knn.predict(X)
            
        return predictions

    def save_knn(self, save_path):
        if not self.fitted:
            raise ValueError("KNN model has not been fitted yet!")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving KNN model to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(self.knn, f)
            
    def load_knn(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No KNN model found at {load_path}")
            
        print(f"Loading KNN model from {load_path}")
        with open(load_path, 'rb') as f:
            self.knn = pickle.load(f)
        self.fitted = True

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
                 ):
        super(Net, self).__init__()
        self.conf = conf
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

        self.label_type = conf.dataprep.label_type
        self.drug_pooled = conf.model_params.badger.drug_pooled

        self.module['knn'] = KNNLayer(n_neighbors=5, output_dim=978)
                
        session_name = f'{self.conf.wandb.project_name}_{self.conf.wandb.session_name}'
        ckpt_path = os.path.join(self.conf.path.checkpoint, session_name)
        embed_ckpt_path = os.path.join('/hdd0/hajung/checkpoint', f'MLP_{self.conf.wandb.session_name}_fold_{self.conf.experiment.fold_num}')
        self.knn_save_path = f'{ckpt_path}_fold_{self.conf.experiment.fold_num}/knn_model.pkl'
        best_metric = '_'.join((self.conf.train_params.best_criteria_metric).split('/'))
        self.embedding_layer_path = f'{embed_ckpt_path}/best_epoch_{best_metric}.mdl'
        
        self.load_embedding_layer()

    
    def load_embedding_layer(self):
        chkpt = torch.load(self.embedding_layer_path, map_location=f"cpu")
        new_state_dict = {}
        for key in chkpt.keys():
            if 'mlp' in key:
                continue
            if 'diff_decoder' in key:
                continue
            if 'module' not in key:
                new_key = 'module.' + key
            else:
                new_key = key
            new_state_dict[new_key] = chkpt[key]

        self.load_state_dict(new_state_dict)
        self.to('cuda')
    
    
    def embed(self, data):
        return_batch = dict()
        X_fuse_list = []
        y_dge_list = []
        sig_ids_list = []
        print(f"Processing Drug Encoding in {len(data)} batches...")
        for idx, batch in tqdm(enumerate(data), total=len(data)):
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
            
            X_fuse_list.append(X_fuse.detach().cpu())
            y_dge_list.append(y_dge.detach().cpu())
            sig_ids_list += sig_ids
            torch.cuda.empty_cache()
        X_fuse_list = torch.cat(X_fuse_list)
        y_dge_list = torch.cat(y_dge_list)
        return X_fuse_list, y_dge_list, sig_ids_list
            

    def train(self, data):
        # data = self.concatenate_batches(data)
        X_fuse, y_dge, _ = self.embed(data) 
        self.module['knn'].fit(X_fuse, y_dge)
        self.module['knn'].save_knn(self.knn_save_path)
        

    def evaluate(self, data):
        # data = self.concatenate_batches(data)
        X_fuse, y_dge, sig_ids = self.embed(data)
        self.module['knn'].load_knn(self.knn_save_path)
        decoded = self.module['knn'].predict(X_fuse)
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