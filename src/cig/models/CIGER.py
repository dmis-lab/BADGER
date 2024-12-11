import pandas as pd
import torch
import torch.nn as nn

from .CIGER_modules.neural_fingerprint import NeuralFingerprint
from .CIGER_modules.attention import Attention
from .XIGER_modules import drug_encoder_modules as dem
# from .XIGER_modules import gene_encoder_modules as gem

def load_gene_encoder(conf):
    if conf.model_params.gene_embed == 'go':
        num_genes        = 978
        gene_input_dim   = 1107 # GO Terms

        G = pd.read_csv(conf.path.dataset+'/gene_features_GO_978.csv', header=None, index_col=0)
        G = torch.FloatTensor(G.sort_index(ascending=True).values)
        gene_embed_table = nn.Embedding(978, gene_input_dim).from_pretrained(G, freeze=False)


    elif conf.model_params.gene_embed == 'string':
        num_genes        = 978
        gene_input_dim   = 128 # node2vec dim

        G = pd.read_csv(conf.path.dataset+'/gene_features_STRING_978.csv', header=None, index_col=0)
        G = torch.FloatTensor(G.sort_index(ascending=True).values)
        gene_embed_table = nn.Embedding(978, gene_input_dim).from_pretrained(G, freeze=False)

    elif conf.model_params.gene_embed == 'random':
        num_genes        = 978
        gene_input_dim   = conf.model_params.hidden_dim
        
        gene_embed_table = nn.Embedding(num_genes, gene_input_dim)

    else:
        raise

    return gene_input_dim, gene_embed_table

def load_drug_encoder(conf):
    if conf.model_params.drug_embed == 'neural':
        drug_encoder = NeuralFingerprint(62, 6, conv_layer_sizes=[64, 64],
                                                output_size=conf.model_params.hidden_dim, 
                                                degree_list=[*range(6)])
    else:
        drug_encoder = dem.load_encoder_module(conf)

    return drug_encoder


class Net(nn.Module):
    def __init__(self, conf):
        super(Net, self).__init__()
        self.module = nn.ModuleDict()

        # Default Hyperparameters?
        if conf.model_params.baseline_default:
            print("Setting Default Model Hyperparameters")
            conf.model_params.gene_embed = 'go'
            conf.model_params.drug_embed = 'neural'
            H = 1024

        # Load Gene, Drug-related Embeddings/Encoder
        gene_input_dim, self.module['gene_embed'] = load_gene_encoder(conf)
        self.module['drug_embed']                 = load_drug_encoder(conf)
        H = conf.model_params.hidden_dim
        all_input_dim = H + gene_input_dim + conf.model_params.meta_dimension

        # Load Multimodal Encoder (Gene+Drug+Cell+@)
        self.module['encoder'] = nn.Sequential(nn.Linear(all_input_dim,H), 
                                               nn.ReLU(), 
                                               nn.Dropout(0.1),
                                               nn.Linear(H,H//2), 
                                               nn.ReLU(), 
                                               nn.Dropout(0.1))
        self.module['decoder'] = nn.Sequential(nn.Linear(H//2,128), 
                                               nn.ReLU(), 
                                               nn.Dropout(0.1),
                                               nn.Linear(128,32), 
                                               nn.ReLU(), 
                                               nn.Dropout(0.1),
                                               nn.Linear(32,1))

        self.module['attention'] = Attention(H//2, n_layers=1, n_heads=1, pf_dim=512, dropout=0.1)
        self.sigmoid = nn.Sigmoid()
        self.label_type = conf.dataprep.label_type

        for n, p in self.named_parameters():
            if p.dim() == 1: nn.init.constant_(p, 0.)
            else: torch.nn.init.xavier_uniform_(p)

    def set_default_hp(self, trainer):
        print("Setting Training Default Hyperparameters")
        trainer.num_epochs = 100
        trainer.batch_size = 256
        trainer.learning_rate = 0.003 

        return trainer

    def forward(self, batch):
        return_batch = dict()
        input_gene, input_drug, input_meta, input_smi, dge, sig_ids = batch
        input_drug = self.module['drug_embed'](mol_batch=input_drug, smi_batch=input_smi)

        if isinstance(self.module['drug_embed'], dem.SmilesChemBERTa):
            return_batch['dump/chemberta'], input_drug = input_drug
        # import pdb; pdb.set_trace()

        input_drug = torch.cat([input_drug, input_meta], dim=1)
        input_drug = input_drug.unsqueeze(1)
        input_drug = input_drug.repeat(1, 978, 1) # unsafe
        
        input_gene = self.module['gene_embed'](input_gene)
        
        input_all  = torch.cat([input_drug, input_gene], dim=2)

        encoded = self.module['encoder'](input_all)

        input_attn, attn = self.module['attention'](encoded, None)
        input_attn = input_attn + encoded

        decoded = self.module['decoder'](input_attn)

        if self.label_type == 'binary' or self.label_type == 'binary_reverse':
            return_batch['task/dge_pred'] = self.sigmoid(decoded.squeeze(2))
            return_batch['task/dge_true'] = dge
        elif self.label_type == 'real' or self.label_type == 'real_reverse':
            return_batch['task/dge_pred'] = decoded.squeeze(2)
            return_batch['task/dge_true'] = dge
        else:
            raise ValueError('Unknown label_type: %s' % self.label_type)
        return_batch['meta/id'] = sig_ids

        return return_batch