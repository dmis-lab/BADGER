import pandas as pd
import torch
import torch.nn as nn

from .DeepCE_modules.neural_fingerprint import NeuralFingerprint
from .DeepCE_modules.drug_gene_attention import DrugGeneAttention
from .XIGER_modules import drug_encoder_modules as dem

def load_gene_encoder(conf):
    if conf.model_params.gene_embed == 'go':
        num_genes        = 978
        gene_input_dim   = 1107 # GO Terms

        G = pd.read_csv(conf.path.dataset+'/gene_features_GO_978.csv', header=None, index_col=0)
        G = torch.FloatTensor(G.sort_index(ascending=True).values)

        gene_encoder = nn.Sequential(nn.Embedding(978, gene_input_dim).from_pretrained(G, freeze=True),
                                     nn.Linear(gene_input_dim, gene_input_dim))

    elif conf.model_params.gene_embed == 'string':
        num_genes        = 978
        gene_input_dim   = 128 # node2vec dim

        G = pd.read_csv(conf.path.dataset+'/gene_features_STRING_978.csv', header=None, index_col=0)
        G = torch.FloatTensor(G.sort_index(ascending=True).values)
        
        gene_encoder = nn.Sequential(nn.Embedding(978, gene_input_dim).from_pretrained(G, freeze=True),
                                     nn.Linear(gene_input_dim, gene_input_dim))

    elif conf.model_params.gene_embed == 'random':
        num_genes        = 978
        gene_input_dim   = conf.model_params.hidden_dim
        
        gene_encoder = nn.Sequential(nn.Embedding(num_genes, gene_input_dim),
                                     nn.Linear(gene_input_dim, gene_input_dim))

    else:
        raise

    return gene_input_dim, gene_encoder

def load_drug_encoder(conf):
    if conf.model_params.drug_embed == 'neural':
        drug_encoder = NeuralFingerprint(62, 6, conv_layer_sizes=[16, 16],
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
            conf.model_params.gene_embed = 'string'
            conf.model_params.drug_embed = 'neural'
            conf.model_params.hidden_dim = 128
            conf.model_params.dropout_rate = 0.1

        D = conf.model_params.dropout_rate
        H = conf.model_params.hidden_dim

        # Load Gene, Drug-related Embeddings/Encoder
        gene_input_dim, self.module['gene_embed'] = load_gene_encoder(conf)
        self.module['drug_embed']                 = load_drug_encoder(conf)
        self.module['xros_attn']                  = DrugGeneAttention(gene_input_dim, gene_input_dim,
                                                                      n_layers=2, n_heads=4, pf_dim=512, dropout=D)
        self.module['meta_embed']                 = nn.Linear(conf.model_params.meta_dimension, 4)

        all_input_dim = H + gene_input_dim + 4
        # Load Multimodal Encoder (Gene+Drug+Cell+@)
        self.module['predictor'] = nn.Sequential(nn.ReLU(),
                                                 nn.Linear(all_input_dim,H), 
                                                 nn.ReLU(), 
                                                 nn.Linear(H,1))
        self.sigmoid = nn.Sigmoid()
        self.label_type = conf.dataprep.label_type

        for n, p in self.named_parameters():
            if 'xros_attn' not in n:
                if p.dim() == 1: nn.init.constant_(p, 0.)
                else: torch.nn.init.xavier_uniform_(p)

    def set_default_hp(self, trainer):        
        print("Setting Default Training Hyperparameters")
        trainer.num_epochs = 100
        trainer.batch_size = 256
        trainer.learning_rate = 0.0005

        return trainer

    def forward(self, batch): # also handle no meta?
        return_batch = dict()
        input_gene, input_drug, masks_drug, input_meta, input_smi, dge, sig_ids = batch
        
        input_drug_atomwise = self.module['drug_embed'](mol_batch=input_drug, smi_batch=input_smi)
        input_drug          = input_drug_atomwise.sum(1)
        input_drug          = input_drug.unsqueeze(1)
        input_drug          = input_drug.repeat(1, 978, 1)

        input_gene    = self.module['gene_embed'](input_gene)

        input_drge, _ = self.module['xros_attn'](input_gene, input_drug_atomwise, None, masks_drug) # find masks later
        input_drge    = torch.cat([input_drge, input_drug], dim=2)

        input_meta    = self.module['meta_embed'](input_meta)
        input_meta    = input_meta.unsqueeze(1).repeat(1,input_gene.size(1),1)
        input_all     = torch.cat([input_drge, input_meta], dim=2)

        predicted  = self.module['predictor'](input_all)
        if self.label_type == 'binary' or self.label_type == 'binary_reverse':
            return_batch['task/dge_pred'] = self.sigmoid(predicted.squeeze(2))
            return_batch['task/dge_true'] = dge
        elif self.label_type == 'real' or self.label_type == 'real_reverse':
            return_batch['task/dge_pred'] = predicted.squeeze(2)
            return_batch['task/dge_true'] = dge
        else:
            raise ValueError('Unknown label_type: %s' % self.label_type)
        return_batch['meta/id'] = sig_ids

        return return_batch
