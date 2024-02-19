import torch
import torch.nn as nn
from .neural_fingerprint import *
import pandas as pd

from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer

from torch_geometric.utils import to_dense_batch

NUM_TARGET_GENES = 19563 #879
NUM_SIGNAT_GENES = 978
# NUM_PATHWAYS     = 308
NUM_PATHWAYS     = 140
NUM_CELLTYPES    = 230

def load_gene_encoder(conf):
    if conf.model_params.gene_embed == 'go':
        num_genes        = 978
        gene_input_dim   = 1107 # GO Terms

        G = pd.read_csv(conf.path.dataset+'/gene_features_GO.csv', header=None, index_col=0)
        G = torch.FloatTensor(G.sort_index(ascending=True).values)
        gene_embed_table = nn.Embedding(978, gene_input_dim).from_pretrained(G, freeze=False)
    elif conf.model_params.gene_embed == 'string':
        num_genes        = 978
        gene_input_dim   = 128 # node2vec dim

        G = pd.read_csv(conf.path.dataset+'/gene_features_STRING.csv', header=None, index_col=0)
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
    hidden_dim   = conf.model_params.hidden_dim
    dropout_rate = conf.model_params.dropout_rate
    bert_model   = conf.model_params.chemberta.model_name

    if conf.model_params.drug_embed == 'neural':
        drug_encoder = NeuralFingerprint(62, 6, conv_layer_sizes=[64, 64],
                                                output_size=conf.model_params.hidden_dim, 
                                                degree_list=[*range(6)])
    elif conf.model_params.drug_embed == 'chemberta':
        drug_encoder = SmilesChemBERTa(hidden_dim, bert_model)
    elif conf.model_params.drug_embed == 'pharmacophore':
        drug_encoder = PharmacophoreEncoder(hidden_dim, dropout_rate)
    elif conf.model_params.drug_embed == 'fragmentecfp':
        drug_encoder = FragmentEcfpEncoder(hidden_dim, dropout_rate)
    elif conf.model_params.drug_embed == 'neuralfrag':
        drug_encoder = FragmentNeuralEncoder(hidden_dim, dropout_rate)
    else:
        raise

    return drug_encoder

def load_ppi_attention(conf):
    if conf.model_params.chemprof.gsab.attn_supervised == 'decagon_ppi':
        ppi = pd.read_csv(conf.path.dataset+'/ppi_landmark_snap_decagon.csv')
        ppi = torch.FloatTensor(ppi.values)
        ppi = nn.Parameter(ppi)
        ppi.requires_grad = False
        ppi = ppi / (ppi.sum(1) + 1e-5).view(-1,1) # there are zero-vectors?
    elif conf.model_params.chemprof.gsab.attn_supervised == 'diag':
        ppi = torch.diag(torch.ones((978)))
        ppi = nn.Parameter(ppi)
        ppi.requires_grad = False
    elif conf.model_params.chemprof.gsab.attn_supervised == 'string_ppi':
        ppi = pd.read_csv(conf.path.dataset+'/string_ppi_scaled.csv', index_col=0)
        ppi = torch.FloatTensor(ppi.values)
        ppi = nn.Parameter(ppi)
        ppi.requires_grad = False
        ppi = ppi / (ppi.sum(1) + 1e-5).view(-1,1) # there are zero-vectors?
    else:
        ppi = None
    return ppi

def load_pathway_embeddings(conf):
    if conf.model_params.chemprof.pathmab.path_embed == 'kegg':
        path_emb = pd.read_csv(conf.path.dataset+'/pathway_embedding_info.csv', index_col=0)
        path_emb = torch.FloatTensor(path_emb.iloc[:, 2:].values).unsqueeze(0)
        path_emb = nn.Parameter(path_emb)
    elif conf.model_params.chemprof.pathmab.path_embed == 'msig':
        path_emb = pd.read_csv(conf.path.dataset+'/pathway_embedding_msig.csv', index_col=0)
        path_emb = path_emb.sort_index()
        path_emb = torch.FloatTensor(path_emb.values).unsqueeze(0)
        path_emb = nn.Parameter(path_emb)
    else:
        path_emb = nn.Parameter(torch.randn(1, NUM_PATHWAYS, conf.model_params.hidden_dim))
        nn.init.orthogonal_(path_emb)

    # path_emb.requires_grad = False
    return path_emb

def load_signature_gene_embeddings(conf):
    h = conf.model_params.hidden_dim 
    d = conf.model_params.dropout_rate
    if conf.model_params.gene_embed == 'go':
        G = pd.read_csv(conf.path.dataset+'/gene_features_GO_978.csv', header=None, index_col=0)
        G = torch.FloatTensor(G.sort_index(ascending=True).values)
        gene_emb = nn.Parameter(G.unsqueeze(0))
        gene_emb.requires_grad = False
        encoder  = nn.Linear(1107, h)
    elif conf.model_params.gene_embed == 'string':
        G = pd.read_csv(conf.path.dataset+'/gene_features_STRING_978.csv', header=None, index_col=0)
        G = torch.FloatTensor(G.sort_index(ascending=True).values)
        gene_emb = nn.Parameter(G.unsqueeze(0))
        gene_emb.requires_grad = False
        encoder  = None
    else:
        gene_emb = nn.Parameter(torch.randn(1, NUM_SIGNAT_GENES, h))
        encoder  = None

    return gene_emb, encoder


class PharmacophoreEncoder(nn.Module):
    def __init__(self, h: int, d: float):
        super(PharmacophoreEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Embedding(39972+1, h*2, padding_idx=39972),
                                     nn.Linear(h*2, h),
                                     nn.ReLU(),
                                     nn.Dropout(d))

    def forward(self, **kwargs):
        X = kwargs['pcp_batch']
        m = kwargs['pcp_masks']

        return self.encoder(X), m

class FragmentEcfpEncoder(nn.Module):
    def __init__(self, h: int, d: float, ):
        super(FragmentEcfpEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(1024, h*2), 
                                     nn.ReLU(), 
                                     nn.Dropout(d), 
                                     nn.Linear(h*2,h))

    def forward(self, **kwargs):
        X = kwargs['frp_batch']
        m = kwargs['frp_masks']

        return self.encoder(X), m

class FragmentNeuralEncoder(nn.Module):
    def __init__(self, h: int, d: float, ):
        super(FragmentNeuralEncoder, self).__init__()
        self.encoder = NeuralFingerprint(62, 6, conv_layer_sizes=[64, 64], output_size=h, degree_list=[*range(6)])

    def forward(self, **kwargs):
        graph_input = {'mol_batch': kwargs['frg_batch']}
        
        h           = self.encoder(**graph_input)
        X, m        = to_dense_batch(h, graph_input['mol_batch']['batch_index_global'])

        return X, m.float()

class SmilesChemBERTa(nn.Module):
    def __init__(self, h: int, bert_model: str):
        super(SmilesChemBERTa, self).__init__()
        # assert bert_model == 'seyonec/ChemBERTa-zinc-base-v1'
        self.bert = RobertaModel.from_pretrained(bert_model)
        self.tknr = RobertaTokenizer.from_pretrained(bert_model)

        if bert_model=='seyonec/ChemBERTa-zinc-base-v1':
            self.linear_embeddings = nn.Linear(768, h)
            self.linear_pooled     = nn.Linear(768, h)
        elif bert_model=='DeepChem/ChemBERTa-77M-MLM':
            self.linear_embeddings = nn.Linear(384, h)
            self.linear_pooled     = nn.Linear(384, h)
        elif bert_model=='DeepChem/ChemBERTa-77M-MTR':
            self.linear_embeddings = nn.Linear(384, h)
            self.linear_pooled     = nn.Linear(384, h)
        else:
            raise
        # self.encoder = nn.Sequential(nn.Linear(768, h), nn.LeakyReLU(), nn.LayerNorm(h))

    def forward(self, **kwargs):
        X = kwargs['smi_batch']
        tokenized = self.tknr(X, return_tensors='pt', padding=True, truncation=True)
        tokenized['input_ids'] = tokenized['input_ids'].cuda()
        tokenized['attention_mask'] = tokenized['attention_mask'].cuda()

        output = self.bert(**tokenized)
        
        embeddings = self.linear_embeddings(output.last_hidden_state)[:,1:,:]
        pooled     = self.linear_pooled(output.pooler_output)

        return embeddings, tokenized['attention_mask'][:,1:], pooled



