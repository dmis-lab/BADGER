import torch
import torch.nn as nn

from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer


def load_encoder_module(conf):
    hidden_dim = conf.model_params.hidden_dim
    bert_model = conf.model_params.chemberta.model_name

    if conf.model_params.drug_embed == 'chemberta':
        return SmilesChemBERTa(hidden_dim, bert_model)


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