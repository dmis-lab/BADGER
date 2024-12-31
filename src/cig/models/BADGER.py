import pandas as pd
import torch
import torch.nn as nn

from .base import *
from .BADGER_modules.set_transformer_modules import * 
from .BADGER_modules.encoder_sub_modules import * 

NUM_TARGET_GENES = 19563 #879
NUM_SIGNAT_GENES = 978
NUM_PATHWAYS     = 308
NUM_CELLTYPES    = 230

class Drug2PathwaysMAB(nn.Module):
    def __init__(self, conf):
        super(Drug2PathwaysMAB, self).__init__()
        # Module Parameters
        h             = conf.model_params.hidden_dim 
        num_heads     = conf.model_params.badger.pathmab.num_heads
        attn_option   = conf.model_params.badger.pathmab.attn_option
        same_linear   = conf.model_params.badger.pathmab.same_linear
        norm_method   = conf.model_params.badger.pathmab.norm_method
        norm_affine   = conf.model_params.badger.pathmab.norm_affine
        clean_path    = conf.model_params.badger.pathmab.clean_path
        analysis_mode = conf.experiment.analysis_mode

        # Module Initialization
        pmx_args      = (h, num_heads, RFF(h), attn_option, same_linear, norm_method, norm_affine, clean_path, analysis_mode)
        self.pmx      = PoolingMultiheadCrossAttention(*pmx_args)

        # Trainable Embeddings
        self.pathways = load_pathway_embeddings(conf)
        self.pseudo   = nn.Parameter(torch.randn(1, 1, h))
        self.fillmask = nn.Parameter(torch.ones(1,1), requires_grad=False)

        self.representations = [] 
        if analysis_mode: pass 
        self.apply(initialization)
        # This Module is Attention-Supervisable 

    def forward(self, **kwargs):
        '''
            Query: Trainable Gene Pathways
            Key  : Drug Representation
            Value: Drug Representation
        '''
        drug = kwargs['drug_encoded']
        mask = kwargs['drug_masks']
        if drug.dim() == 2: drug = drug.unsqueeze(1)
        drug  = torch.cat([drug, self.pseudo.repeat(drug.size(0),1,1)],1)
        pmask = self.fillmask.repeat(drug.size(0),1)
        mask  = torch.cat([mask, pmask], 1) 

        target_paths, attention = self.pmx(X=self.pathways.repeat(drug.size(0),1,1), Y=drug, Ym=mask)

        return target_paths, attention # torch.Size([B, 308, h])

class Pathways2SignagenesMAB(nn.Module):
    def __init__(self, conf):
        super(Pathways2SignagenesMAB, self).__init__()
        # Module Parameters
        h             = conf.model_params.hidden_dim 
        d             = conf.model_params.dropout_rate
        num_heads     = conf.model_params.badger.signmab.num_heads
        attn_option   = conf.model_params.badger.signmab.attn_option
        same_linear   = conf.model_params.badger.signmab.same_linear
        norm_method   = conf.model_params.badger.signmab.norm_method
        norm_affine   = conf.model_params.badger.signmab.norm_affine
        clean_path    = conf.model_params.badger.signmab.clean_path
        analysis_mode = conf.experiment.analysis_mode
        
        # Module Initialization
        pmx_args      = (h, num_heads, RFF(h), attn_option, same_linear, norm_method, norm_affine, clean_path, analysis_mode)
        self.pmx      = PoolingMultiheadCrossAttention(*pmx_args)

        # Trainable Embeddings
        self.pseudo             = nn.Parameter(torch.randn(1, 1, h))
        self.gene_emb, self.enc = load_signature_gene_embeddings(conf)

        self.representations = [] 
        if analysis_mode: pass 
        self.apply(initialization)
        # This Module is NOT Attention-Supervisable, yet

    def forward(self, **kwargs):
        '''
            Query: Trainable Signature Genes
            Key  : Encoded Gene Pathways
            Value: Encoded Gene Pathways
        '''
        paths = kwargs['paths_encoded']
        paths = torch.cat([paths, self.pseudo.repeat(paths.size(0),1,1)],1)
        sgenes = self.enc(self.gene_emb) if self.enc else self.gene_emb
        signat_genes, attention = self.pmx(X=sgenes.repeat(paths.size(0),1,1), Y=paths)

        return signat_genes, attention # torch.Size([B, 978, h])

class Drug2SignagenesMAB(nn.Module):
    def __init__(self, conf):
        super(Drug2SignagenesMAB, self).__init__()
        # Module Parameters
        h             = conf.model_params.hidden_dim 
        d             = conf.model_params.dropout_rate
        num_heads     = conf.model_params.badger.xmab.num_heads
        attn_option   = conf.model_params.badger.xmab.attn_option
        same_linear   = conf.model_params.badger.xmab.same_linear
        norm_method   = conf.model_params.badger.xmab.norm_method
        norm_affine   = conf.model_params.badger.xmab.norm_affine
        clean_path    = conf.model_params.badger.xmab.clean_path
        analysis_mode = conf.experiment.analysis_mode

        # Module Initialization
        pmx_args      = (h, num_heads, RFF(h), attn_option, same_linear, norm_method, norm_affine, clean_path, analysis_mode)
        self.pmx      = PoolingMultiheadCrossAttention(*pmx_args)

        # Trainable Embeddings
        self.pseudo             = nn.Parameter(torch.randn(1, 1, h))
        self.gene_emb, self.enc = load_signature_gene_embeddings(conf)

        self.representations = [] 
        if analysis_mode: pass 
        self.apply(initialization)
        # This Module is NOT Attention-Supervisable.

    def forward(self, **kwargs):
        '''
            Query: Trainable Gene Pathways
            Key  : Drug Representation
            Value: Drug Representation
        '''
        drug = kwargs['drug_encoded']
        mask = kwargs['drug_masks']
        # import pdb; pdb.set_trace()
        if drug.dim() == 2: drug = drug.unsqueeze(1)
        # drug = torch.cat([drug, self.pseudo.repeat(drug.size(0),1,1)],1)
        sgenes = self.enc(self.gene_emb) if self.enc else self.gene_emb
        signat_genes, attention = self.pmx(X=sgenes.repeat(drug.size(0),1,1), Y=drug, Ym=mask)

        return signat_genes, attention # torch.Size([B, 978, h])

class Gene2GeneSAB(nn.Module):
    def __init__(self, conf):
        super(Gene2GeneSAB, self).__init__()
        # Module Parameters
        num_blocks    = conf.model_params.badger.gsab.num_blocks
        h             = conf.model_params.hidden_dim 
        num_ipoints   = conf.model_params.badger.gsab.isab.num_ipoints
        num_heads     = conf.model_params.badger.gsab.num_heads
        attn_option   = conf.model_params.badger.gsab.attn_option
        same_linear   = conf.model_params.badger.gsab.same_linear
        norm_method   = conf.model_params.badger.gsab.norm_method
        norm_affine   = conf.model_params.badger.gsab.norm_affine
        clean_path    = conf.model_params.badger.gsab.clean_path
        analysis_mode = conf.experiment.analysis_mode

        # Module Initialization
        sab_args      = (h, num_heads, RFF(h), attn_option, same_linear, norm_method, norm_affine, clean_path, analysis_mode)
        self.sab      = nn.ModuleList([SetAttentionBlock(*sab_args) for _ in range(num_blocks)])

        # Self-Attention Supervision?
        self.ppi_attn = load_ppi_attention(conf)

        self.representations = [] 
        if analysis_mode: pass 
        self.apply(initialization)
        # This Module is Attention-Supervisable.

    def forward(self, **kwargs):
        '''
            Query: Encoded Target Genes
            Key  : Encoded Target Genes
            Value: Encoded Target Genes
        '''
        genes     = kwargs['genes_encoded']
        attention = None
        for sab in self.sab:
            genes, attention = sab(genes)
            # genes, attention = sab(genes, m=self.ppi_attn)

        return genes, attention # torch.Size([B, 978, h])

class Gene2GenePlusMAB(nn.Module):
    def __init__(self, conf):
        super(Gene2GenePlusMAB, self).__init__()
        # Module Parameters
        h             = conf.model_params.hidden_dim 
        d             = conf.model_params.dropout_rate
        num_heads     = conf.model_params.badger.plusmab.num_heads
        attn_option   = conf.model_params.badger.plusmab.attn_option
        same_linear   = conf.model_params.badger.plusmab.same_linear
        norm_method   = conf.model_params.badger.plusmab.norm_method
        norm_affine   = conf.model_params.badger.plusmab.norm_affine
        clean_path    = conf.model_params.badger.plusmab.clean_path
        analysis_mode = conf.experiment.analysis_mode

        # Module Initialization
        pmx_args      = (h, num_heads, RFF(h), attn_option, same_linear, norm_method, norm_affine, clean_path, analysis_mode)
        self.pmx      = PoolingMultiheadCrossAttention(*pmx_args)

        self.representations = [] 
        if analysis_mode: pass 
        self.apply(initialization)
        # This Module is NOT Attention-Supervisable.

    def forward(self, **kwargs):
        '''
            Query: Encoded Target Genes 
            Key  : Encoded Target Genes + Pathways
            Value: Encoded Target Genes + Pathways
        '''
        genes            = kwargs['genes_encoded']
        pathways         = kwargs['paths_encoded']
        attention        = None

        queries          = genes 
        keyvalues        = torch.cat([genes, pathways], dim=1) 
        genes, attention = self.pmx(X=queries,Y=keyvalues)

        return genes, attention

class DiffDecoder(nn.Module):
    def __init__(self, conf):
        super(DiffDecoder, self).__init__()
        h = conf.model_params.hidden_dim 
        d = conf.model_params.dropout_rate

        self.decoder = nn.Sequential(nn.Linear(h, h//2),
                                     nn.LeakyReLU(),
                                     nn.Dropout(d),
                                     nn.Linear(h//2, 1))
        self.apply(initialization)

    def forward(self, **kwargs):
        genes = kwargs['genes_encoded']

        return self.decoder(genes)

class DiffDecoderMultiway(nn.Module):
    def __init__(self, conf):
        super(DiffDecoderMultiway, self).__init__()
        h = conf.model_params.hidden_dim 
        d = conf.model_params.dropout_rate

        # self.decoder_batchnorm = nn.BatchNorm1d(h)
        # self.decoder = nn.Sequential(nn.Linear(h, h//2),
        #                              nn.Tanhshrink(),
        #                              nn.Dropout(d),
        #                              nn.Linear(h//2, 1))
        self.decoder_weights   = nn.Parameter(torch.randn(1, NUM_SIGNAT_GENES, h))
        self.decoder_bias      = nn.Parameter(torch.randn(1, NUM_SIGNAT_GENES))
        self.apply(initialization)
        nn.init.zeros_(self.decoder_bias)
        nn.init.orthogonal_(self.decoder_weights)

    def forward(self, **kwargs):
        genes = kwargs['genes_encoded']
        # genes = self.decoder_batchnorm(genes.transpose(1,2)).transpose(1,2)
        genes = (self.decoder_weights.repeat(genes.size(0),1,1) * genes).sum(2)
        # return self.decoder(genes)
        return (genes + self.decoder_bias.repeat(genes.size(0),1)).unsqueeze(2)

class Net(nn.Module):
    def __init__(self, conf):
        super(Net, self).__init__()
        self.module = nn.ModuleDict()
        H = conf.model_params.hidden_dim
        D = conf.model_params.dropout_rate
        self.pmab_loss = True if conf.train_params.badger.pathmab_coef > 0.0 else False
        self.gsab_loss = True if conf.train_params.badger.genesab_coef > 0.0 else False 
        self.gspo_loss = True if conf.train_params.badger.gene_spreadout_coef > 0.0 else False

        self.light_mode  = conf.model_params.badger.light_mode
        self.plus_ultra  = conf.model_params.badger.plus_ultra
        self.with_gsab   = conf.model_params.badger.with_gsab
        self.drug_pooled = conf.model_params.badger.drug_pooled

        if conf.model_params.cell_embed == 'similarity': C = 114
        elif conf.model_params.cell_embed == 'scgpt':    C = 512
        else:                                            C = H

        # Encoding Modules for Drug and Cell
        self.module['comp_encoder']        = load_drug_encoder(conf)
        self.module['cell_encoder']        = nn.Sequential(nn.Linear(C, H),
                                                           nn.ReLU(),
                                                           nn.Dropout(D))
        self.module['drugcell_fusion']     = nn.Sequential(nn.Linear(2*H,H),
                                                           nn.ReLU(),
                                                           nn.Dropout(D))
        self.module['drugcellgene_fusion'] = nn.Sequential(nn.Linear(2*H,H),
                                                           nn.ReLU(),
                                                           nn.Dropout(D))
        # Cross-Attention Modules invovling Genes
        if conf.model_params.badger.light_mode:
            self.module['xros_mab']     = Drug2SignagenesMAB(conf)
        else:
            self.module['path_mab']     = Drug2PathwaysMAB(conf)
            self.module['sign_mab']     = Pathways2SignagenesMAB(conf)
        
        # Self-Attention Module for Genes
        if self.with_gsab:
            if self.plus_ultra:
                self.module['plus_mab']     = Gene2GenePlusMAB(conf)
            else:
                self.module['gene_sab']     = Gene2GeneSAB(conf)

        # Downstream Predictor for DEGs
        if conf.model_params.badger.decoder_multiway:
            self.module['diff_decoder']     = DiffDecoderMultiway(conf)
        else:
            self.module['diff_decoder']     = DiffDecoder(conf)

        self.label_type = conf.dataprep.label_type # ??

    def set_default_hp(self, trainer):

        return trainer

    def load_auxiliary_materials(self, **kwargs):
        return_batch = kwargs['return_batch']

        # Drug & Pathway Genes Related
        if torch.is_tensor(kwargs['tg_cp_attn']) and self.pmab_loss:
            pw_drug_attn = kwargs['tg_cp_attn']
            pw_drug_true = kwargs['tg_cp_attn_true']
            temp         = pw_drug_true.sum(1)
            pw_drug_mask = torch.where(temp>0,1.,0.).unsqueeze(1)

            b       = pw_drug_true.size(0)
            x, y, z = pw_drug_attn.size()

            logits0 = pw_drug_attn.view(x//b,b,y,z).mean(0)[:,:,:-1].sum(2).unsqueeze(2)
            logits1 = pw_drug_attn.view(x//b,b,y,z).mean(0)[:,:,-1].unsqueeze(2)  

            return_batch['task/pw_drug_pred'] = torch.cat([logits1,logits0],2)
            return_batch['task/pw_drug_true'] = pw_drug_true[:,:].long()
            return_batch['task/pw_drug_mask'] = pw_drug_mask.repeat(1,y)

        # Gene-Gene Self-Attention
        if torch.is_tensor(kwargs['sg_sg_attn']) and self.gsab_loss:
            if torch.is_tensor(self.module['gene_sab'].ppi_attn):
                sg_self_attn = kwargs['sg_sg_attn']

                b       = sg_self_attn.size(0)
                x, y, z = sg_self_attn.size()

                sg_self_attn = sg_self_attn.view(x//b,b,y,z).mean(0).mean(0)
                sg_self_attn_true = self.module['gene_sab'].ppi_attn

                return_batch['task/sg_attn_pred'] = sg_self_attn
                return_batch['task/sg_attn_true'] = sg_self_attn_true

        # Other Attention Regularizations are NA
        if torch.is_tensor(kwargs['sg']) and self.gspo_loss:
            return_batch['regu/sg_spreadout'] = kwargs['sg']
            
        return return_batch

    def forward(self, batch):
        return_batch = dict()
        X_drug, X_cell, X_smi, X_frp, m_frp, X_frg, X_pcp, m_pcp, y_dge, y_pw, sig_ids = batch

        # Drug Encoder
        X_drug = self.module['comp_encoder'](mol_batch=X_drug, smi_batch=X_smi, 
                                             pcp_batch=X_pcp, pcp_masks=m_pcp,
                                             frp_batch=X_frp, frp_masks=m_frp,
                                             frg_batch=X_frg)
        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        # Attention Stuff
        if self.light_mode:
            sign_genes, sg_cp_attn = self.module['xros_mab'](drug_encoded=X_fuse, drug_masks=X_mask)
            tg_cp_attn, sg_tg_attn = None, None
        else:
            path_genes, tg_cp_attn = self.module['path_mab'](drug_encoded=X_fuse, drug_masks=X_mask)
            sign_genes, sg_tg_attn = self.module['sign_mab'](paths_encoded=path_genes)
            sg_cp_attn             = None
            X_fuse_pooled          = X_fuse.sum(1) / (X_mask.sum(1).unsqueeze(1) + 1e-5)
            X_fuse_pooled          = X_fuse_pooled.unsqueeze(1).repeat(1,sign_genes.size(1),1)
            sign_genes             = torch.cat([sign_genes,X_fuse_pooled],-1)
            sign_genes             = self.module['drugcellgene_fusion'](sign_genes)

        if self.with_gsab:
            if self.plus_ultra:
                attn_genes, sg_sg_attn     = self.module['plus_mab'](genes_encoded=sign_genes, paths_encoded=path_genes)
            else:
                attn_genes, sg_sg_attn     = self.module['gene_sab'](genes_encoded=sign_genes)
            sign_genes = sign_genes + attn_genes
        else:
            sg_sg_attn = None

        # Downstream Decoder
        decoded                    = self.module['diff_decoder'](genes_encoded=sign_genes)
        
        if self.label_type == 'binary' or self.label_type == 'binary_reverse':
            return_batch['task/dge_pred'] = self.sigmoid(decoded.squeeze(2))
            return_batch['task/dge_true'] = y_dge
        elif self.label_type == 'real' or self.label_type == 'real_reverse':
            return_batch['task/dge_pred'] = decoded.squeeze(2)
            return_batch['task/dge_true'] = y_dge
        else:
            raise ValueError('Unknown label_type: %s' % self.label_type)
        return_batch['meta/id'] = sig_ids

        return_batch = self.load_auxiliary_materials(return_batch=return_batch,
                                                     sg_cp_attn=sg_cp_attn,
                                                     sg_cp_attn_true=None,
                                                     tg_cp_attn=tg_cp_attn,
                                                     tg_cp_attn_true=y_pw,
                                                     sg_tg_attn=sg_tg_attn,
                                                     sg_tg_attn_true=None,
                                                     sg_sg_attn=sg_sg_attn,
                                                     sg_sg_attn_true=None,
                                                     sg=sign_genes) # not gradiented?

        return return_batch

    @torch.no_grad()
    def infer(self, batch):
        return_batch = dict()
        X_drug, X_cell, X_smi, X_frp, m_frp, X_frg, X_pcp, m_pcp, y_dge, y_pw, sig_ids = batch

        # Drug Encoder
        X_drug = self.module['comp_encoder'](mol_batch=X_drug, smi_batch=X_smi, 
                                             pcp_batch=X_pcp, pcp_masks=m_pcp,
                                             frp_batch=X_frp, frp_masks=m_frp,
                                             frg_batch=X_frg)
        X_drug, X_mask = X_drug

        # Cell Encoder
        X_cell = self.module['cell_encoder'](X_cell)

        return_batch['vecs/drug'] = X_drug
        return_batch['vecs/cell'] = X_cell

        # Drug-Cell Fusion!
        if X_drug.dim() == 3:
            X_cell = X_cell.unsqueeze(1).repeat(1,X_drug.size(1),1) * X_mask.unsqueeze(2)
        X_fuse = self.module['drugcell_fusion'](torch.cat([X_drug,X_cell],-1))
        if X_fuse.dim() == 3:
            X_fuse = X_fuse * X_mask.unsqueeze(2)

        # Attention Stuff
        if self.light_mode:
            sign_genes, sg_cp_attn = self.module['xros_mab'](drug_encoded=X_fuse, drug_masks=X_mask)
            tg_cp_attn, sg_tg_attn = None, None
        else:
            path_genes, tg_cp_attn = self.module['path_mab'](drug_encoded=X_fuse, drug_masks=X_mask)
            return_batch['vecs/pathways'] = path_genes
            sign_genes, sg_tg_attn = self.module['sign_mab'](paths_encoded=path_genes)
            sg_cp_attn             = None
            X_fuse_pooled          = X_fuse.sum(1) / (X_mask.sum(1).unsqueeze(1) + 1e-5)
            X_fuse_pooled          = X_fuse_pooled.unsqueeze(1).repeat(1,sign_genes.size(1),1)
            sign_genes             = torch.cat([sign_genes,X_fuse_pooled],-1)
            sign_genes             = self.module['drugcellgene_fusion'](sign_genes)

        if self.plus_ultra:
            sign_genes, sg_sg_attn     = self.module['plus_mab'](genes_encoded=sign_genes, paths_encoded=path_genes)
        else:
            sign_genes, sg_sg_attn     = self.module['gene_sab'](genes_encoded=sign_genes)
            return_batch['vecs/genes'] = sign_genes

        # Downstream Decoder
        decoded                    = self.module['diff_decoder'](genes_encoded=sign_genes)
        
        if self.label_type == 'binary' or self.label_type == 'binary_reverse':
            return_batch['task/dge_pred'] = self.sigmoid(decoded.squeeze(2))
            return_batch['task/dge_true'] = y_dge
        elif self.label_type == 'real' or self.label_type == 'real_reverse':
            return_batch['task/dge_pred'] = decoded.squeeze(2)
            return_batch['task/dge_true'] = y_dge
        else:
            raise ValueError('Unknown label_type: %s' % self.label_type)
        return_batch['meta/id'] = sig_ids

        return_batch['attn/sg_cp']      = sg_cp_attn
        return_batch['attn/tg_cp']      = tg_cp_attn
        return_batch['attn/tg_cp_true'] = y_pw  
        return_batch['attn/sg_tg']      = sg_tg_attn 
        return_batch['attn/sg_sg']      = sg_sg_attn
        return_batch['vecs/decoded']    = sign_genes

        return return_batch 