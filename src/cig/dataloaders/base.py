import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
import os
import sys
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
import time
import itertools
from collections import defaultdict
import math
from rdkit.Chem.Scaffolds.MurckoScaffold import *
from omegaconf import OmegaConf
from .molecules import Molecules
from collections import Counter
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.BRICS import BRICSDecompose
from multiprocessing.pool import Pool

###############################################
#                                             #
#              Dataset Base Class             #
#                                             #
###############################################


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def onek_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def mulk_encoding(x_set, full_set):
    y = np.zeros(len(full_set))
    for x in x_set: y[full_set.index(x)] = 1.0

    return y.reshape(1,-1)

def featurization(x):

    return x

def check_exists(path):
    return True if os.path.isfile(path) and os.path.getsize(path) > 0 else False

def add_index(input_array, ebd_size):
    add_idx, temp_arrays = 0, []
    for i in range(input_array.shape[0]):
        temp_array = input_array[i,:,:]
        masking_indices = temp_array.sum(1).nonzero()
        temp_array += add_idx
        temp_arrays.append(temp_array)
        add_idx = masking_indices[0].max()+1
    new_array = np.concatenate(temp_arrays, 0)

    return new_array.reshape(-1)

def get_substruct_smiles(smiles):
    tic = time.perf_counter()
    info = dict()
    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)
    fp = AllChem.GetMorganFingerprint(mol, 2, bitInfo=info)
    ec = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useChirality=False))
    X = []

    for k, v in info.items():
        for a, N in v:
            amap = dict()
            env  = Chem.FindAtomEnvironmentOfRadiusN(mol, N, a)
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            subsmiles = Chem.MolToSmiles(submol)
            if len(subsmiles) > 0: X.append(subsmiles)

    X = list(set(X))
    # print(f"Code Execution: {toc - tic:0.4f} seconds")
    return X

#
def load_cell_embeddings(conf):
    num_cells = 10 #156
    cell_input_dim = 12328 # L1000 genes
    cell_embedding_table = pd.read_csv(conf.path.dataset+f'/cell_embeddings_{conf.model_params.cell_embed}.csv', index_col=0)
    # cell_embed_table = nn.Embedding(num_cells, cell_input_dim).from_pretrained(G, freeze=False)
    cell_embeddings_index = cell_embedding_table.index
    cell_embeddings_numpy = cell_embedding_table.values
    cell_embeddings_numpy = [cell_embeddings_numpy[i,:] for i in range(cell_embeddings_numpy.shape[0])]

    return dict(zip(cell_embeddings_index,cell_embeddings_numpy))


class CigDatasetBase(Dataset):
    def __init__(self, conf):
        self.dataset_path = conf.path.dataset
        self.dataset_type = conf.experiment.dataset # only badger
        selection_method = conf.dataprep.selection # ciger
        self.cell2vec = load_cell_embeddings(conf)
        self.unique_meta  = dict()
        self.split_type = conf.dataprep.split_type
        self.folds = conf.experiment.folds
        self.model_name = conf.model_params.pred_model

        self.chemsig_dataframe = pd.read_csv(self.dataset_path+f'/chemical_signatures_{self.dataset_type}.csv', index_col='sig_id')
        self.drugcmp_dataframe = pd.read_csv(self.dataset_path+f'/drug_smiles_{self.dataset_type}.csv', index_col='pert_id')
        if selection_method:
            pert_ids = pd.read_csv(self.dataset_path+f'/pert_ids_selected_{selection_method}.csv').columns.values.tolist()
            self.chemsig_dataframe  = self.chemsig_dataframe[self.chemsig_dataframe['pert_id'].isin(pert_ids)]
        if conf.dataprep.filter_tas:
            self.chemsig_dataframe = self.chemsig_dataframe[self.chemsig_dataframe['tas_score']>conf.dataprep.filter_tas]
        if conf.dataprep.filter_qc:
            self.chemsig_dataframe = self.chemsig_dataframe[self.chemsig_dataframe['qc_pass']==1]

        if conf.dev_mode.debugging or conf.dev_mode.toy_test:
            self.chemsig_dataframe = self.chemsig_dataframe.iloc[:100, :]
            temp = self.drugcmp_dataframe
            self.drugcmp_dataframe = temp[temp.index.isin(self.chemsig_dataframe['pert_id'].unique().tolist())]

        print("Total Number of Original Data Instances:    ", self.chemsig_dataframe.shape[0])
        df = pd.read_csv(self.dataset_path+'/gene2index.csv')
        self.gene2idx = dict(zip(df['gene_name'],df['gene_id']))
        self.idx2gene = dict(zip(df['gene_id'],df['gene_name']))
        self.idx2gene[19563] = 'UNK'
        self.gene2idx['UNK'] = 19563

        self.unique_meta['cell_id']       = self.chemsig_dataframe['cell_id'].unique().tolist() + ['UNK']
        self.unique_meta['pert_id']       = self.chemsig_dataframe['pert_id'].unique().tolist() + ['UNK']
        # self.unique_meta['pert_type']  = self.chemsig_dataframe['pert_type'].unique().tolist() + ['UNK']
        self.unique_meta['pert_idose'] = self.chemsig_dataframe['pert_idose'].unique().tolist() + ['UNK']
        # self.unique_meta['tas_score'] = self.chemsig_dataframe['tas_score'].unique().tolist() + ['UNK']
        # self.unique_meta['qc_pass'] = self.chemsig_dataframe['qc_pass'].unique().tolist() + ['UNK']
        self.unique_meta['pert_itime']    = self.chemsig_dataframe['pert_itime'].unique().tolist() + ['UNK']
        self.unique_meta['pair_id']       = list(zip(self.chemsig_dataframe['pert_id'],self.chemsig_dataframe['cell_id']))
        self.chemsig_dataframe['pair_id'] = list(zip(self.chemsig_dataframe['pert_id'],self.chemsig_dataframe['cell_id']))

        # nongene_features = ['pert_id', 'pert_type', 'cell_id', 'pert_idose']
        nongene_features = ['pert_id', 'cell_id', 'pair_id', 'pert_idose', 'tas_score', 'qc_pass', 'pert_itime']
        self.chemsig_dataframe.drop_duplicates(nongene_features, inplace=True)
        self.genesig_dataframe = self.chemsig_dataframe.drop(nongene_features, axis=1)

        gene_features = self.genesig_dataframe.columns.tolist()
        self.genesig_dataframe = self.genesig_dataframe.reindex(sorted(gene_features), axis=1)
        self.gene_columns = self.genesig_dataframe.columns.values.tolist()

        self.pytr_instances, self.meta_instances = [], []
        self.kfold_splits = []

        print("Total Number of Filtered Data Instances:    ", self.chemsig_dataframe.shape[0])
        print("Unique Number of Cell IDs including UNK:    ", len(self.unique_meta['cell_id']))
        print("Unique Number of Pert IDs including UNK:    ", len(self.unique_meta['pert_id']))
        # print("Unique Number of Pert Types including UNK:  ", len(self.unique_meta['pert_type']))
        # print("Unique Number of Pert IDoses including UNK: ", len(self.unique_meta['pert_idose']))
        print("")
        # self.make_smiles_pharma_dict()
        # self.sigFactory = self.load_sigfactory()

    def __len__(self):

        return len(self.meta_instances)

    def __getitem__(self, idx):

        return self.pytr_instances[idx]


    def create_rdkit_mol(self, smiles):    
        mol = Chem.MolFromSmiles(smiles)
    
        return mol

    def create_pharma_fingerprint(self, mol):
        # mol = Chem.MolFromSmiles(smiles)
        fp  = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

        return np.array(fp)

    def create_ecfp_fingerprint(self, mol):
        try:
            ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=1024))
        except Exception as e:
            mol.UpdatePropertyCache()
            FastFindRings(mol)
            ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=1024))

        return ecfp

    def create_brics_fingerprint_set(self, mol):
        fragments = list(BRICSDecompose(mol))

        return [self.create_ecfp_fingerprint(Chem.MolFromSmiles(f)) for f in fragments]

    def create_brics_smiles_set(self, mol):
        fragments = list(BRICSDecompose(mol))

        return [f for f in fragments]

    def get_pharmacophores(self, smiles):
        print(smiles)
        mol = Chem.MolFromSmiles(smiles)
        fp  = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

        return np.array(fp.GetOnBits())

    def make_smiles_pharma_dict(self):
        self.smiles2pcp = dict()
        unique_smiles = self.drugcmp_dataframe['smiles'].unique().tolist()
        current_smiles2pcp = pickle.load(open('smiles2pcp.pickle', 'rb'))
        for smiles in unique_smiles:
            if smiles not in current_smiles2pcp.keys():
                print(smiles)
                self.smiles2pcp[smiles] = self.get_pharmacophores(smiles)
            else:
                self.smiles2pcp[smiles] = current_smiles2pcp[smiles]

    def make_splits(self):
        if self.model_name in ['deepce', 'ciger']:
            self.meta_instances = [(sig_id, pert_id, cell_id[0]) for (sig_id, pert_id, cell_id) in self.meta_instances]
        kf = KFold(n_splits=self.folds, shuffle=True)
        for (train_cell_ids, test_cell_ids), (train_pert_ids, test_pert_ids) in zip(kf.split(self.unique_meta['cell_id']), kf.split(self.unique_meta['pert_id'])):
            train_cell_ids, valid_cell_ids = train_test_split(train_cell_ids, test_size=0.25)
            train_cell_ids = np.array(self.unique_meta['cell_id'])[train_cell_ids].tolist()
            valid_cell_ids = np.array(self.unique_meta['cell_id'])[valid_cell_ids].tolist()
            test_cell_ids  = np.array(self.unique_meta['cell_id'])[test_cell_ids].tolist()
            assert len(set(train_cell_ids) & set(valid_cell_ids)) == 0
            assert len(set(valid_cell_ids) & set(test_cell_ids))  == 0
            assert len(set(train_cell_ids) & set(test_cell_ids))  == 0


            train_cell_indices = [idx for idx, meta in enumerate(self.meta_instances) if meta[2] in train_cell_ids]
            valid_cell_indices = [idx for idx, meta in enumerate(self.meta_instances) if meta[2] in valid_cell_ids]
            test_cell_indices  = [idx for idx, meta in enumerate(self.meta_instances) if meta[2] in  test_cell_ids]
            
            train_pert_ids, valid_pert_ids = train_test_split(train_pert_ids, test_size=0.25)
            train_pert_ids = np.array(self.unique_meta['pert_id'])[train_pert_ids].tolist()
            valid_pert_ids = np.array(self.unique_meta['pert_id'])[valid_pert_ids].tolist()
            test_pert_ids  = np.array(self.unique_meta['pert_id'])[test_pert_ids].tolist()
            assert len(set(train_pert_ids) & set(valid_pert_ids)) == 0
            assert len(set(valid_pert_ids) & set(test_pert_ids))  == 0
            assert len(set(train_pert_ids) & set(test_pert_ids))  == 0
            train_pert_indices = [idx for idx, meta in enumerate(self.meta_instances) if meta[1] in train_pert_ids]
            valid_pert_indices = [idx for idx, meta in enumerate(self.meta_instances) if meta[1] in valid_pert_ids]
            test_pert_indices  = [idx for idx, meta in enumerate(self.meta_instances) if meta[1] in test_pert_ids]
            
            
            train_pair_indices = [idx for idx, meta in enumerate(self.meta_instances) if (meta[1] in train_pert_ids) and (meta[2] in train_cell_ids)]
            valid_pair_indices = [idx for idx, meta in enumerate(self.meta_instances) if (meta[1] in valid_pert_ids) and (meta[2] in valid_cell_ids)]
            test_pair_indices  = [idx for idx, meta in enumerate(self.meta_instances) if (meta[1] in test_pert_ids) and (meta[2] in test_cell_ids)]

            if self.split_type == 'cell_id':
                train_indices, valid_indices, test_indices = train_cell_indices, valid_cell_indices, test_cell_indices
            elif self.split_type == 'pert_id':
                train_indices, valid_indices, test_indices = train_pert_indices, valid_pert_indices, test_pert_indices
            elif self.split_type == 'pair_id':
                train_indices, valid_indices, test_indices = train_pair_indices, valid_pair_indices, test_pair_indices
            else:
                raise
            
            assert len(set(train_indices) & set(valid_indices)) == 0
            assert len(set(valid_indices) & set(test_indices))  == 0
            assert len(set(train_indices) & set(test_indices))  == 0
            self.kfold_splits.append((train_indices, valid_indices, test_indices))

        print("Number of Training Instances:  ", len(train_indices))
        print("Number of Validation Instances:", len(valid_indices))
        print("Number of Test Instances:      ", len(test_indices))
        print("")

    def get_training_statistics(self, fold, topK=200):
        train_indices = self.kfold_splits[fold][0]
        train_sig_ids = [self.meta_instances[i][0] for i in train_indices]
        train_degs    = self.genesig_dataframe.loc[train_sig_ids,:]

        top50, bot50 = [], []
        for gene_df_row in train_degs.iterrows():
            top50.append(gene_df_row[1].sort_values(ascending=False).index[:topK].values.reshape(1,topK))
            bot50.append(gene_df_row[1].sort_values(ascending=True).index[:topK].values.reshape(1,topK))
        
        top50 = pd.DataFrame(np.vstack(top50))
        bot50 = pd.DataFrame(np.vstack(bot50))

        top50 = top50[[i for i in range(topK)]].apply(pd.Series.value_counts).sum(1)
        bot50 = bot50[[i for i in range(topK)]].apply(pd.Series.value_counts).sum(1)

        self.train_stats = {'top': top50, 'bot': bot50, 'N': len(train_indices)}


    def make_meta_vector(self, zip_meta_features):
        meta_vector = []
        # unk = np.ndarray(shape=(1, self.cell_emb_table.shape[1]))
        for v, m in zip_meta_features:
            # if m == 'cell_id':
            #     if v == 'UNK':
            #         cell_emb = unk
            #     else:
            #         cell_emb = self.cell_emb_table.loc[v].values.reshape(1,-1)
            #     meta_vector.append(cell_emb)
            # else:
            #     meta_vector.append(np.array(onek_encoding_unk(v, self.unique_meta[m])).reshape(1,-1))
            meta_vector.append(np.array(onek_encoding_unk(v, self.unique_meta[m])).reshape(1,-1))
        return np.hstack(meta_vector)


###############################################
#                                             #
#              Collate Functions              #
#                                             #
###############################################



def stack_and_pad(arr_list, max_length=None):
    M = max([x.shape[0] for x in arr_list]) if not max_length else max_length
    N = max([x.shape[1] for x in arr_list])
    T = np.zeros((len(arr_list), M, N))
    t = np.zeros((len(arr_list), M))
    s = np.zeros((len(arr_list), M, N))

    for i, arr in enumerate(arr_list):
        # sum of 16 interaction type, one is enough
        if len(arr.shape) > 2:
            arr = (arr.sum(axis=2) > 0.0).astype(float)
        T[i, 0:arr.shape[0], 0:arr.shape[1]] = arr
        t[i, 0:arr.shape[0]] = 1 if arr.sum() != 0.0 else 0
        s[i, 0:arr.shape[0], 0:arr.shape[1]] = 1 if arr.sum() != 0.0 else 0
    return T, t, s

def stack_and_pad_with(arr_list, max_length=None, padding_idx=0):
    M = max([x.shape[0] for x in arr_list]) if not max_length else max_length
    N = max([x.shape[1] for x in arr_list])
    # T = np.zeros((len(arr_list), M, N))
    T = np.full((len(arr_list), M, N), padding_idx)
    t = np.zeros((len(arr_list), M))
    s = np.zeros((len(arr_list), M, N))

    for i, arr in enumerate(arr_list):
        # sum of 16 interaction type, one is enough
        if len(arr.shape) > 2:
            arr = (arr.sum(axis=2) > 0.0).astype(float)
        T[i, 0:arr.shape[0], 0:arr.shape[1]] = arr
        t[i, 0:arr.shape[0]] = 1 if arr.sum() != 0.0 else 0
        s[i, 0:arr.shape[0], 0:arr.shape[1]] = 1 if arr.sum() != 0.0 else 0
    return T, t, s

def stack_and_pad_2d(arr_list, block='lower_left', max_length=None):
    max0 = max([a.shape[0] for a in arr_list]) if not max_length else max_length
    max1 = max([a.shape[1] for a in arr_list])
    list_shapes = [a.shape for a in arr_list]

    final_result = np.zeros((len(arr_list), max0, max1))
    final_masks_2d = np.zeros((len(arr_list), max0))
    final_masks_3d = np.zeros((len(arr_list), max0, max1))

    if block == 'upper_left':
        for i, shape in enumerate(list_shapes):
            # sum of 16 interaction type, one is enough
            if len(arr_list[i].shape) > 2:
                arr_list[i] = (arr_list[i].sum(axis=2) == True).astype(float)
            final_result[i, :shape[0], :shape[1]] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], :shape[1]] = 1
    elif block == 'lower_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, max1-shape[1]:] = 1
    elif block == 'lower_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, :shape[1]] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, :shape[1]] = 1
    elif block == 'upper_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], max1-shape[1]:] = 1
    else:
        raise

    return final_result, final_masks_2d, final_masks_3d

def stack_and_pad_3d(arr_list, block='lower_left'):
    max0 = max([a.shape[0] for a in arr_list])
    max1 = max([a.shape[1] for a in arr_list])
    max2 = max([a.shape[2] for a in arr_list])
    list_shapes = [a.shape for a in arr_list]

    final_result = np.zeros((len(arr_list), max0, max1, max2))
    final_masks_2d = np.zeros((len(arr_list), max0))
    final_masks_3d = np.zeros((len(arr_list), max0, max1))
    final_masks_4d = np.zeros((len(arr_list), max0, max1, max2))

    if block == 'upper_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], :shape[1], :shape[2]] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], :shape[1]] = 1
            final_masks_4d[i, :shape[0], :shape[1], :] = 1
    elif block == 'lower_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, max1-shape[1]:] = 1
            final_masks_4d[i, max0-shape[0]:, max1-shape[1]:, :] = 1
    elif block == 'lower_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, :shape[1]] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, :shape[1]] = 1
            final_masks_4d[i, max0-shape[0]:, :shape[1], :] = 1
    elif block == 'upper_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], max1-shape[1]:] = 1
            final_masks_4d[i, :shape[0], max1-shape[1]:, :] = 1
    else:
        raise

    return final_result, final_masks_2d, final_masks_3d, final_masks_4d

def ds_normalize(input_array):
    # Doubly Stochastic Normalization of Edges from CVPR 2019 Paper
    assert len(input_array.shape) == 3
    input_array = input_array / np.expand_dims(input_array.sum(1)+1e-8, axis=1)
    output_array = np.einsum('ijb,jkb->ikb', input_array,
                             input_array.transpose(1, 0, 2))
    output_array = output_array / (output_array.sum(0)+1e-8)

    return output_array