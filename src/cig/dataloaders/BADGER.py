from .base import *

def convert_list_smiles_to_molecules(list_smiles):
    molecules = Molecules(list_smiles)
    node_repr = torch.cuda.FloatTensor([node.data for node in molecules.get_node_list('atom')])
    edge_repr = torch.cuda.FloatTensor([node.data for node in molecules.get_node_list('bond')])

    return {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}

def convert_list_smiles_to_molecules_with_batch_index(list_list_smiles):
    batch_index_global = []
    idx = 0
    for list_smiles in list_list_smiles:
        for _ in list_smiles:
            batch_index_global.append(idx)
        idx += 1

    molecules = Molecules(sum(list_list_smiles, []))
    node_repr = torch.cuda.FloatTensor([node.data for node in molecules.get_node_list('atom')])
    edge_repr = torch.cuda.FloatTensor([node.data for node in molecules.get_node_list('bond')])

    return {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr, 
            'batch_index_global': torch.cuda.LongTensor(batch_index_global)}

class CigDatasetPreload(CigDatasetBase):
    def __init__(self, conf):
        super().__init__(conf)
        self.meta_features = []
        if conf.dataprep.use_pert_id:    self.meta_features.append('pert_id')
        if conf.dataprep.use_pert_type:  self.meta_features.append('pert_type')
        if conf.dataprep.use_cell_id:    self.meta_features.append('cell_id')
        if conf.dataprep.use_pert_idose: self.meta_features.append('pert_idose')

        self.dti_dataframe             = pd.read_csv(self.dataset_path+f'/drug_target_information_{conf.model_params.badger.pathmab.path_embed}.csv', 
                                                    index_col='pert_id', dtype=object)
        self.dti_dataframe['pathway']  = self.dti_dataframe['pathway'].fillna('UNKNOWN').astype(str)
        self.unique_pathways           = sorted(self.dti_dataframe['pathway'].unique().tolist())

        for idx, sig_id in tqdm(enumerate(self.chemsig_dataframe.index)):
            try:
                drug_id       = self.chemsig_dataframe.loc[sig_id, 'pert_id']
                cell_id       = self.chemsig_dataframe.loc[sig_id, 'cell_id']
                genes_chemsig = self.genesig_dataframe.loc[sig_id, :].to_numpy().reshape(1,-1)
                drug_smiles   = self.drugcmp_dataframe.loc[drug_id, 'smiles']

                if drug_id in self.dti_dataframe.index:
                    pathways  = self.dti_dataframe.loc[drug_id, :]
                    if not isinstance(pathways, pd.Series):
                        pathways = pathways['pathway'].unique().tolist()
                    else:
                        pathways  = [pathways['pathway']]
                    if len(pathways) > 1 and 'UNKNOWN' in pathways: pathways.remove('UNKNOWN')
                    pathways  = mulk_encoding(pathways, self.unique_pathways)
                else:
                    pathways  = np.zeros((1,len(self.unique_pathways)))

                # This will be deprecated sooner or later...
                meta_values   = [self.chemsig_dataframe.loc[sig_id, m] for m in self.meta_features]
                if len(meta_values) > 0:
                    meta_vector   = self.make_meta_vector(zip(meta_values, self.meta_features))
                    self.meta_dim = meta_vector.shape[1]
                else:
                    meta_vector   = np.zeros((1,1))
                    self.meta_dim = 0

                cell_vector = self.cell2vec[cell_id].reshape(1,-1)
                # drug_dosage = float(self.chemsig_dataframe.loc[sig_id, 'pert_idose'].split(' uM')[0])
                drug_frag_ecfps  = self.create_brics_fingerprint_set(self.create_rdkit_mol(drug_smiles))
                drug_frag_smiles = self.create_brics_smiles_set(self.create_rdkit_mol(drug_smiles)) 
                # pharma_indices = self.get_pharmacophores(drug_smiles)
                # pharma_indices = self.smiles2pcp[drug_smiles]
                pharma_indices = np.array([0,1,2,3])

                self.pytr_instances.append((genes_chemsig, drug_smiles, pharma_indices, drug_frag_ecfps, 
                                            drug_frag_smiles, pathways[0,:-1], cell_vector, sig_id))
                self.meta_instances.append((sig_id, drug_id, cell_id, meta_values))
            except Exception as e:
                print(e)
                pass

        assert len(self.pytr_instances) == len(self.meta_instances)
        self.data_indices = [*range(len(self.pytr_instances))]

        print("")
        print("Total Number of Data Instances:", len(self.data_indices))
        print("Use Meta Feature [pert_id]    :", conf.dataprep.use_pert_id)
        print("Use Meta Feature [pert_type]  :", conf.dataprep.use_pert_type)
        print("Use Meta Feature [cell_id]    :", conf.dataprep.use_cell_id)
        print("Use Meta Feature [pert_idose] :", conf.dataprep.use_pert_idose)
        print("")

class CigDatasetAutoload(CigDatasetBase):
    def __init__(self, conf):
        super().__init__(conf)
        self.meta_features = []
        if conf.dataprep.use_pert_id:    self.meta_features.append('pert_id')
        if conf.dataprep.use_pert_type:  self.meta_features.append('pert_type')
        if conf.dataprep.use_cell_id:    self.meta_features.append('cell_id')
        if conf.dataprep.use_pert_idose: self.meta_features.append('pert_idose')

        self.dti_dataframe             = pd.read_csv(self.dataset_path+'/drug_target_information_msig.csv', 
                                                     index_col='pert_id', dtype=object)
        self.dti_dataframe['pathway']  = self.dti_dataframe['pathway'].fillna('UNKNOWN').astype(str)
        self.unique_pathways           = sorted(self.dti_dataframe['pathway'].unique().tolist())
        # [target-genes x (drug + no effect)] -> need two indices

        for idx, sig_id in tqdm(enumerate(self.chemsig_dataframe.index)):
            try:
                drug_id       = self.chemsig_dataframe.loc[sig_id, 'pert_id']
                cell_id       = self.chemsig_dataframe.loc[sig_id, 'cell_id']
                genes_chemsig = self.genesig_dataframe.loc[sig_id, :].to_numpy().reshape(1,-1)
                drug_smiles   = self.drugcmp_dataframe.loc[drug_id, 'smiles']
                meta_values   = [self.chemsig_dataframe.loc[sig_id, m] for m in self.meta_features]

                self.meta_instances.append((sig_id, drug_id, cell_id, meta_values))

            except Exception as e:
                print(e)
                pass

        # assert len(self.pytr_instances) == len(self.meta_instances)
        self.data_indices = [*range(len(self.meta_instances))]

        print("")
        print("Total Number of Data Instances:", len(self.data_indices))
        print("Use Meta Feature [pert_id]    :", conf.dataprep.use_pert_id)
        print("Use Meta Feature [pert_type]  :", conf.dataprep.use_pert_type)
        print("Use Meta Feature [cell_id]    :", conf.dataprep.use_cell_id)
        print("Use Meta Feature [pert_idose] :", conf.dataprep.use_pert_idose)
        print("")
        # print(len(self.data_indices)) # 4165 -> 4123 -> 3294
        # import pdb; pdb.set_trace()

    def __getitem__(self, idx):
        sig_id, drug_id, cell_id, meta_values = self.meta_instances[idx]
        drug_smiles   = self.drugcmp_dataframe.loc[drug_id, 'smiles']
        genes_chemsig = self.genesig_dataframe.loc[sig_id, :].to_numpy().reshape(1,-1)

        if drug_id in self.dti_dataframe.index:
            pathways  = self.dti_dataframe.loc[drug_id, :]
            if not isinstance(pathways, pd.Series):
                pathways = pathways['pathway'].unique().tolist()
            else:
                pathways  = [pathways['pathway']]
            if len(pathways) > 1 and 'UNKNOWN' in pathways: pathways.remove('UNKNOWN')
            pathways  = mulk_encoding(pathways, self.unique_pathways)
        else:
            pathways  = np.zeros((1,len(self.unique_pathways)))

        if len(meta_values) > 0:
            meta_vector   = self.make_meta_vector(zip(meta_values, self.meta_features))
            self.meta_dim = meta_vector.shape[1]
        else:
            meta_vector   = np.zeros((1,1))
            self.meta_dim = 0

        cell_vector = self.cell2vec[cell_id].reshape(1,-1)
        drug_frag_ecfps = self.create_brics_fingerprint_set(self.create_rdkit_mol(drug_smiles))
        drug_frag_smiles = self.create_brics_smiles_set(self.create_rdkit_mol(drug_smiles)) 
        pharma_indices = np.array([0,1,2,3])
        
        pytr_instances =(genes_chemsig, drug_smiles, pharma_indices, drug_frag_ecfps, 
                         drug_frag_smiles, pathways[0,:-1], cell_vector, sig_id)
        
        return pytr_instances

    # to-do
    def make_inference_data(self, input_smiles, input_cellid, input_pertid):
        list_tensors = []

        x = convert_list_smiles_to_molecules([input_smiles])
        list_tensors.append(x)

        x = torch.cuda.FloatTensor(self.cell2vec[input_cellid].reshape(1,-1))
        list_tensors.append(x)
        list_tensors.append([input_smiles])

        x = self.create_brics_fingerprint_set(self.create_rdkit_mol(input_smiles))
        x, m, _ = stack_and_pad([np.vstack(x)])
        x = torch.cuda.FloatTensor(x)
        m = torch.cuda.FloatTensor(m)
        list_tensors.append(x)
        list_tensors.append(m)

        list_tensors.append(None)
        list_tensors.append(None)

        y = torch.cuda.FloatTensor(np.zeros((1,978)))
        list_tensors.append(y)
        y = torch.cuda.LongTensor(np.zeros((1,len(self.unique_pathways))))
        list_tensors.append(y)
        list_tensors.append(input_pertid)

        return list_tensors

def collate_fn(batch):
    list_tensors = []
    list_genes_chemsig = [x[0] for x in batch]
    list_drug_smiles   = [x[1] for x in batch]
    list_pcp_indices   = [x[2].reshape(1,-1) for x in batch]
    #
    list_frag_ecfps    = [np.vstack(x[3]) for x in batch]
    list_frag_smiles   = [x[4] for x in batch]
    #
    list_target_genes  = [x[5] for x in batch]
    list_cell_vectors  = [x[6] for x in batch]
    list_sig_id        = [x[-1] for x in batch]

    # Graph-based Drug Compounds
    x = convert_list_smiles_to_molecules(list_drug_smiles)
    list_tensors.append(x)

    # Cell Representation
    x = torch.cuda.FloatTensor(np.vstack(list_cell_vectors))
    list_tensors.append(x)
    list_tensors.append(list_drug_smiles)

    # Set of Fragments (Drug Compounds)
    x, m, _ = stack_and_pad(list_frag_ecfps)
    x = torch.cuda.FloatTensor(x)
    m = torch.cuda.FloatTensor(m)
    list_tensors.append(x)
    list_tensors.append(m)

    # NEW!!! List of Graph-based Fragments (Drug Comopunds)
    x = convert_list_smiles_to_molecules_with_batch_index(list_frag_smiles)
    list_tensors.append(x)

    # Pharmaphocores (Drug Compounds)
    x, _, m = stack_and_pad_with(list_pcp_indices, padding_idx=39972)
    x = torch.cuda.LongTensor(x.squeeze(1))
    m = torch.cuda.FloatTensor(m.squeeze(1))
    list_tensors.append(x)
    list_tensors.append(m)

    y = torch.cuda.FloatTensor(np.vstack(list_genes_chemsig))
    list_tensors.append(y)
    y = torch.cuda.LongTensor(np.vstack(list_target_genes))
    list_tensors.append(y)
    list_tensors.append(list_sig_id)

    return list_tensors

if __name__ == '__main__':
    args = OmegaConf.load("debug.yaml")
    print(args.path.dataset)

    dataset = CigDataset(args)
    dataset.make_splits_pert_id()
    import pdb; pdb.set_trace()

    from torch.utils.data import DataLoader, SubsetRandomSampler

    sampler    = SubsetRandomSampler([*range(100)])
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, collate_fn=collate_fn)

    for batch in dataloader:

        break

    exit()