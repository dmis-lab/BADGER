from .base import *

class CigDataset(CigDatasetBase):
    def __init__(self, conf):
        super().__init__(conf)
        meta_features = []
        if conf.dataprep.baseline_default:
            print("Setting Default Dataset Hyperparameters")
            conf.dataprep.use_pert_id = False
            conf.dataprep.use_pert_type = False
            conf.dataprep.use_cell_id = True
            conf.dataprep.use_pert_idose = False
        if conf.dataprep.use_pert_id:    meta_features.append('pert_id')
        if conf.dataprep.use_pert_type:  meta_features.append('pert_type')
        if conf.dataprep.use_cell_id:    meta_features.append('cell_id')
        if conf.dataprep.use_pert_idose: meta_features.append('pert_idose')

        for idx, sig_id in enumerate(self.chemsig_dataframe.index):
            try:
                drug_id       = self.chemsig_dataframe.loc[sig_id, 'pert_id']
                genes_chemsig = self.genesig_dataframe.loc[sig_id, :].to_numpy().reshape(1,-1)
                drug_smiles   = self.drugcmp_dataframe.loc[drug_id, 'smiles']

                meta_values   = [self.chemsig_dataframe.loc[sig_id, m] for m in meta_features]
                meta_vector   = self.make_meta_vector(zip(meta_values, meta_features))
                self.meta_dim = meta_vector.shape[1]

                self.pytr_instances.append((genes_chemsig, drug_smiles, meta_vector, sig_id))
                self.meta_instances.append((sig_id, drug_id, meta_values))
            except Exception as e:
                print(e)
                pass

        assert len(self.pytr_instances) == len(self.meta_instances)
        self.data_indices = [*range(len(self.pytr_instances))]

        print("Total Number of Data Instances:", len(self.data_indices))
        print("Use Meta Feature [pert_id]    :", conf.dataprep.use_pert_id)
        print("Use Meta Feature [pert_type]  :", conf.dataprep.use_pert_type)
        print("Use Meta Feature [cell_id]    :", conf.dataprep.use_cell_id)
        print("Use Meta Feature [pert_idose] :", conf.dataprep.use_pert_idose)
        print("")
        # print(len(self.data_indices)) # 4165 -> 4123 -> 3294
        # import pdb; pdb.set_trace()

def convert_list_smiles_to_molecules(list_smiles):
    molecules = Molecules(list_smiles)
    node_repr = torch.cuda.FloatTensor([node.data for node in molecules.get_node_list('atom')])
    edge_repr = torch.cuda.FloatTensor([node.data for node in molecules.get_node_list('bond')])

    return {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}

def collate_fn(batch):
    list_tensors = []

    list_gene_indices  = [np.arange(978, dtype=int) for _ in batch]
    list_genes_chemsig = [x[0] for x in batch]
    list_drug_smiles   = [x[1] for x in batch]
    list_meta_vectors  = [x[2] for x in batch]
    list_sig_id        = [x[3] for x in batch]

    x = torch.cuda.LongTensor(np.vstack(list_gene_indices))
    list_tensors.append(x)
    x = convert_list_smiles_to_molecules(list_drug_smiles)
    list_tensors.append(x)
    x = torch.cuda.FloatTensor(np.vstack(list_meta_vectors))
    list_tensors.append(x)
    #
    list_tensors.append(list_drug_smiles)
    #
    y = torch.cuda.FloatTensor(np.vstack(list_genes_chemsig))
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