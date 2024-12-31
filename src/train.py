import random
import argparse
import tarfile
import wandb
import pickle
import setproctitle
import os 
import os.path
import json
import numpy as np
from omegaconf import OmegaConf
from random import sample

from os import kill
from os import getpid
from signal import SIGKILL
import time

from trainer import *

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

num_cpus = os.cpu_count()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

ngpus_per_node = torch.cuda.device_count()

parser = argparse.ArgumentParser()
parser.add_argument('--session_name',    '-sn', default='defaultsession', type=str)


args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_wandb(conf):
    wandb_init = dict()
    wandb_init['project'] = conf.wandb.project_name
    wandb_init['group'] = conf.wandb.session_name
    if not conf.experiment.test_mode:
        wandb_init['name'] = f'training_{conf.experiment.dataset}' 
    else:
        wandb_init['name'] = f'testing_{conf.experiment.dataset}' 
    wandb_init['notes'] = conf.wandb.session_name
    os.environ['WANDB_START_METHOD'] = 'thread'

    return wandb_init

def reset_wandb_env():
    exclude = {'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_API_KEY',}
    for k, v in os.environ.items():
        if k.startswith('WANDB_') and k not in exclude:
            del os.environ[k]

def load_dataset_collate_model(conf):
    
    if conf.model_params.pred_model == 'ciger':
        from cig.dataloaders.CIGER import CigDataset
        from cig.dataloaders.CIGER import collate_fn
        from cig.models.CIGER import Net

    if conf.model_params.pred_model == 'deepce':
        from cig.dataloaders.DeepCE import CigDataset
        from cig.dataloaders.DeepCE import collate_fn
        from cig.models.DeepCE import Net

    if conf.model_params.pred_model == 'badger':
        from cig.dataloaders.BADGER import CigDatasetPreload as CigDataset
        from cig.dataloaders.BADGER import collate_fn
        from cig.models.BADGER import Net   
    
    if conf.model_params.pred_model == 'mlp':
        from cig.dataloaders.BADGER import CigDatasetPreload as CigDataset
        from cig.dataloaders.BADGER import collate_fn
        from cig.models.MLP import Net  
    
    if conf.model_params.pred_model == 'knn':
        parent_path = os.getcwd()
        embed_ckpt_path = os.path.join(parent_path, 'checkpoint', f'Baseline_mlp_{conf.wandb.group_name}_fold_{conf.experiment.fold_num}')
        
        if not os.path.exists(embed_ckpt_path):
            print("\nFor the fusion layer that combines drug and cell embeddings, the MLP should be trained first. Please try again after the MLP training is completed.\n")
            def modify_config(config_path, output_path=None):
                config = OmegaConf.load(config_path)
                config.wandb.session_name = 'mlp'
                config.model_params.pred_model = 'mlp'
                save_path = output_path if output_path else config_path
                with open(save_path, 'w') as f:
                    OmegaConf.save(config=config, f=f)
            modify_config(os.path.join(parent_path, 'sessions/knn.yaml'), os.path.join(parent_path, 'sessions/mlp.yaml'))
            if conf.experiment.fold_num == 0:
                print(f"Run: python {os.path.join(parent_path, 'run.py')} -sn mlp -sf {conf.experiment.fold_num} -ef {conf.experiment.fold_num+1}")
            else:
                print(f"Run: python {os.path.join(parent_path, 'run.py')} -sn mlp -sf {conf.experiment.fold_num} -ef {conf.experiment.fold_num}")
            import sys; sys.exit()
        
        from cig.dataloaders.BADGER import CigDatasetPreload as CigDataset
        from cig.dataloaders.BADGER import collate_fn
        from cig.models.kNN import Net      

    
    
    if conf.dataprep.baseline_default:
        DATASET_CACHE = f'{conf.path.dataset}/{conf.model_params.pred_model}_{conf.experiment.dataset}_baseline_default.dataset'
    else:
        DATASET_CACHE = f'{conf.path.dataset}/{conf.model_params.pred_model}_{conf.dataprep.split_type}_{conf.experiment.dataset}'
        DATASET_CACHE += f'_labeltype_{conf.dataprep.label_type}'
        DATASET_CACHE += f'_splittype_{conf.dataprep.split_type}'
        DATASET_CACHE += f'_drug_emb_{conf.model_params.drug_embed}'
        DATASET_CACHE += f'_cell_emb_{conf.model_params.cell_embed}'
        DATASET_CACHE += f'_gene_emb_{conf.model_params.gene_embed}'
        DATASET_CACHE += f'_path_emb_{conf.model_params.badger.pathmab.path_embed}'
        DATASET_CACHE += f'_randomseed_{conf.experiment.random_seed}'
        DATASET_CACHE += '.dataset'

    if not (conf.dev_mode.debugging or conf.dev_mode.toy_test):
        if not os.path.isfile(DATASET_CACHE):
            dataset = CigDataset(conf)
            conf.model_params.meta_dimension = dataset.meta_dim
            dataset.make_splits()
            if (not conf.dev_mode.debugging) and (not conf.dev_mode.toy_test):
                pickle.dump(dataset, open(DATASET_CACHE, 'wb'))
                print("Saved Dataset Cache to", DATASET_CACHE)
        else:        
            print("Loaded Dataset Cache from", DATASET_CACHE)
            dataset = pickle.load(open(DATASET_CACHE, 'rb'))
            conf.model_params.meta_dimension = dataset.meta_dim
    else:
        dataset = CigDataset(conf)
        conf.model_params.meta_dimension = dataset.meta_dim
        dataset.make_splits()

    net = Net(conf)

    return dataset, collate_fn, net

def load_pretrained_model(args, net):
    if args.finetune_name != 'None':
        session_name = f'{args.project_name}_{args.session_name}_{args.group_name}'
        original_path = os.path.join(args.checkpoint_path, session_name)
        CHECKPOINT_PATH = f'{original_path}_fold_{args.fold_num}_mea_{args.ba_measure}' 
        model_config    = f'{CHECKPOINT_PATH}/model_config.pkl'
        best_model      = f'{CHECKPOINT_PATH}/best_epoch.mdl'
        last_model      = f'{CHECKPOINT_PATH}/last_epoch.mdl'
        assert os.path.isfile(model_config), f"{model_config} DOES NOT EXIST!"
        assert os.path.isfile(best_model),   f"{best_model} DOES NOT EXIST!"

        print("Loaded Pretrained Model from ", CHECKPOINT_PATH)
        args.checkpoint_path = args.checkpoint_path[:-1] + '_' + args.finetune_name + '/'
        print("ADJUSTING HYPERPARMETERS RELATED TO BADGER")
        torch.cuda.empty_cache()
    return args, net

def run_debug_mode(conf, dataset, collate_fn, net):
    setup_seed(conf.experiment.random_seed)
    setproctitle.setproctitle(f'{conf.model_params.pred_model}_fold_{conf.experiment.fold_num}_debug')
    session_name = 'testproject_testgroup_testsession'
    conf.path.checkpoint = os.path.join(conf.path.checkpoint, session_name)

    # Distributed DataLoaders
    ddp_batch_size = conf.train_params.batch_size//ngpus_per_node
    samplers = [SubsetRandomSampler(x) for x in dataset.kfold_splits[conf.experiment.fold_num-1]]
    train      = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[0], collate_fn=collate_fn)
    valid      = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[1], collate_fn=collate_fn)
    test       = DataLoader(dataset, batch_size=ddp_batch_size, sampler=samplers[2], collate_fn=collate_fn)

    rank = 0
    trainer = Trainer(conf, rank, wandb_run=None)
    dataset.get_training_statistics(conf.experiment.fold_num-1)
    trainer.loss_function.train_stats = dataset.train_stats

    # train, valid, test and save the model
    trained_model, train_loss, valid_loss = trainer.train_valid(net, train, None, valid)
    if rank == 0: print('Finish Debugging Mode')

def run_single_fold(rank, ngpus_per_node, conf, dataset, collate_fn, net):
    setup_seed(conf.experiment.random_seed)
    pid = getpid()
    print(f'Running Process with PID: {pid}')

    setproctitle.setproctitle(f'{conf.model_params.pred_model}_fold_{conf.experiment.fold_num}_gpu_{rank}')
    session_name = f'{conf.wandb.project_name}_{conf.wandb.session_name}_{conf.wandb.group_name}'
    conf.path.checkpoint = os.path.join(conf.path.checkpoint, session_name)

    # WANDB setup /// args
    if rank == 0:
        reset_wandb_env()
        wandb_init = setup_wandb(conf)
        wandb_init['name'] += f'_{conf.model_params.pred_model}_fold_{conf.experiment.fold_num}'
        run = wandb.init(**wandb_init, settings=wandb.Settings(start_method='thread'))
        run.define_metric('train/step'); run.define_metric('train/*', step_metric='train/step')
        run.define_metric('valid/step'); run.define_metric('valid/*', step_metric='valid/step')
        run.define_metric('test/step'); run.define_metric('test/*', step_metric='test/step')
        run.watch(net, log="gradients", log_freq=500)
    else: run = None

    # initailize pytorch distributed
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', 
            init_method=f'tcp://localhost:{conf.ddp.port}',
            rank=rank, world_size=ngpus_per_node)

    trainer = Trainer(conf, rank, run)
    trainer = net.set_default_hp(trainer) if conf.train_params.baseline_default else trainer
    dataset.get_training_statistics(conf.experiment.fold_num-1)
    trainer.loss_function.train_stats = dataset.train_stats

    ddp_batch_size = trainer.batch_size//ngpus_per_node
    if rank == 0:
        print('Batch size', trainer.batch_size)
        print('Distributed batch size', ddp_batch_size)
        print("")

    dataloaders, train_sampler = [], None

    for idx, indices in enumerate(dataset.kfold_splits[conf.experiment.fold_num-1]):
        if idx == 0: 
            import pickle; pickle.dump({idx:indices}, open('dataset_fold_indices.pkl', 'wb'))
            if conf.experiment.train_subsample:
                indices = sample(indices, len(indices)//100)
                print("Subset Size: ", len(indices))
        sampler = DistributedSampler(Subset(dataset,indices), shuffle=True)
        loader  = DataLoader(Subset(dataset,indices), batch_size=ddp_batch_size, 
                                                      sampler=sampler, 
                                                      collate_fn=collate_fn)
        dataloaders.append(loader)
        if idx == 0: train_sampler = sampler
    del dataset
    train, valid, test = dataloaders
    
    
    if conf.dev_mode.toy_test: 
        print("Toy Test Mode"); trainer.num_epochs = 2
    if rank == 0 and not conf.experiment.test_mode:
        pickle.dump(conf, open(f'{trainer.checkpoint_path}/model_config.pkl', 'wb'))

    if not conf.experiment.test_mode:    
        net = trainer.train_valid(net, train, train_sampler, valid)
    else:
        if conf.model_params.pred_model == 'knn':
            pass
        else:
            BEST_METRIC = conf.train_params.best_criteria_metric.replace('/', '_')
            print("TEST MODE: ", f'{trainer.checkpoint_path}/best_epoch_{BEST_METRIC}.mdl')
            chkpt = torch.load(f'{trainer.checkpoint_path}/best_epoch_{BEST_METRIC}.mdl', map_location=f"cuda:{rank}")
            new_state_dict = {}
            for key in chkpt.keys():
                if 'module' not in key:
                    new_key = 'module.' + key  # 'module.' 제거
                else:
                    new_key = key
                new_state_dict[new_key] = chkpt[key]

            net.load_state_dict(new_state_dict)
        net = trainer.test(net, test)
    
    
    
        
    print(net)
    print(f'FINISHED: {conf.model_params.pred_model}_fold_{conf.experiment.fold_num}_gpu_{rank}')
    if rank == 0:
        omegaconf_path = f'{trainer.checkpoint_path}/config.yaml'
        OmegaConf.save(config=conf, f=open(omegaconf_path, 'w'))
    time.sleep(120)


def run_single_fold_multi_gpu(ngpus_per_node, conf, dataset, collate_fn, net):
    torch.multiprocessing.spawn(run_single_fold, 
                                args=(ngpus_per_node, conf, dataset, collate_fn, net), 
                                nprocs=ngpus_per_node, 
                                join=True)
    print("Finished Multiprocessing")


def setup_gpu(conf):
    if torch.cuda.is_available():
        gpu_available = os.environ['CUDA_VISIBLE_DEVICES']
        device = f'cuda: {gpu_available}'
    else:
        device = 'cpu'

    print(f'The current world has {ngpus_per_node} GPUs')
    print(f'Current device is {device}\n')
    
    return conf

if __name__ == "__main__":
    print(args)
    conf = OmegaConf.load(f'sessions/{args.session_name}.yaml')
    setup_seed(conf.experiment.random_seed)
    wandb_init = setup_wandb(conf)
    dataset, collate_fn, net = load_dataset_collate_model(conf)
    conf.gene_columns = dataset.gene_columns

    if conf.dev_mode.debugging:
        run_debug_mode(conf, dataset, collate_fn, net)
    else:
        conf = setup_gpu(conf)
        run_single_fold_multi_gpu(ngpus_per_node, conf, dataset, collate_fn, net)