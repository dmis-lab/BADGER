from typing import Any, Callable, List, Tuple, Union

from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from utils import *
from datetime import datetime


from time import sleep
import pickle

import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score as aup
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from lifelines.utils import concordance_index as c_index
import pandas as pd

torch.autograd.set_detect_anomaly(True)

SCHEDULER_FIRSTSTEP = [None]
SCHEDULER_BATCHWISE = [None]
SCHEDULER_EPOCHWISE = [None]

class Trainer:
    def __init__(self, conf, rank, wandb_run=None, ddp_mode=True):
        self.debug_mode    = conf.dev_mode.debugging
        self.ddp_mode      = ddp_mode
        self.rank          = rank
        self.wandb         = wandb_run 
        self.gene_columns  = conf.gene_columns

        self.pred_model    = conf.model_params.pred_model
        self.label_type    = conf.dataprep.label_type
        self.batch_size    = conf.train_params.batch_size
        self.num_epochs    = conf.train_params.num_epochs
        self.learning_rate = conf.train_params.learning_rate
        self.weight_decay  = conf.train_params.weight_decay
        self.loss_function = load_loss_function(conf, rank)
        self.main_coef     = conf.train_params.main_coef
        self.pathmab_coef  = conf.train_params.badger.pathmab_coef
        self.genesab_coef  = conf.train_params.badger.genesab_coef
        self.genespo_coef  = conf.train_params.badger.gene_spreadout_coef
        self.sign          = -1 if 'reverse' in self.label_type else 1

        self.num_patience  = conf.train_params.early_patience
        self.best_criteria = conf.train_params.best_criteria_metric

        # self.checkpoint_path = f'{conf.path.checkpoint}_fold_{conf.experiment.fold_num}_seed_{conf.experiment.random_seed}' 
        self.checkpoint_path = f'{conf.path.checkpoint}_fold_{conf.experiment.fold_num}'
        if not conf.experiment.test_mode:
            os.makedirs(self.checkpoint_path, exist_ok=True)

        self.save_name = conf.wandb.session_name
        self.lookup_values = dict({'true_dgeR': [], 'pred_dgeR': [], 
                                   'true_dgeB': [], 'pred_dgeB': [],
                                   'true_dpwB': [], 'pred_dpwB': [],
                                   'meta_id': []}) 

        self.best_valid_metric = {
            'dge/pos/hitratio/10':   -np.Inf,
            'dge/pos/hitratio/50':   -np.Inf,
            'dge/pos/hitratio/100':  -np.Inf,
            'dge/pos/hitratio/200':  -np.Inf,
            'dge/pos/jaccard/10':    -np.Inf,
            'dge/pos/jaccard/50':    -np.Inf,
            'dge/pos/jaccard/100':   -np.Inf,
            'dge/pos/jaccard/200':   -np.Inf,
            'dge/pos/ndcg':          -np.Inf,

            'dge/neg/hitratio/10':   -np.Inf,
            'dge/neg/hitratio/50':   -np.Inf,
            'dge/neg/hitratio/100':  -np.Inf,
            'dge/neg/hitratio/200':  -np.Inf,
            'dge/neg/jaccard/10':    -np.Inf,
            'dge/neg/jaccard/50':    -np.Inf,
            'dge/neg/jaccard/100':   -np.Inf,
            'dge/neg/jaccard/200':   -np.Inf,
            'dge/neg/ndcg':          -np.Inf,

            'dge/abs/hitratio/10':   -np.Inf,
            'dge/abs/hitratio/50':   -np.Inf,
            'dge/abs/hitratio/100':  -np.Inf,
            'dge/abs/hitratio/200':  -np.Inf,
            'dge/abs/jaccard/10':    -np.Inf,
            'dge/abs/jaccard/50':    -np.Inf,
            'dge/abs/jaccard/100':   -np.Inf,
            'dge/abs/jaccard/200':   -np.Inf,
            'dge/abs/ndcg':          -np.Inf,

            'dge/rmse':               np.Inf}

        self.current_valid_metric = {
            'dge/pos/hitratio/10':   -np.Inf,
            'dge/pos/hitratio/50':   -np.Inf,
            'dge/pos/hitratio/100':  -np.Inf,
            'dge/pos/hitratio/200':  -np.Inf,
            'dge/pos/jaccard/10':    -np.Inf,
            'dge/pos/jaccard/50':    -np.Inf,
            'dge/pos/jaccard/100':   -np.Inf,
            'dge/pos/jaccard/200':   -np.Inf,
            'dge/pos/ndcg':          -np.Inf,

            'dge/neg/hitratio/10':   -np.Inf,
            'dge/neg/hitratio/50':   -np.Inf,
            'dge/neg/hitratio/100':  -np.Inf,
            'dge/neg/hitratio/200':  -np.Inf,
            'dge/neg/jaccard/10':    -np.Inf,
            'dge/neg/jaccard/50':    -np.Inf,
            'dge/neg/jaccard/100':   -np.Inf,
            'dge/neg/jaccard/200':   -np.Inf,
            'dge/neg/ndcg':          -np.Inf,

            'dge/abs/hitratio/10':   -np.Inf,
            'dge/abs/hitratio/50':   -np.Inf,
            'dge/abs/hitratio/100':  -np.Inf,
            'dge/abs/hitratio/200':  -np.Inf,
            'dge/abs/jaccard/10':    -np.Inf,
            'dge/abs/jaccard/50':    -np.Inf,
            'dge/abs/jaccard/100':   -np.Inf,
            'dge/abs/jaccard/200':   -np.Inf,
            'dge/abs/ndcg':          -np.Inf,

            'dge/rmse':               np.Inf}
        
        assert self.best_criteria in self.best_valid_metric.keys()

        if self.rank == 0:
            print("Prediction Model:  ", self.pred_model)
            print("Label Type:        ", self.label_type)
            print("# of Epochs:       ", self.num_epochs)
            print("Learning Rate:     ", self.learning_rate)
            print("Weight Decay:      ", self.weight_decay)
            print("Loss Function:     ", self.loss_function)
            print("")

    def print0(self, text: str):
        if self.rank == 0:
            print(text)

    def calculate_losses(self, batch):
        total_loss = []
        sign = -1 if 'reverse' in self.label_type else 1
        masked_nll = Masked_NLLLoss(self.rank, [1., 10.])
        l1_loss    = AttentionL1Loss(self.rank)
        spreadout_loss = SpreadoutLoss(self.rank, 10.)

        # self.print0(batch['task/dge_pred'][:10,:10])
        # self.print0(batch['task/dge_true'][:10,:10])

        if 'task/dge_true' in batch.keys():
            total_loss.append(self.loss_function(batch['task/dge_pred'], 
                                            sign*batch['task/dge_true']) * self.main_coef)

        if 'task/pw_drug_true' in batch.keys():
            total_loss.append(masked_nll(batch['task/pw_drug_pred'],
                                         batch['task/pw_drug_true'],
                                         batch['task/pw_drug_mask']) * self.pathmab_coef)

        if 'task/sg_attn_true' in batch.keys():
            total_loss.append(l1_loss(batch['task/sg_attn_pred'],
                                      batch['task/sg_attn_true']) * self.genesab_coef)

        if 'task/ppge_true' in batch.keys():
            raise

        if 'dump/chemberta' in batch.keys():  
            total_loss.append(batch['dump/chemberta'].sum()*0.0)

        if 'regu/sg_spreadout' in batch.keys():
            total_loss.append(spreadout_loss(batch['regu/sg_spreadout']) * self.genespo_coef)

        return total_loss 

    def check_wrong_losses(self, train_loss, valid_loss):
        if not np.isfinite(train_loss): print("ABNORMAL TRAIN LOSS"); return True
        if not np.isfinite(valid_loss): print("ABNORMAL VALID LOSS"); return True
        if train_loss < 0: print("NEGATIVE TRAIN LOSS"); return True
        if valid_loss < 0: print("NEGATIVE VALID LOSS"); return True
        return False

    def check_valid_progress(self):
        sub_measures = ['rmse']
        if self.best_criteria in sub_measures:
            return self.best_valid_metric[self.best_criteria] > self.current_valid_metric[self.best_criteria]
        else:
            return self.best_valid_metric[self.best_criteria] < self.current_valid_metric[self.best_criteria]

    def reset_lookup_values(self):
        self.lookup_values = dict({'true_dgeR': [], 'pred_dgeR': [], 
                                   'true_dgeB': [], 'pred_dgeB': [],
                                   'true_dpwB': [], 'pred_dpwB': [],
                                   'meta_id': []})

    def store_lookup_values(self, batch):
        self.lookup_values['meta_id'].extend(batch['meta/id'])

        if 'real' in self.label_type:
            self.lookup_values['true_dgeR'].extend(numpify(batch['task/dge_true'])*self.sign)
            self.lookup_values['pred_dgeR'].extend(numpify(batch['task/dge_pred']))
        
        if 'binary' in self.label_type:
            raise

        if 'task/pw_drug_true' in batch.keys():
            for i in range(batch['task/pw_drug_true'].size(0)):
                if batch['task/pw_drug_mask'][i,:].sum() != 0:
                    y      = batch['task/pw_drug_true'][i,:].view(-1)
                    yhat   = batch['task/pw_drug_pred'][i,:,1].view(-1)
                    m      = batch['task/pw_drug_mask'][i,:].view(-1)
                    labels = numpify(y[m>0.])
                    logits = numpify(yhat[m>0.])

                    self.lookup_values['true_dpwB'].append(labels)
                    self.lookup_values['pred_dpwB'].append(logits)

    def wandb_lookup_values(self, label, epoch, losses):
        if not self.ddp_mode: return
        num_ranks = torch.cuda.device_count()
        wandb_dict = {f'{label}/step': epoch}

        rankwise_losses = [None for _ in range(num_ranks)]
        dist.all_gather_object(rankwise_losses, losses)
        rankwise_losses = np.vstack(rankwise_losses).mean(0).tolist()

        self.print0(f"Loss Report for Epoch #{epoch}")
        self.print0(f"Batchwise Loss for {label} Data Partition")
        for idx, loss in enumerate(rankwise_losses):
            self.print0(f"Batchwise Loss Term Index {idx+1}: {loss:.3f}")
            wandb_dict[f'{label}/loss/idx{idx+1}'] = loss

        if len(self.lookup_values['true_dgeR']) > 0:
            y, yhat = [None for _ in range(num_ranks)], [None for _ in range(num_ranks)]
            dist.all_gather_object(y,    self.lookup_values['true_dgeR'])
            dist.all_gather_object(yhat, self.lookup_values['pred_dgeR'])

            ids = [None for _ in range(num_ranks)]
            dist.all_gather_object(ids, self.lookup_values['meta_id'])
            ids = sum(ids, [])

            y, yhat = np.vstack(y), np.vstack(yhat)
            hitratio_dict, jaccard_dict = hitratio_jaccard_(y, yhat)

            for k, v in hitratio_dict.items():
                wandb_dict[f'{label}/dge/{k}'] = v

            for k, v in jaccard_dict.items():
                wandb_dict[f'{label}/dge/{k}'] = v

            z, zhat = np.where(y < 0, 0, y), np.where(yhat < 0, 0, yhat)
            wandb_dict[f'{label}/dge/pos/ndcg']          = ndcg(z, zhat)

            z, zhat = np.where(-y < 0, 0, -y), np.where(-yhat < 0, 0, -yhat)
            wandb_dict[f'{label}/dge/neg/ndcg']          = ndcg(z, zhat)


            wandb_dict[f'{label}/dge/rmse']              = rmse(y, yhat)

            if label == 'valid':
                for k, v in wandb_dict.items():
                    temp = k.split(f'{label}/')[1]
                    self.current_valid_metric[temp] = v

            if self.rank == 0:
                if label == 'test':
                    wandb_dict[f'{label}/dge/pos/precision/10']  = precision_k(y, yhat, 10)
                    wandb_dict[f'{label}/dge/pos/precision/50']  = precision_k(y, yhat, 50)
                    wandb_dict[f'{label}/dge/pos/precision/100'] = precision_k(y, yhat, 100)
                    wandb_dict[f'{label}/dge/pos/precision/200'] = precision_k(y, yhat, 200)
                    wandb_dict[f'{label}/dge/pos/kendall_tau']   = kendall_tau(y, yhat)
                    wandb_dict[f'{label}/dge/pos/map']           = mean_average_precision(y, yhat)
    
                    wandb_dict[f'{label}/dge/neg/precision/10']  = precision_k(-y, -yhat, 10)
                    wandb_dict[f'{label}/dge/neg/precision/50']  = precision_k(-y, -yhat, 50)
                    wandb_dict[f'{label}/dge/neg/precision/100'] = precision_k(-y, -yhat, 100)
                    wandb_dict[f'{label}/dge/neg/precision/200'] = precision_k(-y, -yhat, 200)
                    wandb_dict[f'{label}/dge/neg/kendall_tau']   = kendall_tau(-y, -yhat)
                    wandb_dict[f'{label}/dge/neg/map']           = mean_average_precision(-y, -yhat)
                    
                    yhat = pd.DataFrame(yhat, index=ids)
                    y    = pd.DataFrame(y,    index=ids)
                    yhat.to_csv(self.checkpoint_path + f'/pred_results_{label}_dge.csv')
                    y.to_csv(self.checkpoint_path + f'/true_results_{label}_dge.csv')

        if len(self.lookup_values['true_dpwB']) > 0:
            auc_list, aup_list, f1_list, acc_list = [], [], [], []
            for y, yhat in zip(self.lookup_values['true_dpwB'], self.lookup_values['pred_dpwB']):
                try:
                    auc_list.append(auc(y, yhat))
                    aup_list.append(aup(y, yhat))
                    yhat = (yhat > 0.5).reshape(-1) 
                    f1_list.append(f1(y, yhat))
                    acc_list.append(acc(y, yhat))
                except:
                    pass

            auc_gathered = [None for _ in range(num_ranks)]
            aup_gathered = [None for _ in range(num_ranks)]
            f1_gathered  = [None for _ in range(num_ranks)]
            acc_gathered  = [None for _ in range(num_ranks)]
            dist.all_gather_object(auc_gathered, auc_list)
            dist.all_gather_object(aup_gathered, aup_list)
            dist.all_gather_object(f1_gathered,  f1_list)
            dist.all_gather_object(acc_gathered, acc_list)

            if self.rank == 0:
                auc_gathered = sum(auc_gathered, [])
                aup_gathered = sum(aup_gathered, [])
                f1_gathered  = sum(f1_gathered, [])
                acc_gathered = sum(acc_gathered, [])

                wandb_dict[f'{label}/dpw/auroc']    = np.mean(auc_gathered)
                wandb_dict[f'{label}/dpw/auprc']    = np.mean(aup_gathered)
                wandb_dict[f'{label}/dpw/f1score']  = np.mean(f1_gathered)
                wandb_dict[f'{label}/dpw/accuracy'] = np.mean(acc_gathered)

        if self.rank == 0: self.wandb.log(wandb_dict)

        return

    def get_optimizer(self, model):
        if self.pred_model in ['mlp', 'ciger', 'deepce', 'badger']:
            optimizer = optim.Adam(model.parameters(), 
                                  lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise
        return optimizer

    def get_scheduler(self):
        if self.pred_model in ['mlp', 'ciger', 'deepce', 'badger']:
            scheduler = DummyScheduler()
        else: 
            raise
        return scheduler

    def train_valid(self, model, train, train_sampler=None, valid=None):
        self.train_steps = len(train)
        num_ranks = torch.cuda.device_count()
        print(f"RANK: {self.rank+1} | Training Batches: {len(train)}, Validation Batches: {len(valid)}")
        EARLY_STOPPING = False

        if self.pred_model == 'knn':
            model.train(train)
        
        else:
            model = model.to(self.rank)

            self.optimizer = self.get_optimizer(model)
            self.scheduler = self.get_scheduler()

            for epoch in range(self.num_epochs):
                if train_sampler: train_sampler.set_epoch(epoch)
                train_loss, model = self.train_step(model, train, epoch)
                self.wandb_lookup_values('train', epoch, train_loss) 
                self.reset_lookup_values()

                eval_loss, _ = self.eval_step(model, valid)
                self.wandb_lookup_values('valid', epoch, eval_loss)
                self.reset_lookup_values()


                if not self.check_valid_progress():
                    self.num_patience -= 1
                    if self.num_patience == 0: 
                        EARLY_STOPPING = True

                if self.rank == 0:
                    if self.best_valid_metric['dge/pos/ndcg'] < self.current_valid_metric['dge/pos/ndcg']:
                        print("Saving Model Checkpoint with Best Validation Performance... [pos/ndcg]")
                        self.best_valid_metric['dge/pos/ndcg'] = self.current_valid_metric['dge/pos/ndcg']
                        torch.save(model.module.state_dict(), self.checkpoint_path + f'/best_epoch_dge_pos_ndcg.mdl')
                    if self.best_valid_metric['dge/neg/ndcg'] < self.current_valid_metric['dge/neg/ndcg']:
                        print("Saving Model Checkpoint with Best Validation Performance... [neg/ndcg]")
                        self.best_valid_metric['dge/neg/ndcg'] = self.current_valid_metric['dge/neg/ndcg']
                        torch.save(model.module.state_dict(), self.checkpoint_path + f'/best_epoch_dge_neg_ndcg.mdl')

                if EARLY_STOPPING: break

        return model 

    def train_step(self, model, data, epoch=0):
        model.train()
        if not self.debug_mode:
            torch.distributed.barrier()
        batchwise_loss = [] 

        if self.pred_model in SCHEDULER_FIRSTSTEP:
            self.scheduler.step()
        for idx, batch in enumerate(data):
            self.optimizer.zero_grad()
            batch = model(batch)
            loss = self.calculate_losses(batch)
            sum(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            if self.pred_model in SCHEDULER_BATCHWISE: 
                self.scheduler.step()
            sleep(0.01)

            batchwise_loss.append(list(map(lambda x: x.item(), loss)))

            self.store_lookup_values(batch)
            del batch; torch.cuda.empty_cache()
        if self.pred_model in SCHEDULER_EPOCHWISE:
            self.scheduler.step()

        return np.array(batchwise_loss).mean(0), model

    @torch.no_grad()
    def eval_step(self, model, data):
        model.eval()
        batchwise_loss = []

        for idx, batch in enumerate(data):
            batch = model(batch)            
            loss = self.calculate_losses(batch)
            sleep(0.01)

            batchwise_loss.append(list(map(lambda x: x.item(), loss)))

            self.store_lookup_values(batch)
            del batch; torch.cuda.empty_cache()

        return np.array(batchwise_loss).mean(0), model

    @torch.no_grad()
    def test(self, model, test):
        if self.pred_model == 'knn':
            batch = model.evaluate(test)
            
            yhat = pd.DataFrame(batch['task/dge_pred'], index=batch['meta/id'])
            y    = pd.DataFrame(batch['task/dge_true'],    index=batch['meta/id'])
            yhat.to_csv(self.checkpoint_path + f'/pred_results_test_dge.csv')
            y.to_csv(self.checkpoint_path + f'/true_results_test_dge.csv')
        else:
            print(f"RANK: {self.rank} | Test Batches: {len(test)}")
            model = model.to(self.rank)
            # model = DDP(model, device_ids=[self.rank])
            model = DDP(model, device_ids=[self.rank], find_unused_parameters=False)
            print("Testing Model on RANK: ", self.rank)
            eval_loss, _ = self.eval_step(model, test)
            self.wandb_lookup_values('test',0, eval_loss)
            self.reset_lookup_values()

        return model

