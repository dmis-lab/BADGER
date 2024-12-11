import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import torch.nn.functional as F
from typing import Any, Callable, List, Tuple, Union
from extras import *
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score, ndcg_score, average_precision_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import scipy.special as ssp
from torchmetrics.functional import pairwise_cosine_similarity

from collections import Counter

############################
#                          #
#      Loss Functions      #
#                          #   
############################

class point_wise_shrinkage(nn.Module):
    '''
    https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.pdf
    '''
    def __init__(self, rank, **kwargs):
        super(point_wise_shrinkage, self).__init__()
        self.a_coef = torch.FloatTensor([kwargs['a_coef']]).to(rank)
        self.c_coef = torch.FloatTensor([kwargs['c_coef']]).to(rank)

        self.criterion1 = nn.MSELoss(reduction='none').to(rank)
        self.criterion2 = nn.L1Loss(reduction='none').to(rank)

    def forward(self, pred, true):
        l2_loss = self.criterion1(pred, true)
        l1_loss = self.criterion2(pred, true)
        sh_loss = l2_loss / (1 + (self.a_coef*(self.c_coef - l1_loss)).exp())

        return sh_loss.mean()


# def load_coef(reg):
#     coef = pd.read_csv(f'data/{reg}50_genes_coef.csv', index_col=0).values
#     return coef

class test_loss_function(nn.Module):
    def __init__(self, rank):
        super(test_loss_function, self).__init__()
        self.criterion = nn.MSELoss(reduction='none').to(rank)

        self.coef = torch.cuda.FloatTensor([1.0 for _ in range(10)]+[0.0 for _ in range(968)])

    def forward(self, pred, true):
        loss = self.criterion(pred, true)

        return (loss * self.coef.view(1,-1)).sum()


class point_wise_mse(nn.Module):
    def __init__(self, rank):
        super(point_wise_mse, self).__init__()
        self.criterion = nn.MSELoss().to(rank)

    def forward(self, pred, true):

        return self.criterion(pred, true)

class point_wise_mse_trainweighted(nn.Module):
    def __init__(self, rank):
        super(point_wise_mse_trainweighted, self).__init__()
        self.criterion = nn.MSELoss(reduction='none').to(rank)
        # self.top_genes_coef = self.train_stats['top'].values
        # self.bot_genes_coef = self.train_stats['bot'].values

    def forward(self, pred, true):
        print("This is a Base Module")
        raise

class point_wise_mse_trainweighted_idf(point_wise_mse_trainweighted):
    def __init__(self, rank):
        super(point_wise_mse_trainweighted_idf, self).__init__(rank)            
    def forward(self, pred, true):
        top_coef = self.train_stats['top'].values
        N        = self.train_stats['N']
        top_coef = np.log(N / (top_coef+1))
        top_coef = torch.cuda.FloatTensor(top_coef)
        top_mse  = (self.criterion(pred, true) * top_coef).mean()

        bot_coef = self.train_stats['bot'].values
        N        = self.train_stats['N']
        bot_coef = np.log(N / (bot_coef+1))
        bot_coef = torch.cuda.FloatTensor(bot_coef)
        bot_mse  = (self.criterion(pred, true) * bot_coef).mean()

        return (top_mse + bot_mse) / 2

class point_wise_mse_trainweighted_idf_max(point_wise_mse_trainweighted):
    def __init__(self, rank):
        super(point_wise_mse_trainweighted_idf_max, self).__init__(rank)
        
    def forward(self, pred, true):
        top_coef = self.train_stats['top'].values
        N        = self.train_stats['N']
        top_coef = np.log(top_coef.max() / (top_coef+1))
        top_coef = torch.cuda.FloatTensor(top_coef)
        top_mse  = (self.criterion(pred, true) * top_coef).mean()

        bot_coef = self.train_stats['bot'].values
        N        = self.train_stats['N']
        bot_coef = np.log(bot_coef.max() / (bot_coef+1))
        bot_coef = torch.cuda.FloatTensor(bot_coef)
        bot_mse  = (self.criterion(pred, true) * bot_coef).mean()

        return (top_mse + bot_mse) / 2

class point_wise_mse_trainweighted_idf_max_modified(point_wise_mse_trainweighted):
    def __init__(self, rank):
        super(point_wise_mse_trainweighted_idf_max_modified, self).__init__(rank)
        
    def forward(self, pred, true):
        top_coef = self.train_stats['top'].values
        N        = self.train_stats['N']
        top_coef = np.log(1/ ((top_coef/top_coef.max())+1))
        # top_coef = np.log(top_coef.max() / (top_coef+1))
        top_coef = torch.cuda.FloatTensor(top_coef)
        top_mse  = (self.criterion(pred, true) * top_coef).mean()

        bot_coef = self.train_stats['bot'].values
        N        = self.train_stats['N']
        bot_coef = np.log(1 / ((bot_coef/bot_coef.max())+1))
        # bot_coef = np.log(bot_coef.max() / (bot_coef+1))
        bot_coef = torch.cuda.FloatTensor(bot_coef)
        bot_mse  = (self.criterion(pred, true) * bot_coef).mean()

        return (top_mse + bot_mse) / 2

class point_wise_mse_trainweighted_idf_prob(point_wise_mse_trainweighted):
    def __init__(self, rank):
        super(point_wise_mse_trainweighted_idf_prob, self).__init__(rank)
        
    def forward(self, pred, true):

        top_coef = self.train_stats['top'].values
        N        = self.train_stats['N']
        top_coef = np.log((N - top_coef) / (top_coef+1))
        top_coef = torch.cuda.FloatTensor(top_coef)
        top_mse  = (self.criterion(pred, true) * top_coef).mean()

        bot_coef = self.train_stats['bot'].values
        N        = self.train_stats['N']
        bot_coef = np.log((N - bot_coef) / (bot_coef+1))
        bot_coef = torch.cuda.FloatTensor(bot_coef)
        bot_mse  = (self.criterion(pred, true) * bot_coef).mean()

        return (top_mse + bot_mse) / 2

def skewed_normal(x, mean, sd, alpha=-0.25):
    t = (x-mean) / sd
    denom = math.sqrt(2 * math.pi)
    numer = np.exp(-0.5* (t**2))
    cdf_t = 0.5 * (1 + math.erf(alpha * t / math.sqrt(2)))
    pdf_t = numer / denom
    return (2*pdf_t*cdf_t)

class point_wise_mse_trainweighted_skewed_normal(nn.Module):
    def __init__(self, rank):
        super(point_wise_mse_trainweighted_skewed_normal, self).__init__()
        self.criterion = nn.MSELoss(reduction='none').to(rank)
        
    def forward(self, pred, true):
        top_coef = self.train_stats['top'].values
        top_coef = self.train_stats['top'].apply(lambda x: skewed_normal(x, np.mean(top_coef), np.std(top_coef)))
   
        top_coef = torch.cuda.FloatTensor(top_coef)
        top_mse  = (self.criterion(pred, true) * top_coef).mean()

        bot_coef = self.train_stats['bot'].values
        bot_coef = self.train_stats['bot'].apply(lambda x: skewed_normal(x, np.mean(bot_coef), np.std(bot_coef)))
        bot_coef = torch.cuda.FloatTensor(bot_coef)
        bot_mse  = (self.criterion(pred, true) * bot_coef).mean()

        return (top_mse + bot_mse) / 2
    
class point_wise_mse_batchweighted(nn.Module):
    def __init__(self, rank):
        super(point_wise_mse_batchweighted, self).__init__()
        self.criterion = nn.MSELoss(reduction='none').to(rank)
        self.alpha, self.beta = 3.85, 1.

    def forward(self, pred, true):
        batch_size  = true.size(0)
        true_ranked = numpify(torch.argsort(true, descending=True)[:, :50])
        true_counts = Counter(true_ranked.reshape(-1).tolist())

        top_genes_coef = np.zeros(978)
        for k,v in true_counts.items():
            top_genes_coef[k] = self.alpha / (1 + np.exp(v)) if v != 0 else 0.
            # top_genes_coef[k] = np.log(batch_size / (1+v))
        top_genes_coef = torch.cuda.FloatTensor(top_genes_coef)
        top_mse = (self.criterion(pred, true) * top_genes_coef).mean()

        true_ranked = numpify(torch.argsort(true, descending=False)[:, :50])
        true_counts = Counter(true_ranked.reshape(-1).tolist())
        bot_genes_coef = np.zeros(978)
        for k,v in true_counts.items():
            bot_genes_coef[k] = self.alpha / (1 + np.exp(v)) if v != 0 else 0.
        bot_genes_coef = torch.cuda.FloatTensor(bot_genes_coef)
        bot_mse = (self.criterion(pred, true) * bot_genes_coef).mean()

        return (top_mse + bot_mse) / 2

class point_wise_mse_dynamic(nn.Module):
    def __init__(self, rank):
        super(point_wise_mse_dynamic, self).__init__()
        self.criterion = nn.MSELoss(reduction='none').to(rank)
        self.alpha = 2
        
    def forward(self, pred, true):
        batch_size  = true.size(0)
        coef = torch.abs(pred-true)
        coef = torch.where(coef > self.alpha, coef, coef/2)
        
        return (coef * self.criterion(pred, true)).mean()


class classification_cross_entropy(nn.Module):
    def __init__(self, rank):
        super(classification_cross_entropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss().to(rank)

    def forward(self, pred, true):
        shape = pred.size()
        true = true.view(shape[0] * shape[1])
        pred = pred.view(shape[0] * shape[1], shape[2])

        return self.criterion(pred, true)

class pair_wise_ranknet(nn.Module):
    """
    From RankNet to LambdaRank to LambdaMART: An Overview
    :param predict: [batch, ranking_size]
    :param label: [batch, ranking_size]
    :return:
    """
    def __init__(self, rank):
        super(pair_wise_ranknet, self).__init__()
        self.rank = rank

    def forward(self, pred, true):
        pred_diffs = torch.unsqueeze(pred, dim=2) - torch.unsqueeze(pred, dim=1)  # computing pairwise differences, i.e., Sij or Sxy
        pred_pairwise_cmps = tor_batch_triu(pred_diffs, k=1, device=self.rank) # k should be 1, thus avoids self-comparison
        tmp_true_diffs = torch.unsqueeze(true, dim=2) - torch.unsqueeze(true, dim=1)  # computing pairwise differences, i.e., Sij or Sxy
        std_ones = torch.ones(tmp_true_diffs.size()).to(self.rank).double()
        std_minus_ones = std_ones - 2.0
        true_diffs = torch.where(tmp_true_diffs > 0, std_ones, tmp_true_diffs)
        true_diffs = torch.where(true_diffs < 0, std_minus_ones, true_diffs)
        true_pairwise_cmps = tor_batch_triu(true_diffs, k=1, device=self.rank)  # k should be 1, thus avoids self-comparison
        loss_1st_part = (1.0 - true_pairwise_cmps) * pred_pairwise_cmps * 0.5   # cf. the equation in page-3
        loss_2nd_part = torch.log(torch.exp(-pred_pairwise_cmps) + 1.0)    # cf. the equation in page-3
        loss = torch.sum(loss_1st_part + loss_2nd_part)
        
        return loss.to(self.rank)


class pair_wise_ranknet_trainweighted_idf(nn.Module):
    """
    From RankNet to LambdaRank to LambdaMART: An Overview
    :param predict: [batch, ranking_size]
    :param label: [batch, ranking_size]
    :return:
    """
    def __init__(self, rank):
        super(pair_wise_ranknet_trainweighted_idf, self).__init__()
        self.rank = rank

    def forward(self, pred, true):
        top_coef = self.train_stats['top'].values
        N        = self.train_stats['N']
        top_coef = np.log(top_coef.max() / (top_coef+1))
        top_coef = torch.cuda.FloatTensor(top_coef)
        top_coef = top_coef.view(1,-1,1).repeat(pred.size(0),1,978)
        top_coef = tor_batch_triu(top_coef, k=1, device=self.rank)

        bot_coef = self.train_stats['bot'].values
        N        = self.train_stats['N']
        bot_coef = np.log(bot_coef.max() / (bot_coef+1))
        bot_coef = torch.cuda.FloatTensor(bot_coef)
        bot_coef = bot_coef.view(1,-1,1).repeat(pred.size(0),1,978)
        bot_coef = tor_batch_triu(bot_coef, k=1, device=self.rank)

        pred_diffs = torch.unsqueeze(pred, dim=2) - torch.unsqueeze(pred, dim=1)  # computing pairwise differences, i.e., Sij or Sxy
        pred_pairwise_cmps = tor_batch_triu(pred_diffs, k=1, device=self.rank) # k should be 1, thus avoids self-comparison
        tmp_true_diffs = torch.unsqueeze(true, dim=2) - torch.unsqueeze(true, dim=1)  # computing pairwise differences, i.e., Sij or Sxy
        std_ones = torch.ones(tmp_true_diffs.size()).to(self.rank).double()
        std_minus_ones = std_ones - 2.0
        true_diffs = torch.where(tmp_true_diffs > 0, std_ones, tmp_true_diffs)
        true_diffs = torch.where(true_diffs < 0, std_minus_ones, true_diffs)
        true_pairwise_cmps = tor_batch_triu(true_diffs, k=1, device=self.rank)  # k should be 1, thus avoids self-comparison
        loss_1st_part = (1.0 - true_pairwise_cmps) * pred_pairwise_cmps * 0.5   # cf. the equation in page-3
        loss_2nd_part = torch.log(torch.exp(-pred_pairwise_cmps) + 1.0)    # cf. the equation in page-3

        loss_top_coef = (loss_1st_part + loss_2nd_part) * top_coef
        loss_bot_coef = (loss_1st_part + loss_2nd_part) * bot_coef
        loss = torch.sum(loss_top_coef + loss_bot_coef) * 0.5
        
        return loss.to(self.rank)


class list_wise_listnet(nn.Module):
    def __init__(self, rank):
        super(list_wise_listnet, self).__init__()
        self.rank = rank

    def forward(self, pred, true):
        true = F.softmax(true, dim=1)
        pred = F.softmax(pred, dim=1)
        loss = -(true * torch.log(pred)).sum(dim=1).mean()
        
        return loss.to(self.rank)

class list_wise_listmle(nn.Module):
    def __init__(self, rank):
        super(list_wise_listmle, self).__init__()
        self.rank = rank

    def forward(self, pred, true):
        shape = true.size()
        index = torch.argsort(true, descending=True)
        tmp = torch.zeros(shape[0] * shape[1], dtype=torch.int64).to(self.rank)
        for i in range(0, shape[0] * shape[1], shape[1]):
            tmp[i:(i + shape[1])] += i
        index = index.view(shape[0] * shape[1])
        index += tmp
        pred = pred.view(shape[0] * shape[1])
        pred = pred[index]
        pred = pred.view(shape[0], shape[1])
        pred_logcumsum = apply_LogCumsumExp(pred)
        loss = (pred_logcumsum - pred).sum(dim=1).mean()
        
        return loss.to(self.rank)

class list_wise_rankcosine(nn.Module):
    def __init__(self, rank):
        super(list_wise_rankcosine, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=1).to(rank)

    def forward(self, pred, true):

        return torch.mean((1.0 - self.criterion(pred, true)) / 0.5)

 
class list_wise_ndcg(nn.Module):
    def __init__(self, rank):
        super(list_wise_ndcg, self).__init__()
        self.rank = rank

    def forward(self, pred, true):
        approx_nDCG = apply_ApproxNDCG_OP(pred, true)
        loss = -torch.mean(approx_nDCG)

        return loss.to(self.rank)

class hybrid_cosine_mse(nn.Module):
    def __init__(self, rank):
        super(hybrid_cosine_mse, self).__init__()
        self.rank = rank

        self.loss1 = list_wise_rankcosine(rank)
        self.loss2 = point_wise_mse(rank)

    def forward(self, pred, true):

        return self.loss1(pred, true) + self.loss2(pred, true)

class hybrid_cosine_mse_batchweighted(nn.Module):
    def __init__(self, rank):
        super(hybrid_cosine_mse_batchweighted, self).__init__()
        self.rank = rank

        self.loss1 = list_wise_rankcosine(rank)
        self.loss2 = point_wise_mse_batchweighted(rank)

    def forward(self, pred, true):

        return self.loss1(pred, true) + self.loss2(pred, true)

class hybrid_cosine_mse_trainweighted_idf(nn.Module):
    def __init__(self, rank):
        super(hybrid_cosine_mse_trainweighted_idf, self).__init__()
        self.rank = rank

        self.loss1 = list_wise_rankcosine(rank)
        self.loss2 = point_wise_mse_trainweighted_idf(rank)

    def forward(self, pred, true):
        self.loss2.train_stats = self.train_stats

        return self.loss1(pred, true) + self.loss2(pred, true)

class hybrid_cosine_mse_trainweighted_idf_max(nn.Module):
    def __init__(self, rank):
        super(hybrid_cosine_mse_trainweighted_idf_max, self).__init__()
        self.rank = rank

        self.loss1 = list_wise_rankcosine(rank)
        self.loss2 = point_wise_mse_trainweighted_idf_max(rank)

    def forward(self, pred, true):
        self.loss2.train_stats = self.train_stats

        return self.loss1(pred, true) + self.loss2(pred, true)

class hybrid_cosine_mse_trainweighted_idf_max_modified(nn.Module):
    def __init__(self, rank):
        super(hybrid_cosine_mse_trainweighted_idf_max_modified, self).__init__()
        self.rank = rank

        self.loss1 = list_wise_rankcosine(rank)
        self.loss2 = point_wise_mse_trainweighted_idf_max_modified(rank)

    def forward(self, pred, true):
        self.loss2.train_stats = self.train_stats

        return self.loss1(pred, true) + self.loss2(pred, true)

class hybrid_cosine_mse_trainweighted_idf_prob(nn.Module):
    def __init__(self, rank):
        super(hybrid_cosine_mse_trainweighted_idf_prob, self).__init__()
        self.rank = rank

        self.loss1 = list_wise_rankcosine(rank)
        self.loss2 = point_wise_mse_trainweighted_idf_prob(rank)

    def forward(self, pred, true):
        self.loss2.train_stats = self.train_stats

        return self.loss1(pred, true) + self.loss2(pred, true)

class hybrid_cosine_mse_trainweighted_skewed_normal(nn.Module):
    def __init__(self, rank):
        super(hybrid_cosine_mse_trainweighted_skewed_normal, self).__init__()
        self.rank = rank
        self.beta = 0.5
        self.loss1 = list_wise_rankcosine(rank)
        self.loss2 = point_wise_mse_trainweighted_skewed_normal(rank)

    def forward(self, pred, true):
        self.loss2.train_stats = self.train_stats
        return ((1-self.beta)*self.loss1(pred, true)) + (self.beta*self.loss2(pred, true))
    
class margin_rankingloss(nn.Module):
    def __init__(self, rank):
        super(margin_rankingloss, self).__init__()
        self.rank = rank
        self.margin = 10
        self.criterion = nn.MarginRankingLoss(self.margin, reduction='none').to(rank)
    def forward(self, pred, true):
        pred_ = torch.argsort(torch.argsort(pred, axis=1))
        true_ = torch.argsort(torch.argsort(true, axis=1))
        
        pred = pred - pred + pred_
        true = true - true + true_
 
        y =torch.where((pred-true)<1, 1, -1)
        return torch.mean(self.criterion(pred, true, y))

class margin_rankingloss_trainweighted_idf_max_modified(margin_rankingloss):
    def __init__(self, rank):
        super(margin_rankingloss_trainweighted_idf_max_modified, self).__init__(rank)
        
    def forward(self, pred, true):
        pred = torch.argsort(pred, axis=1)
        true = torch.argsort(true, axis=1)
        y =torch.where((pred-true)<1, 1, -1)
        
        top_coef = self.train_stats['top'].values
        N        = self.train_stats['N']
        top_coef = np.log(1/ ((top_coef/top_coef.max())+1))
        top_coef = torch.cuda.FloatTensor(top_coef)
        top_coef = top_coef[torch.where(true < 50, top_coef, 1)]
        top_mse  = (self.criterion(pred, true) * top_coef).mean()

        bot_coef = self.train_stats['bot'].values
        N        = self.train_stats['N']
        bot_coef = np.log(1 / ((bot_coef/bot_coef.max())+1))
        bot_coef = torch.cuda.FloatTensor(bot_coef)
        bot_coef = bot_coef[torch.where(true > 927, bot_coef, 1)]
        bot_mse  = (self.criterion(pred, true) * bot_coef).mean()

        return (top_mse + bot_mse) / 2
        

loss_function_dict = {
    'point_wise_mse': point_wise_mse,
    'pair_wise_ranknet': pair_wise_ranknet,
    'pair_wise_ranknet_trainweighted_idf': pair_wise_ranknet_trainweighted_idf,
    'list_wise_listnet': list_wise_listnet,
    'list_wise_listmle': list_wise_listmle,
    'list_wise_rankcosine': list_wise_rankcosine,
    'list_wise_ndcg': list_wise_ndcg,
    'deepce': point_wise_mse,
    'ciger': list_wise_rankcosine,
    'point_wise_mse_dynamic': point_wise_mse_dynamic,
    'point_wise_mse_trainweighted_skewed_normal': point_wise_mse_trainweighted_skewed_normal,
    'hybrid_cosine_mse_batchweighted': hybrid_cosine_mse_batchweighted,
    'point_wise_mse_trainweighted': point_wise_mse_trainweighted,
    'point_wise_mse_trainweighted_idf':      point_wise_mse_trainweighted_idf,
    'point_wise_mse_trainweighted_idf_max':  point_wise_mse_trainweighted_idf_max,
    'point_wise_mse_trainweighted_idf_max_modified':  point_wise_mse_trainweighted_idf_max_modified,
    'point_wise_mse_trainweighted_idf_prob': point_wise_mse_trainweighted_idf_prob,
    'hybrid_cosine_mse': hybrid_cosine_mse,
    'hybrid_cosine_mse_trainweighted_idf':      hybrid_cosine_mse_trainweighted_idf,
    'hybrid_cosine_mse_trainweighted_idf_max':  hybrid_cosine_mse_trainweighted_idf_max,
    'hybrid_cosine_mse_trainweighted_idf_max_modified':  hybrid_cosine_mse_trainweighted_idf_max_modified,
    'hybrid_cosine_mse_trainweighted_idf_prob': hybrid_cosine_mse_trainweighted_idf_prob,
    'hybrid_cosine_mse_trainweighted_skewed_normal': hybrid_cosine_mse_trainweighted_skewed_normal,
    'point_wise_shrinkage': point_wise_shrinkage,
    'margin_rankingloss': margin_rankingloss,
    'test_loss_function': test_loss_function
}

def load_loss_function(conf, rank):
    a_coef = conf.train_params.point_wise_shrinkage.a_coef
    c_coef = conf.train_params.point_wise_shrinkage.c_coef
    
    if conf.train_params.baseline_default:
        if conf.model_params.pred_model in loss_function_dict.keys():
            return loss_function_dict[conf.model_params.pred_model](rank)
    if 'shrinkage' in conf.train_params.loss_function:
        return loss_function_dict[conf.train_params.loss_function](rank, a_coef=a_coef, c_coef=c_coef)
    else:
        return loss_function_dict[conf.train_params.loss_function](rank)

class Masked_NLLLoss(nn.Module):
    def __init__(self, rank, weight=[]):
        super(Masked_NLLLoss, self).__init__()
        if len(weight) == 0:
            #self.criterion = nn.NLLLoss(reduce=False).to(rank)
            self.criterion = nn.NLLLoss(reduction='none').to(rank)
        else:
            #self.criterion = nn.NLLLoss(reduce=False, weight=torch.cuda.FloatTensor(weight)).to(rank)
            self.criterion = nn.NLLLoss(reduction='none', weight=torch.cuda.FloatTensor(weight)).to(rank)

    def forward(self, pred, label, mask):
        pred  = (pred.view(pred.size(0)*pred.size(1),pred.size(2))+0.1).log()
        label = label.view(-1)

        mask  = mask.view(-1)
        loss  = torch.sum(self.criterion(pred, label)*mask) / torch.sum(mask).clamp(min=1e-10)

        return loss  

class AttentionL1Loss(nn.Module):
    def __init__(self, rank):
        super(AttentionL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction='sum').to(rank)

    def forward(self, pred, label):
        loss = self.criterion(pred, label)

        return loss

class SpreadoutLoss(nn.Module):
    def __init__(self, rank, tau=10.):
        super(SpreadoutLoss, self).__init__()
        self.tau = tau

    def forward(self, embeddings):
        vdim = embeddings.size(2)**0.5
        loss = 0.
        for i in range(embeddings.size(1)):
            pls = pairwise_cosine_similarity(embeddings[:,i,:])
            pls = torch.nan_to_num(pls, nan=0.0, posinf=1.0, neginf=-1.0)
            pls = pls.sum() / 2.
            loss += pls

        return loss / embeddings.size(1)

class DummyScheduler:
    def __init__(self):
        x = 0
    def step(self):
        return 

def numpify(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return [element.detach().cpu().numpy() for element in tensor]
    else:
        return tensor


################################
#                              #
#      Evaluation Metrics      #
#                              #   
################################


def hitratio_jaccard_(y, yhat):
    # Positive
    ranked_y    = np.argsort(y,    axis=1)
    ranked_yhat = np.argsort(yhat, axis=1)
    hitratio_dict = {
        'pos/hitratio/10':  [],
        'pos/hitratio/50':  [],
        'pos/hitratio/100': [],
        'pos/hitratio/200': [],
        'neg/hitratio/10':  [],
        'neg/hitratio/50':  [],
        'neg/hitratio/100': [],
        'neg/hitratio/200': [],
        'abs/hitratio/10':  [],
        'abs/hitratio/50':  [],
        'abs/hitratio/100': [],
        'abs/hitratio/200': []
    }

    jaccard_dict = {
        'pos/jaccard/10':  [],
        'pos/jaccard/50':  [],
        'pos/jaccard/100': [],
        'pos/jaccard/200': [],
        'neg/jaccard/10':  [],
        'neg/jaccard/50':  [],
        'neg/jaccard/100': [],
        'neg/jaccard/200': [],
        'abs/jaccard/10':  [],
        'abs/jaccard/50':  [],
        'abs/jaccard/100': [],
        'abs/jaccard/200': []
    }

    for K in [10, 50, 100, 200]:
        ranked_y_K    = ranked_y[::-1][:,:K]
        ranked_yhat_K = ranked_yhat[::-1][:,:K]
        for i in range(len(ranked_y_K)):
            ranked_y_K_ith    = set(ranked_y_K[i])
            ranked_yhat_K_ith = set(ranked_yhat_K[i])
            hits = len(ranked_y_K_ith & ranked_yhat_K_ith)
            hitratio_dict[f'pos/hitratio/{K}'].append(hits/K)
            U = len(ranked_y_K_ith | ranked_yhat_K_ith)
            jaccard_dict[f'pos/jaccard/{K}'].append(hits/U)
        hitratio_dict[f'pos/hitratio/{K}'] = np.mean(hitratio_dict[f'pos/hitratio/{K}'])
        jaccard_dict[f'pos/jaccard/{K}']   = np.mean(jaccard_dict[f'pos/jaccard/{K}'])

    for K in [10, 50, 100, 200]:
        ranked_y_K    = ranked_y[:,:K]
        ranked_yhat_K = ranked_yhat[:,:K]
        for i in range(len(ranked_y_K)):
            ranked_y_K_ith    = set(ranked_y_K[i])
            ranked_yhat_K_ith = set(ranked_yhat_K[i])
            hits = len(ranked_y_K_ith & ranked_yhat_K_ith)
            hitratio_dict[f'neg/hitratio/{K}'].append(hits/K)
            U = len(ranked_y_K_ith | ranked_yhat_K_ith)
            jaccard_dict[f'neg/jaccard/{K}'].append(hits/U)
        hitratio_dict[f'neg/hitratio/{K}'] = np.mean(hitratio_dict[f'neg/hitratio/{K}'])
        jaccard_dict[f'neg/jaccard/{K}']   = np.mean(jaccard_dict[f'neg/jaccard/{K}'])

    ranked_y    = np.argsort(np.abs(y),    axis=1)
    ranked_yhat = np.argsort(np.abs(yhat), axis=1)

    for K in [10, 50, 100, 200]:
        ranked_y_K    = ranked_y[::-1][:,:K]
        ranked_yhat_K = ranked_yhat[::-1][:,:K]
        for i in range(len(ranked_y_K)):
            ranked_y_K_ith    = set(ranked_y_K[i])
            ranked_yhat_K_ith = set(ranked_yhat_K[i])
            hits = len(ranked_y_K_ith & ranked_yhat_K_ith)
            hitratio_dict[f'abs/hitratio/{K}'].append(hits/K)
            U = len(ranked_y_K_ith | ranked_yhat_K_ith)
            jaccard_dict[f'abs/jaccard/{K}'].append(hits/U)
        hitratio_dict[f'abs/hitratio/{K}'] = np.mean(hitratio_dict[f'abs/hitratio/{K}'])
        jaccard_dict[f'abs/jaccard/{K}']   = np.mean(jaccard_dict[f'abs/jaccard/{K}'])

    return hitratio_dict, jaccard_dict


def precision_k(label_test, label_predict, k):
    num_pos = 200
    num_neg = 200
    label_test = np.argsort(label_test, axis=1)
    label_predict = np.argsort(label_predict, axis=1)
    precision_k_neg = []
    precision_k_pos = []
    # neg_test_set = label_test[:, :num_neg]
    pos_test_set = label_test[:, -num_pos:]
    # neg_predict_set = label_predict[:, :k]
    pos_predict_set = label_predict[:, -k:]
    for i in range(len(pos_test_set)):
        # neg_test = set(neg_test_set[i])
        pos_test = set(pos_test_set[i])
        # neg_predict = set(neg_predict_set[i])
        pos_predict = set(pos_predict_set[i])
        # precision_k_neg.append(len(neg_test.intersection(neg_predict)) / k)
        precision_k_pos.append(len(pos_test.intersection(pos_predict)) / k)
    return np.mean(precision_k_pos)


def kendall_tau(label_test, label_predict):
    score = []
    for lb_test, lb_predict in zip(label_test, label_predict):
        tau, p_value = kendalltau(lb_test, lb_predict)
        score.append(tau)
    return np.mean(score)


def mean_average_precision(label_test, label_predict):
    k = 200
    score = []
    pos_idx = np.argsort(label_test, axis=1)[:, (-k):]
    label_test_binary = np.zeros_like(label_test)
    for i in range(len(label_test_binary)):
        label_test_binary[i][pos_idx[i]] = 1
        score.append(average_precision_score(label_test_binary[i], label_predict[i]))
    return np.mean(score)


def rmse(label_test, label_predict):
    return np.sqrt(mean_squared_error(label_test, label_predict))


def correlation(label_test, label_predict, correlation_type):
    if correlation_type == 'pearson':
        corr = pearsonr
    elif correlation_type == 'spearman':
        corr = spearmanr
    else:
        raise ValueError("Unknown correlation type: %s" % correlation_type)
    score = []
    for lb_test, lb_predict in zip(label_test, label_predict):
        score.append(corr(lb_test, lb_predict)[0])
    return np.mean(score), score


def auroc(label_test, label_predict):
    label_test = label_test.reshape(-1)
    label_predict = label_predict.reshape(-1)
    return roc_auc_score(label_test, label_predict)


def auprc(label_test, label_predict):
    label_test = label_test.reshape(-1)
    label_predict = label_predict.reshape(-1)
    return average_precision_score(label_test, label_predict)


def ndcg(label_test, label_predict):
    return ndcg_score(label_test, label_predict)


def ndcg_per_sample(label_test, label_predict):
    score = []
    for i in range(len(label_test)):
        score.append(ndcg_score(label_test[i].reshape(1, 978), label_predict[i].reshape(1, 978)))
    return score


def ndcg_random(label_test):
    label_test = np.repeat(label_test, 100, axis=0)
    label_predict = np.array([np.random.permutation(978) for i in range(len(label_test))])
    return ndcg_score(label_test, label_predict)


def auroc_per_cell(label_test, label_predict, cell_idx):
    score = []
    for c_idx in cell_idx:
        lb_test = label_test[c_idx].reshape(-1)
        lb_predict = label_predict[c_idx].reshape(-1)
        score.append(roc_auc_score(lb_test, lb_predict))
    return score


def ndcg_per_cell(label_test, label_predict, cell_idx):
    score = []
    for c_idx in cell_idx:
        lb_test = label_test[c_idx]
        lb_predict = label_predict[c_idx]
        score.append(ndcg_score(lb_test, lb_predict))
    return score