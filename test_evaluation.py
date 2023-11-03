import os
import random
import numpy as np
import torch
import pickle
from sklearn import metrics
import gc
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Set seed for reproducibility
def set_seed(seed=43):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False

set_seed()

def make_2M_filter(len_high_size):
    loss_filter = np.zeros(shape=(len_high_size, len_high_size)) +np.diag(np.ones(shape=(len_high_size,)), k=0)
    for i in range(1,20):
        loss_filter = loss_filter + np.diag(np.ones(shape=(len_high_size-i,)), k=-i) + np.diag(np.ones(shape=(len_high_size-i,)), k=i)
    return loss_filter

def extract_dig(matrix):
    all_diags = []
    for i in range(1, 21):
        diag = np.diag(matrix, k=i)
        all_diags.append(diag)  
    vector = np.concatenate(all_diags)
    return vector

def run_mae(Hr_pred_HiC, Hr_true_HiC):
    loss_filter = np.ones(shape=(40, 40)) - np.diag(np.ones(shape=(40,)), k=0) 
    filter_2M = make_2M_filter(40)
    loss_filter = np.multiply(loss_filter, filter_2M)
    key_list = list(Hr_true_HiC.keys())
    mae_l1 = torch.nn.L1Loss(reduction='mean')
    MAE=0
    count=0
    for key in key_list:
        if (key[0],key[1],key[2]) in Hr_pred_HiC.keys():
            if np.sum(Hr_pred_HiC[(key[0],key[1],key[2])] !=0):
                count +=1
                true = np.multiply(loss_filter,Hr_true_HiC[key])
                key=(key[0],key[1],key[2])
                pred = np.multiply(loss_filter,Hr_pred_HiC[key])
            pred_vector=extract_dig(pred)
            true_vector=extract_dig(true)
            mae = mae_l1(torch.tensor(pred_vector), torch.tensor(true_vector))
            MAE += mae
            count +=1
    MAE_avg = float(MAE/count)
    return MAE_avg

def run_f1_score(Hr_pred_HiC, Hr_true_HiC):
    loss_filter = np.ones(shape=(40, 40)) - np.diag(np.ones(shape=(40,)), k=0) 
    filter_2M = make_2M_filter(40)
    loss_filter = np.multiply(loss_filter, filter_2M)
    key_list = list(Hr_true_HiC.keys())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    threshold = 0
    F1=[]
    macro_F1=[]
    for key in key_list:
        if (key[0],key[1],key[2]) in Hr_pred_HiC.keys():
            if np.sum(Hr_pred_HiC[(key[0],key[1],key[2])] !=0):
                HR_m = torch.tensor(np.multiply(loss_filter,Hr_true_HiC[key]))
                key=(key[0],key[1],key[2])
                pred_m = torch.tensor(np.multiply(loss_filter,Hr_pred_HiC[key]))
            HR_m = torch.where(HR_m > threshold, torch.ones_like(HR_m), torch.zeros_like(HR_m))
            pred_m = torch.where(pred_m > threshold, torch.ones_like(pred_m), torch.zeros_like(pred_m))
            pred_vector=extract_dig(pred_m)
            true_vector=extract_dig(HR_m)
            f1 = metrics.f1_score(true_vector, pred_vector, average='macro') 
            F1.append(f1)
    F1_avg = np.nanmean(F1)
    return F1_avg

if __name__ == '__main__':

    path= os.getcwd()
    with open(path+'/output/test_Lr_dic.pickle', 'rb') as file:
        Lr_HiC = pickle.load(file)
    with open(path+'/output/test_predict_dic.pickle', 'rb') as file:
        Hr_pred_HiC = pickle.load(file)
    with open(path+'/output/test_true_dic.pickle', 'rb') as file:
        Hr_true_HiC = pickle.load(file)
        

    Lr_MAE_avg = run_mae(Lr_HiC, Hr_true_HiC)
    print('Low resolution average MAE: ',Lr_MAE_avg)
    pred_MAE_avg = run_mae(Hr_pred_HiC, Hr_true_HiC)
    print('Prediction average MAE: ',pred_MAE_avg)

    Lr_F1_avg = run_f1_score(Lr_HiC, Hr_true_HiC)
    print('Low resolution average macro F1: ',Lr_F1_avg)
    pred_F1_avg = run_f1_score(Hr_pred_HiC, Hr_true_HiC)
    print('Prediction average macro F1: ',pred_F1_avg)

    
    key_list=list(Lr_HiC.keys())
    for i in range(len(key_list)):
        sample_num = i
        print('ploting ',sample_num)
        if sample_num==20:
            break
        Lr=np.squeeze(Lr_HiC[key_list[i]])
        Pred=np.squeeze(Hr_pred_HiC[key_list[i]])
        true=np.squeeze(Hr_true_HiC[key_list[i]])
        fig, axs = plt.subplots(1,3,figsize=(15, 5))
        fig.suptitle('Lr{}'.format(sample_num))
        ax = axs[0].imshow(np.log1p(Lr), cmap='RdBu_r', vmin= np.percentile(np.log1p(Lr), 0), vmax= np.percentile(np.log1p(Lr), 100))
        axs[0].set_title('Lr{}'.format(sample_num))
        fig.colorbar(ax, ax=axs[0])
        ax = axs[1].imshow(np.log1p(Pred), cmap='RdBu_r', vmin= np.percentile(np.log1p(Pred), 0), vmax= np.percentile(np.log1p(Pred), 100))
        axs[1].set_title('Pred{}'.format(sample_num))
        fig.colorbar(ax, ax=axs[1])
        ax = axs[2].imshow(np.log1p(true), cmap='RdBu_r', vmin= np.percentile(np.log1p(true), 0), vmax= np.percentile(np.log1p(true), 100))
        axs[2].set_title('true{}'.format(sample_num))
        fig.colorbar(ax, ax=axs[2])
        plt.tight_layout()
        with SummaryWriter('runs/heatmap') as writer:
            writer.add_figure('Lr Pred true {}'.format(sample_num), plt.gcf(), global_step=sample_num)
            writer.close()
    
    


        
