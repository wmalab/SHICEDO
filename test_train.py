import os
import random
import numpy as np
import torch 
from torch.utils.data import TensorDataset, DataLoader
from model_loss import fit,model



seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False


def load_data(file_path,BATCH_SIZE):
    path = file_path +'train_large_img.pt'
    train_large_img = torch.load(path)
    path = file_path +'train_regular_img.pt'
    train_regular_img = torch.load(path)
    path = file_path + 'train_true_img.pt'
    train_True_img = torch.load(path)
    path = file_path +'vali_large_img.pt'
    vali_large_img = torch.load(path)
    path = file_path +'vali_regular_img.pt'
    vali_regular_img = torch.load(path)
    path = file_path +'vali_true_img.pt'
    vali_True_img = torch.load(path)
    train_dataset = TensorDataset(train_large_img,train_regular_img,train_True_img)
    train_allstage_data = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True) 
    vali_dataset = TensorDataset(vali_large_img,vali_regular_img,vali_True_img)
    vali_allstage_data = DataLoader(vali_dataset,batch_size=BATCH_SIZE,shuffle=True) 
    return train_allstage_data, vali_allstage_data


def train(config):
    resuming_training = False
    load_epoch = None
    train_data,vali_data = load_data(config["root_path"]+config["file_path"],config["BATCH_SIZE"])
    saved_model_dir = os.path.join(config["root_path"], 'saved_model')
    Gen = model.make_generator(config=config,len_high_size=config["len_size"], scale_list=config["scale_list"])
    Dis = model.make_discriminator(config=config,len_high_size=config["len_size"])
    fit.run_fit(config,Gen, Dis, train_data,vali_data, config["EPOCHS"], config["len_size"], config["scale_list"])


if __name__ == '__main__':


    config = {
    "root_path": os.getcwd(),
    "file_path": '/data/Lee/1mb/16/',
    "scale_list": [0,1],
    "len_size":40,
    "EPOCHS":100,
    "gen_rank1_1st":786,
    "dis_rank1_1st":256,  
    "featureloss_num_layers":3,
    "lr1":0.00001,
    "BATCH_SIZE":128,
    "dis_se_reduction1":128,
    "dis_se_reduction2":4, 
    "gen_se_reduction1":256,
    "gen_se_reduction2":4,
    "weight_list":[0.01, 0.01, 0.4, 1.0, 0.2],
    }
    
    train(config)



