import torch
import torch.nn as nn
import random
import numpy as np
from torchvision import models
from torchmetrics import StructuralSimilarityIndexMeasure
from model_loss.autoencoder import ConvAutoencoder
# Seed
seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False

def get_encoder(model,layer_num):
    if layer_num == 2:
        new_model = nn.Sequential(
              model.conv1,
              model.norm_1,
              model.se_1,
              nn.ReLU(),
              model.pool,
              model.conv2,
              model.norm_2,
              model.se_2,
              nn.ReLU(),
              model.pool,
              model.conv4,
              model.norm_4,
              model.se_4,
              nn.ReLU(),)
    elif layer_num == 1:
        new_model = nn.Sequential(
              model.conv1,
              model.norm_1,
              model.se_1,
              nn.ReLU(),
              model.pool,
              model.conv2,
              model.norm_2,
              model.se_2,
              nn.ReLU(),)
    elif layer_num == 0:
        new_model = nn.Sequential(
              model.conv1,
              model.norm_1,
              model.se_1,
              nn.ReLU(),)
    else: 
        new_model = nn.Sequential(
              model.conv1,
              model.norm_1,
              model.se_1,
              nn.ReLU(),)
    return new_model


def cal_autoencoder_feature_loss(filtered_fake_hic_high,filtered_real_hic_high,num_layer,path):
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            gen = nn.DataParallel(gen)
            dis = nn.DataParallel(dis)
    model = ConvAutoencoder()
    model_weights_path = path+'/model_loss/autoencoder_saved_model/model/model_checkpoint.pt'
    checkpoint = torch.load(model_weights_path)
    model.load_state_dict(checkpoint[0])
    f_loss_sum = 0
    mae = torch.nn.L1Loss(reduction='mean')
    for i in range(num_layer):
        encoder = get_encoder(model,num_layer)
        encoder = encoder.to(device)
        f_fake = encoder(filtered_fake_hic_high)
        f_real = encoder(filtered_real_hic_high)
        f_loss_sum += mae(f_fake,f_real)
    return f_loss_sum    

def discriminator_bce_loss(real_output, fake_output):
    loss_object = torch.nn.BCEWithLogitsLoss()
    real_loss = loss_object(real_output, torch.ones_like(real_output))
    generated_loss = loss_object(fake_output, torch.zeros_like(fake_output))
    total_disc_loss = real_loss + generated_loss
    if torch.isnan(total_disc_loss):
        print('dis loss nan!!!!! ',total_disc_loss)
    return total_disc_loss

def generator_bce_loss(d_pred):
    loss_object = torch.nn.BCEWithLogitsLoss()
    gan_loss = loss_object(d_pred, torch.ones_like(d_pred))
    if torch.isnan(gan_loss):
        print('gen loss nan!!!!! ',gan_loss)
    return gan_loss

def generator_ssim_loss(y_pred, y_true):  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    SSIM_loss = StructuralSimilarityIndexMeasure(kernel_size=11).to(device,dtype=torch.float)
    ssim = SSIM_loss(y_pred, y_true) 
    ssimloss = (1-ssim)/2
    return ssimloss

def generator_mse_loss(y_pred, y_true):  
    mse = torch.nn.MSELoss(reduction='mean')
    mseloss = mse(y_pred, y_true) 
    return mseloss

def generator_mae_loss(y_pred, y_true):  
    mae = torch.nn.L1Loss(reduction='mean')
    maeloss = mae(y_pred, y_true) 
    return maeloss

def generator_loss(Gen, Dis, img_all,config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Gen = Gen.to(device, dtype=torch.float)
    Dis = Dis.to(device, dtype=torch.float)
    fake_hic_high = Gen(img_all[0],img_all[1])
    real_hic_high = img_all[-1].clone()
    feature_loss = cal_autoencoder_feature_loss(fake_hic_high,real_hic_high,num_layer=config["featureloss_num_layers"],path=config["root_path"])
    with torch.no_grad():
        fake_dis_pred,fake_v,fake_out2,fake_out3 = Dis(fake_hic_high)
        real_dis_pred,real_v,real_out2,real_out3 = Dis(real_hic_high)
    v_loss = generator_mae_loss(fake_v, real_v)
    out2_loss = generator_mae_loss(fake_out2, real_out2)
    out3_loss = generator_mae_loss(fake_out3, real_out3)
    fm_loss = v_loss + out2_loss + out3_loss
    bce_loss_high_0 = generator_bce_loss(fake_dis_pred)
    mse_loss_high_1 = generator_mae_loss(fake_hic_high, real_hic_high)
    SSIM_loss = generator_ssim_loss(fake_hic_high, real_hic_high)
    return SSIM_loss.float(),bce_loss_high_0.float(), mse_loss_high_1.float(), feature_loss.float(), fm_loss.float()

def discriminator_loss(Gen, Dis, img_all):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        fake_hic_high = Gen(img_all[0],img_all[1])
    fake_hic_high = fake_hic_high.to(device,dtype=torch.float)
    real_hic_high = img_all[-1].clone()
    with torch.autocast(device_type="cuda"):
        disc_generated_output,_, _, _ = Dis(fake_hic_high)
        disc_real_output,_, _, _ = Dis(real_hic_high)
        loss = discriminator_bce_loss(disc_real_output, disc_generated_output)
    return loss

