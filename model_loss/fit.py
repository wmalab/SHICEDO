import os
import torch 
import numpy as np
from model_loss import model, Loss_functions
import random
# Seed Setting
seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False

def make_loss_filter(len_high_size):
    """Creates a loss filter for a given size."""
    loss_filter = np.ones(shape=(len_high_size, len_high_size)) - np.diag(np.ones(shape=(len_high_size,)))
    return loss_filter

def run_fit(config, gen, dis, train_data, vali_data, epochs, len_high_size, scale, valid_dataset=None):
    """Main training loop for the model."""
    
    # Device Configuration
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Model & Optimizer Configuration
    generator_optimizer = torch.optim.Adam(gen.parameters(), lr=config["lr1"])
    discriminator_optimizer = torch.optim.Adam(dis.parameters(), lr=0.000001)
    gen, dis = gen.to(device, dtype=torch.float).float(), dis.to(device, dtype=torch.float).float()

    # Loss Filter Configuration
    Loss_filter = make_loss_filter(len_high_size)
    large_filter = make_loss_filter(len_high_size + 4)
    regular_filter = torch.unsqueeze(torch.from_numpy(Loss_filter), 0).unsqueeze(0).repeat(config['BATCH_SIZE'], 1, 1, 1).to(device, dtype=torch.float)
    large_filter = torch.unsqueeze(torch.from_numpy(large_filter), 0).unsqueeze(0).repeat(config['BATCH_SIZE'], 1, 1, 1).to(device, dtype=torch.float)

    torch.cuda.empty_cache()
    
    # Training Loop
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        Gen_loss_weight = [config['weight_list'][i] for i in [4, 0, 3, 2, 1]]
        Gen_loss, G_bce_h, G_mse_h, G_ssim_h, G_feature, Fm_loss, Dis_loss = [], [], [], [], [], [], []

        for i, (large_img, regular_img, True_img) in enumerate(train_data):
            # if i % 10 == 0:
            #     print(f'data group #: {i}')
            # Prepare Data
            large_img, regular_img, True_img = large_img.to(device, dtype=torch.float), regular_img.to(device, dtype=torch.float), True_img.to(device, dtype=torch.float)

            # Handle varying batch sizes
            if regular_img.shape[0] != regular_filter.shape[0]:
                temp_regular_filter, temp_large_filter = custom_filters(regular_img.shape[0], len_high_size, device)
                regular_img, True_img, large_img = torch.mul(regular_img, temp_regular_filter), torch.mul(True_img, temp_regular_filter), torch.mul(large_img, temp_large_filter)
            else:
                regular_img, True_img, large_img = torch.mul(regular_img, regular_filter), torch.mul(True_img, regular_filter), torch.mul(large_img, large_filter)

            matrix_all = [large_img, regular_img, True_img]

            # Generator Optimization
            generator_optimizer.zero_grad()
            g_ssim_h, g_bce_h, g_mse_h, g_feature, fm_loss = Loss_functions.generator_loss(Gen=gen, Dis=dis, img_all=matrix_all, config=config)
            generator_total_loss = sum([loss * weight for loss, weight in zip([g_ssim_h, g_bce_h, g_mse_h, g_feature, fm_loss], Gen_loss_weight)])
            generator_total_loss.backward()
            generator_optimizer.step()

            # Discriminator Optimization
            discriminator_optimizer.zero_grad()
            dis_loss = Loss_functions.discriminator_loss(Gen=gen, Dis=dis, img_all=matrix_all)
            dis_loss.backward()
            discriminator_optimizer.step()

            # Record Losses
            Gen_loss.append(generator_total_loss.cpu().detach().numpy())
            G_ssim_h.append(g_ssim_h.cpu().detach().numpy())
            G_bce_h.append(g_bce_h.cpu().detach().numpy())
            G_mse_h.append(g_mse_h.cpu().detach().numpy())
            G_feature.append(g_feature.cpu().detach().numpy())
            Fm_loss.append(fm_loss.cpu().detach().numpy())
            Dis_loss.append(dis_loss.cpu().detach().numpy())
        
        train_matrix_loss = np.nanmean(G_ssim_h)+np.nanmean(G_mse_h)+np.nanmean(G_feature)
        print('Matrix training loss: ',train_matrix_loss)

        # Validation Loop
        avg_val_mse_loss, avg_val_feature_loss, avg_val_SSIM_loss = validate_model(vali_data, gen, dis, config, len_high_size, regular_filter, large_filter, device)
        vali_matrix_loss = avg_val_mse_loss+avg_val_feature_loss+avg_val_SSIM_loss
        
        os.makedirs("trained_model", exist_ok=True)
        if epoch == 0:
            best_vali_loss = vali_matrix_loss
            torch.save((gen.state_dict(), generator_optimizer.state_dict()), "trained_model/gen_checkpoint.pt")
            torch.save((dis.state_dict(), discriminator_optimizer.state_dict()), "trained_model/dis_checkpoint.pt")
        elif vali_matrix_loss<=best_vali_loss:
            torch.save((gen.state_dict(), generator_optimizer.state_dict()), "trained_model/gen_checkpoint.pt")
            torch.save((dis.state_dict(), discriminator_optimizer.state_dict()), "trained_model/dis_checkpoint.pt")
    print("Finished Training!")

def custom_filters(batch_size, len_high_size, device):
    """Create filters for varying batch sizes."""
    temp_Loss_filter = make_loss_filter(len_high_size)
    temp_regular_filter = torch.unsqueeze(torch.from_numpy(temp_Loss_filter), 0).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device, dtype=torch.float)
    temp_large_loss_filter = make_loss_filter(len_high_size + 4)
    temp_large_filter = torch.unsqueeze(torch.from_numpy(temp_large_loss_filter), 0).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device, dtype=torch.float)
    return temp_regular_filter, temp_large_filter

    
def validate_model(vali_data, gen, dis, config, len_high_size, regular_filter, large_filter, device):
    """Evaluate the model on the validation set."""
    val_mse_loss, val_feature_loss, val_SSIM_loss, val_steps = 0, 0, 0, 0

    for i, (large_img, regular_img, True_img) in enumerate(vali_data):
        with torch.no_grad():
            
            # Move data to the specified device
            large_img, regular_img, True_img = large_img.float().to(device), regular_img.float().to(device), True_img.float().to(device)

            # Check for batch size consistency
            if regular_img.shape[0] != regular_filter.shape[0]:
                temp_regular_filter, temp_large_filter = custom_filters(regular_img.shape[0], len_high_size, device)
                regular_img = torch.mul(regular_img, temp_regular_filter)
                True_img = torch.mul(True_img, temp_regular_filter)
                large_img = torch.mul(large_img, temp_large_filter)
            else:
                regular_img = torch.mul(regular_img, regular_filter)
                True_img = torch.mul(True_img, regular_filter)
                large_img = torch.mul(large_img, large_filter)

            matrix_all = [large_img, regular_img, True_img]

            # Compute losses using your defined loss functions (assuming they are similar to the training loop)
            g_ssim_h, g_bce_h, g_mse_h, g_feature, fm_loss = Loss_functions.generator_loss(Gen=gen, Dis=dis, img_all=matrix_all, config=config)

            # Accumulate the losses
            val_mse_loss += g_mse_h.cpu().numpy()
            val_feature_loss += g_feature.cpu().numpy()
            val_SSIM_loss += g_ssim_h.cpu().numpy()
            val_steps += 1

    # Compute average losses
    avg_val_mse_loss = val_mse_loss / val_steps
    avg_val_feature_loss = val_feature_loss / val_steps
    avg_val_SSIM_loss = val_SSIM_loss / val_steps

    return avg_val_mse_loss, avg_val_feature_loss, avg_val_SSIM_loss

