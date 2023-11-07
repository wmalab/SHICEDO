import os
import random
import numpy as np
import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader
from model_loss import model, fit
import gc


# Set seed for reproducibility
def set_seed(seed=43):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False

set_seed()

def make_loss_filter(len_size, batch_size=None, device=None):
    loss_filter = fit.make_loss_filter(len_size)
    if batch_size:
        loss_filter = torch.unsqueeze(torch.from_numpy(np.array(loss_filter)), 0)
        loss_filter = torch.unsqueeze(loss_filter, 0)
        loss_filter = loss_filter.repeat(batch_size, 1, 1, 1)
        loss_filter = loss_filter.to(device, dtype=torch.float)
    return loss_filter

def predict(model_path, ds, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load model
    Generator = model.make_generator(config=config, len_high_size=config["len_size"], scale_list=config["scale_num"])
    generator_optimizer = torch.optim.Adam(Generator.parameters())
    model_state, optimizer_state = torch.load(model_path)
    Generator.load_state_dict(model_state)
    Generator.to(device).float()
    
    batch_size = config["BATCH_SIZE"]
    regular_filter = make_loss_filter(config["len_size"], batch_size, device)
    large_filter = make_loss_filter(config["len_size"] + 4, batch_size, device)
    pred_dic = {}
    for i, (index,large_img,regular_img,True_img) in enumerate(ds):
        if i % 10000 == 0:
                print('pred sample: ',i)
        
        large_img, regular_img, True_img = [img.float().to(device) for img in [large_img, regular_img, True_img]]
        if regular_img.shape[0] != regular_filter.shape[0]:
            temp_regular_filter = make_loss_filter(config["len_size"], regular_img.shape[0], device)
            temp_large_filter = make_loss_filter(config["len_size"] + 4, regular_img.shape[0], device)
            regular_img, True_img, large_img = [torch.mul(img, filter) for img, filter in zip([regular_img, True_img, large_img], [temp_regular_filter, temp_regular_filter, temp_large_filter])]
        else:
            regular_img, True_img, large_img = [torch.mul(img, filter) for img, filter in zip([regular_img, True_img, large_img], [regular_filter, regular_filter, large_filter])]
  
        index = tuple(np.squeeze(index.cpu().detach().numpy()))
        fake_hic_high = Generator(large_img,regular_img)
        out_matrix = np.squeeze(fake_hic_high.cpu().detach().numpy())
        Loss_filter= fit.make_loss_filter(config["len_size"])
        out_matrix = np.multiply(out_matrix, Loss_filter)
        pred_dic[index] = out_matrix
    print('Done predict!!!')
    return  pred_dic 

def load_data(config):
    paths = [os.path.join(config["root_dir"]+config["data_dir"], suffix) for suffix in ['test_index.pt', 'test_large_img.pt', 'test_regular_img.pt', 'test_true_img.pt']]
    return [torch.load(path) for path in paths]

def run(model_path=None, config=None):
    index, large_img, regular_img, True_img = load_data(config)
    test_data = DataLoader(TensorDataset(index, large_img, regular_img, True_img))
    print('Start predict !!!')
    predict_hic_dic = predict(model_path, test_data, config)
    print('Done predict !!!')
    Lr_dic, true_dic, cell_list = {}, {}, []
    for i, (index, _, regular_img, True_img) in enumerate(test_data):
        index = tuple(np.squeeze(index.detach().cpu().numpy()))
        if index[0] not in cell_list:
            cell_list.append(index[0])
        Lr_dic[index] = np.squeeze(regular_img[:, 0, :, :].detach().cpu().numpy())
        true_dic[index] = np.squeeze(True_img.detach().cpu().numpy())
    # Save output
    path = os.path.join('output')
    os.makedirs(path, exist_ok=True)
    for name, dic in zip(['_Lr_dic', '_true_dic', '_predict_dic'], [Lr_dic, true_dic, predict_hic_dic]):
        with open(os.path.join(path, 'test' + name + '.pickle'), 'wb') as handle:
            pickle.dump(dic, handle)
    return Lr_dic, predict_hic_dic, true_dic


if __name__ == '__main__':
    config = {
        "len_size": 40,
        "root_dir": os.getcwd(),
        "data_dir": '/data/Lee/1mb/16/',
        "scale_num": [0, 1],
        "gen_rank1_1st": 786,
        "dis_rank1_1st": 256,
        "featureloss_num_layers": 3,
        "lr1": 0.00001,
        "BATCH_SIZE": 1,
        "dis_se_reduction1": 128,
        "dis_se_reduction2": 4,
        "gen_se_reduction1": 256,
        "gen_se_reduction2": 4,
        "G_loss_w_ssim": 0.5,
        "G_loss_w_bce": 0.1,
        "G_loss_w_mae": 0.5,
        "G_loss_w_gfeature": 0.5,
        "G_loss_w_fm": 1.0,
    }

    Lr_HiC, Hr_pred_HiC, Hr_true_HiC = run(model_path=os.path.join(config["root_dir"], 'pretrained_model', 'gen_checkpoint.pt'),config=config)

