import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Seed
seed = 43
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer_1d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class Weight_R1M(nn.Module):
    def __init__(self, input_shape):
        super(Weight_R1M, self).__init__()
        w_init_empty = torch.empty((1,input_shape[1], 1, 1),dtype=torch.float32)
        w_init = nn.init.uniform_(w_init_empty, a=0.0, b=4.0)
        self.weight = torch.nn.Parameter(w_init, requires_grad=True)
    def forward(self, input):
        self.weight = nn.Parameter(nn.ReLU()(self.weight))
        return torch.mul(input, self.weight)

class Normal(nn.Module):
    def __init__(self, input_dim):
        super(Normal, self).__init__()
        w_init_empty = torch.empty((1, 1,input_dim, 1),dtype=torch.float32)
        w_init = torch.nn.init.ones_(w_init_empty)
        self.weight = torch.nn.Parameter(w_init, requires_grad=True)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def forward(self, input):
        rowsr = torch.sqrt(torch.sum(torch.mul(input,input), 2, keepdim = True) + torch.tensor(1e-6, dtype=torch.float32))
        colsr = torch.sqrt(torch.sum(torch.mul(input,input), 3, keepdim = True) + torch.tensor(1e-6, dtype=torch.float32))
        sumele = torch.mul(rowsr, colsr)
        Div = torch.where((sumele>0)|(sumele<0), torch.div(input,sumele), torch.tensor(0.).to(self.device))
        self.weight = nn.Parameter(nn.ReLU()(self.weight))
        WT = torch.permute(self.weight, (0, 1, 3, 2))
        M = torch.mul(self.weight, WT)
        out = torch.mul(Div, M)
        return out

class Rank1(nn.Module):
    def __init__(self,in_channels,out_channels,len_size,Weight_R1M_input_shape,rank1_ker,reduction_num):
        super(Rank1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,[1, rank1_ker],stride=(1, 1), padding='valid', bias=True)
        self.conv1_0 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=True)
        torch.nn.init.normal_(self.conv1_0.weight, mean=0.01, std=0.1)
        torch.nn.utils.spectral_norm(self.conv1_0)
        self.se_0 = SELayer_1d(out_channels,reduction=reduction_num)
        self.conv1_1 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=True)
        torch.nn.init.normal_(self.conv1_1.weight, mean=0.01, std=0.1)
        torch.nn.utils.spectral_norm(self.conv1_1)
        self.se_1 = SELayer_1d(out_channels,reduction=reduction_num)
        self.weight_R1M = Weight_R1M(Weight_R1M_input_shape)

    def forward(self,inp):
        out = self.conv1(inp.float())
        v = nn.ReLU()(out)
        v_0 = torch.squeeze(v,dim=3)
        v = self.conv1_0(v_0)
        v = self.se_0(v)
        v = nn.ReLU()(v)
        v = v + v_0
        v= self.conv1_1(v)
        v = self.se_1(v)
        v = nn.ReLU()(v)
        v = v + v_0
        v = torch.unsqueeze(v, 3)
        v = self.weight_R1M(v)
        vt = torch.permute(v, (0,1, 3, 2))
        return v,vt

class reconstruction(nn.Module):
    def __init__(self, channels_decompose, len_size):
        super(reconstruction, self).__init__()
        w_init_empty = torch.empty((1,channels_decompose, 1, 1),dtype=torch.float32)
        w_init = torch.nn.init.ones_(w_init_empty)
        self.weight = torch.nn.Parameter(w_init, requires_grad=True)
    def forward(self,v,vt):
        recon_matrix = torch.mul(torch.mul(v, vt), self.weight)
        return recon_matrix

class skip_con(nn.Module):
    def __init__(self,skip_con_1_input_channels,skip_con_1_out_channels,
                skip_con_1_kernel_size,skip_con_1_stride,
                skip_con_2_input_channels,skip_con_2_out_channels,
                skip_con_2_kernel_size,padding,skip_con_2_stride,reduction_num):
        super(skip_con, self).__init__()
        self.conv_layer_1 = nn.Conv2d(skip_con_1_input_channels,skip_con_1_out_channels,kernel_size=skip_con_1_kernel_size,
                                    stride=skip_con_1_stride, padding=padding, bias=True)
        torch.nn.init.normal_(self.conv_layer_1.weight, mean=0.01, std=0.1)
        torch.nn.utils.spectral_norm(self.conv_layer_1)
        self.se_1 = SELayer(skip_con_1_out_channels,reduction=reduction_num)
        self.conv_layer_2 = nn.Conv2d(skip_con_2_input_channels,skip_con_2_out_channels,kernel_size=skip_con_2_kernel_size,
                                    stride=skip_con_2_stride, padding=padding, bias=True)
        torch.nn.init.normal_(self.conv_layer_2.weight, mean=0.01, std=0.1)
        torch.nn.utils.spectral_norm(self.conv_layer_2)
        self.se_2 = SELayer(skip_con_2_out_channels,reduction=reduction_num)
        
    def forward(self,inp,inp_skip):
        out = self.conv_layer_1(inp)
        out = self.se_1(out)
        out = nn.ReLU()(out)
        out = self.conv_layer_2(out)
        out = self.se_2(out)
        out = nn.ReLU()(out)
        skip_con = out + inp_skip[:,0:1,:,:]
        return skip_con

class loop_block_45123(nn.Module):
    def __init__(self,Dic):
        super(loop_block_45123, self).__init__()
        self.rank1_layer = Rank1(Dic['rank1_1st_channel_in'],Dic['rank1_1st_channel_out'],Dic['block_size'],Dic['Weight_R1M_input_shape'],Dic['small_rank1_ker'],Dic['gen_se_reduction1'])
        self.se_rank1 = SELayer(Dic['rank1_1st_channel_out'],reduction=Dic['gen_se_reduction2'])
        self.reconstruction_layer = reconstruction(Dic['reconstruction_channel'],Dic['block_size'])
        self.se_recon = SELayer(Dic['reconstruction_channel'],reduction=Dic['gen_se_reduction2'])
        self.skip_con_layer = skip_con( Dic['skip_con_1_in_channels'],
                                        Dic['skip_con_1_out_channels'],
                                        Dic['skip_con_1_kernel_size'],
                                        Dic['skip_con_1_stride'],
                                        Dic['skip_con_2_input_channels'],
                                        Dic['skip_con_2_out_channels'],
                                        Dic['skip_con_2_kernel_size'],
                                        Dic['regular_skip_con_padding'],
                                        Dic['skip_con_2_stride'],
                                        Dic['gen_se_reduction2'])
    def forward(self,previous_sacle_out,inp_next_lvl):
        v,vt=self.rank1_layer(inp_next_lvl)
        v = self.se_rank1(v)
        vt = self.se_rank1(vt)
        low_img=self.reconstruction_layer(v,vt)
        low_img = self.se_recon(low_img)
        out = self.skip_con_layer(low_img,inp_next_lvl)
        final_out = torch.cat((previous_sacle_out,out),dim=1) 
        return final_out

class generator(nn.Module):
    def __init__(self,Dic):
        super(generator, self).__init__()
        self.data_list_len = len(Dic['scales_list']) 
        self.rank1_layer = Rank1(Dic['rank1_1st_channel_in'],Dic['rank1_1st_channel_out'],Dic['block_size'],Dic['Weight_R1M_input_shape'],Dic['larg_rank1_ker'],Dic['gen_se_reduction1'])
        self.se_rank1 = SELayer(Dic['rank1_1st_channel_out'],reduction=Dic['gen_se_reduction2'])
        self.reconstruction_layer = reconstruction(Dic['reconstruction_channel'],Dic['block_size'])
        self.se_recon = SELayer(Dic['reconstruction_channel'],reduction=Dic['gen_se_reduction2'])
        self.skip_con_layer = skip_con(Dic['skip_con_1_in_channels'],
                                        Dic['skip_con_1_out_channels'],
                                        Dic['skip_con_1_kernel_size'],
                                        Dic['skip_con_1_stride'],
                                        Dic['skip_con_2_input_channels'],
                                        Dic['skip_con_2_out_channels_pre_loop'],
                                        Dic['skip_con_2_kernel_size'],
                                        Dic['large_skip_con_padding'],
                                        Dic['skip_con_2_stride'],
                                        Dic['gen_se_reduction2'])
        self.loop_layers = nn.ModuleList()
        for i in range(len(Dic['scales_list']) - 1):
            loop_block = loop_block_45123(Dic)
            self.loop_layers.append(loop_block)
        self.final_cov_layer0 = nn.Conv2d(Dic['final_con_0_chanel_in'],Dic['final_con_0_chanel_out'],3, stride=1, padding='same',bias=True)
        torch.nn.init.normal_(self.final_cov_layer0.weight, mean=0.01, std=0.1)
        torch.nn.utils.spectral_norm(self.final_cov_layer0)
        self.se_0 = SELayer(Dic['final_con_0_chanel_out'],reduction=Dic['gen_se_reduction2'])
        self.final_cov_layer1 = nn.Conv2d(Dic['final_con_1_chanel_in'],Dic['final_con_1_chanel_out'],3, stride=1, padding='same',bias=True)
        torch.nn.init.normal_(self.final_cov_layer1.weight, mean=0.01, std=0.1)
        torch.nn.utils.spectral_norm(self.final_cov_layer1)
        self.se_1 = SELayer(Dic['final_con_1_chanel_out'],reduction=Dic['gen_se_reduction2'])
        self.final_cov_layer3 = nn.Conv2d(Dic['final_con_3_chanel_in'],Dic['final_con_3_chanel_out'],[1, 1], stride=(1, 1), padding='same')
        torch.nn.init.normal_(self.final_cov_layer3.weight, mean=0.01, std=0.1)       

    def forward(self,large_img,regular_img):
        large_img = large_img[:,0:1,:,:]
        regular_img = regular_img[:,0:1,:,:]
        v,vt = self.rank1_layer(large_img)    
        v = self.se_rank1(v)
        vt = self.se_rank1(vt)
        out = self.reconstruction_layer(v,vt)
        out = self.se_recon(out)
        out = self.skip_con_layer(out,regular_img)
        for i in range(len(self.loop_layers)):
            out = self.loop_layers[i](out,regular_img)
        out = self.final_cov_layer0(out)
        out = self.se_0(out)
        out = nn.ReLU()(out)
        out = self.final_cov_layer1(out)
        out = self.se_1(out)
        out = nn.ReLU()(out)       
        out = self.final_cov_layer3(out)
        out = nn.ReLU()(out)
        out_t = torch.transpose(out, 2, 3)
        result = 0.5*(torch.where(out==0,out+out_t,out) + torch.where(out_t==0,out+out_t,out_t))
        return result

def make_generator(config,len_high_size, scale_list):
    len_high_size=config["len_size"]
    Dic=dict(block_size = len_high_size,
            large_block_size = len_high_size+4,
            scales_list = scale_list,
            Weight_R1M_input_shape = [40,config["gen_rank1_1st"],100,1],
            up_sampl_ratio = 2,
            rank1_1st_channel_in = 1,
            gen_se_reduction1 = config["gen_se_reduction1"], 
            gen_se_reduction2 = config["gen_se_reduction2"], 
            rank1_1st_channel_out = config["gen_rank1_1st"],
            larg_rank1_ker = len_high_size+4,
             small_rank1_ker = len_high_size,
            reconstruction_channel = config["gen_rank1_1st"],
            large_skip_con_padding = 'valid',
            regular_skip_con_padding = 'same',
            skip_con_1_in_channels = config["gen_rank1_1st"],
            skip_con_1_out_channels = config["gen_rank1_1st"],
            skip_con_1_kernel_size = 3,
            skip_con_1_stride = (1, 1),
            skip_con_2_input_channels = config["gen_rank1_1st"],
            skip_con_2_out_channels = config["gen_rank1_1st"],
            skip_con_2_out_channels_pre_loop = config["gen_rank1_1st"],
            skip_con_2_kernel_size = 3,
            skip_con_2_stride = (1, 1),
            upsample_channel_in = config["gen_rank1_1st"],
            upsample_channel_out = config["gen_rank1_1st"],
            normal_chanel = config["gen_rank1_1st"],
            final_con_0_chanel_in = int(2*config["gen_rank1_1st"]),
            final_con_0_chanel_out = 128,
            final_con_1_chanel_in = 128,
            final_con_1_chanel_out = 32,
            final_con_2_chanel_in = 32,
            final_con_2_chanel_out = 8,
            final_con_3_chanel_in = 32,
            final_con_3_chanel_out = 1)
    Gen = generator(Dic) 
    return Gen


class Rank1_dis(nn.Module):
    def __init__(self,in_channels,out_channels,len_size,Weight_R1M_input_shape,rank1_ker,reduction_num):
        super(Rank1_dis, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,[1, rank1_ker],stride=(1, 1), padding='valid', bias=True)
        self.conv1_0 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=True)
        torch.nn.init.normal_(self.conv1_0.weight, mean=0.01, std=0.1)
        torch.nn.utils.spectral_norm(self.conv1_0)
        self.se_0 = SELayer_1d(out_channels,reduction=reduction_num)
        self.conv1_1 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=True)
        torch.nn.init.normal_(self.conv1_1.weight, mean=0.01, std=0.1)
        torch.nn.utils.spectral_norm(self.conv1_1)
        self.se_1 = SELayer_1d(out_channels,reduction=reduction_num)
        self.weight_R1M = Weight_R1M(Weight_R1M_input_shape)

    def forward(self,inp):
        out = self.conv1(inp.float())
        v = nn.LeakyReLU(0.01, inplace=False)(out)
        v_0 = torch.squeeze(v,dim=3)
        v = self.conv1_0(v_0)
        v = self.se_0(v)
        v = nn.LeakyReLU(0.01, inplace=False)(v)
        v = v + v_0
        v= self.conv1_1(v)
        v = self.se_1(v)
        v = nn.LeakyReLU(0.01, inplace=False)(v)
        v = v + v_0
        v = torch.unsqueeze(v, 3)
        v = self.weight_R1M(v)
        vt = torch.permute(v, (0,1, 3, 2))
        return v,vt

class discriminator(nn.Module):
    def __init__(self,Dic):
        super(discriminator, self).__init__()
        self.rank1_layer = Rank1_dis(Dic['rank1_1_channel_in'],Dic['rank1_1_channel_out'],Dic['block_size'],Dic['Weight_R1M_input_shape'],Dic['rank1_ker'],Dic['dis_se_reduction1'])
        self.se_rank1 = SELayer(Dic['rank1_1_channel_out'],reduction=Dic['dis_se_reduction2'])
        self.reconstruction_layer = reconstruction(Dic['reconstruction_channel'],Dic['block_size'])
        self.se_recon = SELayer(Dic['reconstruction_channel'],reduction=Dic['dis_se_reduction2'])
        self.conv1d_layer1 = nn.Conv2d(Dic['conv1d_1_channel_in'],Dic['conv1d_1_channel_out'], kernel_size = 3, stride=1, padding='same',bias=True)
        torch.nn.init.normal_(self.conv1d_layer1.weight, mean=0.01, std=0.1)
        torch.nn.utils.spectral_norm(self.conv1d_layer1)
        self.se_1 = SELayer(Dic['conv1d_1_channel_out'],reduction=Dic['dis_se_reduction2'])
        self.conv1d_layer2 = nn.Conv2d(Dic['conv1d_2_channel_in'],Dic['conv1d_2_channel_out'], kernel_size = 3, stride=1, padding='same',bias=True)
        torch.nn.init.normal_(self.conv1d_layer2.weight, mean=0.01, std=0.1)
        torch.nn.utils.spectral_norm(self.conv1d_layer2)
        self.se_2 = SELayer(Dic['conv1d_2_channel_out'],reduction=Dic['dis_se_reduction2'])
        self.flatten_layer = nn.Flatten(start_dim=1)
        self.last_dense_layer = torch.nn.Linear(Dic['dense_layer_in'], 1,bias=False,dtype=torch.float32)

    def forward(self,inp):
        v,vt = self.rank1_layer(inp)
        v = self.se_rank1(v)
        vt = self.se_rank1(vt)
        out1 = self.reconstruction_layer(v,vt)
        out1 = self.se_recon(out1)
        out2 = self.conv1d_layer1(out1)
        out2 = self.se_1(out2)
        out2 = nn.LeakyReLU(0.01, inplace=False)(out2)
        skip_con = out1 + out2
        out3 = self.conv1d_layer2(skip_con)
        out3 = self.se_2(out3)
        out3 = nn.LeakyReLU(0.01, inplace=False)(out3)
        out4 = self.flatten_layer(out3)
        last = self.last_dense_layer(out4)
        return last,v,out2,out3


def make_discriminator(config,len_high_size):
    Dic_dis=dict(block_size = len_high_size,
            rank1_1_channel_in = 1,
            rank1_ker = len_high_size,
            dis_se_reduction1 = config["dis_se_reduction1"],
            dis_se_reduction2 = config["dis_se_reduction2"],
            Weight_R1M_input_shape = [40,config["dis_rank1_1st"],100,1],
            reconstruction_channel = config["dis_rank1_1st"],
            rank1_1_channel_out = config["dis_rank1_1st"],
            conv1d_1_channel_in = config["dis_rank1_1st"],
            conv1d_1_channel_out = config["dis_rank1_1st"],
            conv1d_2_channel_in = config["dis_rank1_1st"],
            conv1d_2_channel_out = config["dis_rank1_1st"],
            conv1d_3_channel_in = config["dis_rank1_1st"],
            conv1d_3_channel_out = config["dis_rank1_1st"],
            conv1d_4_channel_in = config["dis_rank1_1st"],
            conv1d_4_channel_out = config["dis_rank1_1st"],
            dense_layer_in = int(1600*config["dis_rank1_1st"]))
    Dis = discriminator(Dic_dis) 
    return Dis

