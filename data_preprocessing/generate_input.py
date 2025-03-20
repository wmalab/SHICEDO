import numpy as np
import os
from scHiCTools import scHiCs
import glob
from sklearn.model_selection import train_test_split
import torch
from combine_tensor import Combine_tensor

def divide_pieces_hic(hic_matrix, block_size=40,scales=[0, 1], chromosome_num=22):
    M = hic_matrix
    num_of_group_data = int(np.ceil((M.shape[1]/block_size )))
    num_of_group_data = (num_of_group_data-1)*2-1+1
    data_dic = {}
    for i in range(num_of_group_data):
        data_dic[(chromosome_num,i)]=[]
    size_M = np.arange(len(scales))
    for Q in range(len(scales)):
        if len(scales)!= 1 and Q == 0:
            padding=((0,0),(2,2),(2,2))
            matrix_high = block_size + padding[1][0]*2
            new_M = np.pad(M, padding, mode='constant',constant_values=0)
        else:
            padding=0
            matrix_high = block_size
            new_M = M  
        for i in range(num_of_group_data):
            if i != num_of_group_data -1:
                center_x = int(i*(block_size/2)+(matrix_high/2))
                center_y = int(i*(block_size/2)+(matrix_high/2))
                temp_m = new_M[:, (int(center_x-(matrix_high/2))):int((center_x+(matrix_high/2))) , int((center_y-(matrix_high/2))):int((center_y+(matrix_high/2)))]
                data_dic[(chromosome_num,i)].append(temp_m)
            else:
                matrix_size = new_M.shape[1]
                center_x = matrix_size-int(0.5*matrix_high)
                center_y = matrix_size-int(0.5*matrix_high)
                temp_m = new_M[:,(int(center_x-(matrix_high/2))):int((center_x+(matrix_high/2))) , int((center_y-(matrix_high/2))):int((center_y+(matrix_high/2)))]
                data_dic[(chromosome_num,i)].append(temp_m) 
    return data_dic

def process_sichic(ds_cells,true_cells,resolution_value,resolution_filename,contact_threshold,ds_ratio):
    temp_cell_names_path = 'process_data/Nagano_process/'+resolution_filename+'_con'+contact_threshold+'/'+ds_ratio+'/norm/' + stage + '/'
    cell_names = [s.replace(temp_cell_names_path, '') for s in ds_cells]
    temp_true_cell_names_path = 'process_data/Nagano_process/'+resolution_filename+'_con'+contact_threshold+'/filter_true_norm/' + stage + '/'
    true_cell_names = [s.replace(temp_true_cell_names_path, '') for s in true_cells]
    ds_data = scHiCs(ds_cells,
           reference_genome='mm9',
           resolution=resolution_value,
           format='shortest_score',
           adjust_resolution=True,
           chromosomes='except Y',
           operations=None,
           store_full_map=True)
    ds_dic = ds_data.full_maps      
    M_lr_dic={}
    for key in ds_dic.keys():
        M_lr_dic[key]={}
    for key in ds_dic.keys():
        matrix_list=[]
        for i in range(ds_dic[key].shape[0]):
            Ml = ds_dic[key][i,:,:]
            matrix_list.append(Ml)
        M_lr_dic[key] = dict(zip(cell_names, matrix_list))
    true_data = scHiCs(true_cells,
           reference_genome='mm9',
           resolution=resolution_value,
           format='shortest_score',
           adjust_resolution=True,
           chromosomes='except Y',
           operations=None,
           store_full_map=True)
    true_dic = true_data.full_maps    
    M_hr_dic={}
    for key in true_dic.keys():
        M_hr_dic[key]={}
    for key in true_dic.keys():
        matrix_list=[]
        for i in range(true_dic[key].shape[0]):
            Mh = true_dic[key][i,:,:]
            matrix_list.append(Mh)
        M_hr_dic[key] = dict(zip(true_cell_names, matrix_list))
    return M_lr_dic,M_hr_dic

def split_combine_train_test(M_lr_dic, M_hr_dic,ds_ratio):
    cells_names = list(M_lr_dic['chr1'].keys())
    hr_cells_names = list(M_hr_dic['chr1'].keys())
    hr_cells_names = [s.replace(ds_ratio+'/norm', 'filter_true_norm') for s in hr_cells_names]
    chrom_list = list(M_lr_dic.keys())
    for chrom in chrom_list:
        for i in range(len(cells_names)):       
            M_lr_dic[chrom][cells_names[i]] = np.stack((M_lr_dic[chrom][cells_names[i]], M_hr_dic[chrom][hr_cells_names[i]]), axis=0)
    stage1_train_keys, stage1_test_keys = train_test_split(cells_names, test_size=0.2, random_state=1)
    stage1_train_keys, stage1_val_keys = train_test_split(stage1_train_keys, test_size=0.125, random_state=1) # 0.125 x 0.8 = 0.1\\

    train_dic={}
    test_dic={}
    vali_dic={}
    for chrom in (chrom_list):
        train_dic[chrom] = {}
        train_dic[chrom] = {}
        train_dic[chrom] = {}
        test_dic[chrom] = {}
        test_dic[chrom] = {}
        test_dic[chrom] = {}
        vali_dic[chrom] = {}
        vali_dic[chrom] = {}
        vali_dic[chrom] = {}
        [train_dic[chrom], vali_dic[chrom], test_dic[chrom]] = map(lambda keys: {x: M_lr_dic[chrom][x] for x in keys}, [stage1_train_keys, stage1_val_keys ,stage1_test_keys])
    return train_dic, vali_dic, test_dic

def  divide_mat(block_size,train_dic, vali_dic, test_dic):
    print('start divid train dic: ')
    for chrom in train_dic.keys():
        for cell in train_dic[chrom].keys():
            mat = train_dic[chrom][cell]
            chr_num = chrom.replace('chr', '')
            divid_dic = divide_pieces_hic(mat, block_size=block_size, scales=[0, 1],chromosome_num=chr_num)
            train_dic[chrom][cell]=divid_dic
    print('start divid vali dic: ')
    for chrom in vali_dic.keys():
        for cell in vali_dic[chrom].keys():
            mat = vali_dic[chrom][cell]
            chr_num = chrom.replace('chr', '')
            divid_dic = divide_pieces_hic(mat, block_size=block_size, scales=[0, 1], chromosome_num=chr_num)
            vali_dic[chrom][cell]=divid_dic
    print('start divid test dic: ')
    for chrom in test_dic.keys():
        for cell in test_dic[chrom].keys():
            mat = test_dic[chrom][cell]
            chr_num = chrom.replace('chr', '')
            divid_dic = divide_pieces_hic(mat, block_size=block_size, scales=[0, 1], chromosome_num=chr_num)
            test_dic[chrom][cell]=divid_dic
    return train_dic, vali_dic, test_dic

def pre_dataset(dic,scale_num,len_size):
    count = 0 
    large_img = np.zeros((1,1,len_size+4,len_size+4))
    regular_img = np.zeros((1,1,len_size,len_size))
    True_img = np.zeros((1,1,len_size,len_size))
    new_dic = {}
    for chrom in dic.keys():
        for cell_key in dic[chrom].keys():
            cell = cell_key.split('CDX')[1]
            for index_key in dic[chrom][cell_key].keys():
                if type(index_key) is tuple:
                    new_key = list(index_key)
                    if new_key[0] == 'X':
                        new_key[0] = '20'
                    new_key[0] = float(new_key[0])
                    new_key.insert(0, float(cell))  
                    new_dic[tuple(new_key)] = dic[chrom][cell_key][index_key]
    print('Done change index! ')

    for key,value in new_dic.items():
        if key != 'M_shape_chr1':
            key = torch.tensor(np.asarray(key))
            key = np.expand_dims(np.expand_dims(np.expand_dims(key, axis=0), axis=0), axis=0)
            if count == 0:
                index = key
            else:
                index = np.concatenate((index,key),axis=0)
            value[0] = np.expand_dims(value[0], axis=0)
            value[1] = np.expand_dims(value[1], axis=0)
            large_img_0 = value[0][:,0:1,:,:]
            regular_img_0 = value[1][:,0:1,:,:]
            True_img_0 = value[1][:,1:2,:,:]
            if count == 0:
                large_img = large_img+large_img_0
                regular_img = regular_img+regular_img_0
                True_img = True_img+True_img_0
            else:
                large_img = np.concatenate((large_img,large_img_0),axis=0)
                regular_img = np.concatenate((regular_img,regular_img_0),axis=0)
                True_img = np.concatenate((True_img,True_img_0),axis=0)
            count +=1
    print('Done new dic ! ')   
    return torch.tensor(index),torch.tensor(large_img),torch.tensor(regular_img),torch.tensor(True_img)

def Run_prepare_data(stage,scale_num,block_size, train_div_dic, vali_div_dic, test_div_dic,resolution_filename,contact_threshold,ds_ratio):
    train_index,train_large_img,train_regular_img,train_true_img = pre_dataset(train_div_dic,scale_num,block_size)
    path = 'process_data/Nagano_process/'+resolution_filename+'_con'+contact_threshold+'/'+ds_ratio+'/input_tensor_'+stage
    if not os.path.exists(path):
        os.makedirs(path)
    save_tensor_path = path +'/'+'train_index.pt'
    torch.save(train_index, save_tensor_path)
    save_tensor_path = path +'/'+'train_large_img.pt'
    torch.save(train_large_img, save_tensor_path)
    save_tensor_path = path +'/'+'train_regular_img.pt'
    torch.save(train_regular_img, save_tensor_path)
    save_tensor_path = path +'/'+'train_true_img.pt'
    torch.save(train_true_img, save_tensor_path)
    vali_index,vali_large_img,vali_regular_img,vali_true_img = pre_dataset(vali_div_dic,scale_num,block_size)
    path = 'process_data/Nagano_process/'+resolution_filename+'_con'+contact_threshold+'/'+ds_ratio+'/input_tensor_'+stage
    if not os.path.exists(path):
        os.makedirs(path)
    save_tensor_path = path +'/'+'vali_index.pt'
    torch.save(vali_index, save_tensor_path)
    save_tensor_path = path +'/'+'vali_large_img.pt'
    torch.save(vali_large_img, save_tensor_path)
    save_tensor_path = path +'/'+'vali_regular_img.pt'
    torch.save(vali_regular_img, save_tensor_path)
    save_tensor_path = path +'/'+'vali_true_img.pt'
    torch.save(vali_true_img, save_tensor_path)
    test_index,test_large_img,test_regular_img,test_true_img = pre_dataset(test_div_dic,scale_num,block_size)
    path = 'process_data/Nagano_process/'+resolution_filename+'_con'+contact_threshold+'/'+ds_ratio+'/input_tensor_'+stage
    if not os.path.exists(path):
        os.makedirs(path)
    save_tensor_path = path +'/'+'test_index.pt'
    torch.save(test_index, save_tensor_path)
    save_tensor_path = path +'/'+'test_large_img.pt'
    torch.save(test_large_img, save_tensor_path)
    save_tensor_path = path +'/'+'test_regular_img.pt'
    torch.save(test_regular_img, save_tensor_path)
    save_tensor_path = path +'/'+'test_true_img.pt'
    torch.save(test_true_img, save_tensor_path)

if __name__ == '__main__':
    stage_list = ['stage1','stage2','stage3','stage4']
    resolution_filename = '1mb'
    contact_threshold = '310000'
    ds_ratio = 'ds_9'
    resolution = 1000000
    block_size = 40
    scale_num = [0,1]
    for stage in stage_list:
        print('start stage: ',stage)
        Ds_cells = sorted(glob.glob('process_data/Nagano_process/'+resolution_filename+'_con'+contact_threshold+'/'+ds_ratio+'/norm/'+ stage +'/*'))
        ds_cells = Ds_cells
        True_cells = sorted(glob.glob('process_data/Nagano_process/'+resolution_filename+'_con'+contact_threshold+'/filter_true_no_inter/filter_true_no_inter_norm/'+ stage +'/*'))
        true_cells = True_cells
        M_lr_dic,M_hr_dic = process_sichic(ds_cells,true_cells,resolution,resolution_filename,contact_threshold,ds_ratio)
        train_dic, vali_dic, test_dic = split_combine_train_test(M_lr_dic, M_hr_dic,ds_ratio)
        train_div_dic, vali_div_dic, test_div_dic = divide_mat(block_size,train_dic, vali_dic, test_dic)
        Run_prepare_data(stage,scale_num,block_size, train_div_dic, vali_div_dic, test_div_dic,resolution_filename,contact_threshold,ds_ratio)
        print('Done stage: ',stage)
    Combine_tensor('process_data/Nagano_process/'+resolution_filename+'_con'+contact_threshold+'/'+ds_ratio+'/input_tensor_')
    