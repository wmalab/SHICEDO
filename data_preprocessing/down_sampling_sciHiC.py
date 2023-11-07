import numpy as np
import pickle
import os
from scHiCTools import scHiCs

def sampling_hic(hic_matrix, sampling_ratio, fix_seed=False):
    """sampling dense hic matrix"""
    m = np.matrix(hic_matrix)
    all_sum = m.sum(dtype='float')
    idx_prob = np.divide(m, all_sum, out=np.zeros_like(m), where=all_sum != 0)
    idx_prob = np.asarray(idx_prob.reshape(
        (idx_prob.shape[0]*idx_prob.shape[1],)))
    idx_prob = np.squeeze(idx_prob)
    sample_number_counts = int(all_sum/(2*sampling_ratio))
    id_range = np.arange(m.shape[0]*m.shape[1])
    if fix_seed:
        np.random.seed(0)
    id_x = np.random.choice(
        id_range, size=sample_number_counts, replace=True, p=idx_prob)
    sample_m = np.zeros_like(m)
    for i in np.arange(sample_number_counts):
        x = int(id_x[i]/m.shape[0])
        y = int(id_x[i] % m.shape[0])
        sample_m[x, y] += 1.0
    sample_m = np.transpose(sample_m) + sample_m
    return np.asarray(sample_m)

def process_and_down_sampleing_HiC(selected_filename,down_sampling_ratio,matrix_resolution,stage_num,resolution,cell_contact_threshold_num):
    path = 'process_data/Nagano_process/'+resolution+'_con'+cell_contact_threshold_num+'/ds_9/stage'+str(stage_num)
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    ds_filename = [s.replace('single_cell/Nagano/1CDX_cells', 'process_data/Nagano_process/'+resolution+'_con'+cell_contact_threshold_num+'/ds_9/stage'+str(stage_num)) for s in selected_filename]
    ds_filename = [s.replace('/new_adj', '') for s in ds_filename]
    loaded_data = scHiCs(selected_filename,
           reference_genome='mm9',
           resolution=matrix_resolution,
           format='shortest_score',
           adjust_resolution=True,
           chromosomes='except Y',
           operations=None,
           store_full_map=True)
    raw_dic = loaded_data.full_maps  

    processed_list={}
    for key in raw_dic.keys():
        processed_list[key]={}
        for i in range(len(ds_filename)):
            processed_list[key][ds_filename[i]]={}
       
    Mh_count,Ml_count=0,0
    for key in raw_dic.keys():
        matrix_list=[]
        for i in range(raw_dic[key].shape[0]):
            Mh = raw_dic[key][i,:,:]
            Ml = sampling_hic(raw_dic[key][i,:,:], sampling_ratio=down_sampling_ratio, fix_seed=True)
            Mh_diag_v = list(np.diagonal(Mh))
            np.fill_diagonal(Mh, 0)
            Mh_mat_sum = np.sum(Mh)
            Mh_diag_sum = np.sum(Mh_diag_v)
            Mh_count += (Mh_mat_sum/2) + Mh_diag_sum
            Ml_diag_v = list(np.diagonal(Ml))
            np.fill_diagonal(Ml, 0)
            Ml_mat_sum = np.sum(Ml)
            Ml_diag_sum = np.sum(Ml_diag_v)
            Ml_count += (Ml_mat_sum/2) + Ml_diag_sum
            matrix_list.append(Ml)
            row, col = np.where(Ml)
            coo = np.rec.fromarrays([row, col, Ml[row, col]], names='row col value'.split())
            # pprint(coo.tolist())
            processed_list[key][ds_filename[i]] = coo.tolist()
            
    with open('process_data/Nagano_process/'+resolution+'_con'+cell_contact_threshold_num+'/ds_9/ds_list_stage'+str(stage_num)+'.pkl', 'wb') as f:
        pickle.dump(processed_list, f) 
        
    return Mh_count, Ml_count        

def save_scihic_list(matrix_resolution,stage_num,resolution,cell_contact_threshold_num):
    dicfile = open('process_data/Nagano_process/'+resolution+'_con'+cell_contact_threshold_num+'/ds_9/ds_list_stage'+str(stage_num)+'.pkl', 'rb')     
    scihic_dic = pickle.load(dicfile)
    for chrom in scihic_dic.keys():
        for cell_path in scihic_dic[chrom].keys():
            cell = cell_path.replace('process_data/Nagano/1CDX_cells/','')
            with open('process_data/Nagano_process/'+resolution+'_con'+cell_contact_threshold_num+'/ds_9/stage{s}/{c}'.format(c=cell,s=stage_num), 'a') as f:
                for i in range(len(scihic_dic[chrom][cell_path])):
                    temp_string = list(scihic_dic[chrom][cell_path][i])
                    temp_string.insert(1, chrom)
                    temp_string.insert(0, chrom)
                    temp_string[1]=temp_string[1]*matrix_resolution
                    temp_string[3]=temp_string[3]*matrix_resolution
                    f.write(" ".join(str(item) for item in temp_string))
                    f.write("\n")

if __name__ == '__main__':
    
    with open("process_data/Nagano_process/stage1_filename", "rb") as fp:   
        stage1_filename_temp = pickle.load(fp)
        stage1_filename = [s.replace('single_cell', 'process_data') for s in stage1_filename_temp]
    with open("process_data/Nagano_process/stage2_filename", "rb") as fp:   
        stage2_filename_temp = pickle.load(fp)
        stage2_filename = [s.replace('single_cell', 'process_data') for s in stage2_filename_temp]
    with open("process_data/Nagano_process/stage3_filename", "rb") as fp:   
        stage3_filename_temp = pickle.load(fp)
        stage3_filename = [s.replace('single_cell', 'process_data') for s in stage3_filename_temp]
    with open("process_data/Nagano_process/stage4_filename", "rb") as fp:   
        stage4_filename_temp = pickle.load(fp)
        stage4_filename = [s.replace('single_cell', 'process_data') for s in stage4_filename_temp]
        
    down_sampling_ratio = 9
    matrix_resolution = 1000000
    resolution='1mb'
    cell_contact_threshold_num = '310000'
    stage_list = [stage1_filename,stage2_filename,stage3_filename,stage4_filename]
    # stage_list = [stage1_filename[0:2],stage2_filename[0:2],stage3_filename[0:2],stage4_filename[0:2]]
    total_Mh_count=0
    total_Ml_count=0
    for stage in range(len(stage_list)):
        Mh_count, Ml_count = process_and_down_sampleing_HiC(stage_list[stage],down_sampling_ratio,matrix_resolution, stage+1,resolution,cell_contact_threshold_num)
        total_Mh_count+= Mh_count
        total_Ml_count+=Ml_count
        save_scihic_list(matrix_resolution,stage+1,resolution,cell_contact_threshold_num)
    print('total_Mh_count: ',total_Mh_count)
    print('total_Ml_count: ',total_Ml_count)
    print('average cell count mh: ',total_Mh_count/(len(stage1_filename)+len(stage2_filename)+len(stage3_filename)+len(stage4_filename)))
    print('average cell count ml: ',total_Ml_count/(len(stage1_filename)+len(stage2_filename)+len(stage3_filename)+len(stage4_filename)))
    print('Finished downsampling! ')
    
    
    
    
    
    
