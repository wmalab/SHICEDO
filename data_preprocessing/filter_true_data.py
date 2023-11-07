import numpy as np
import pickle
import os
from scHiCTools import scHiCs
import glob

def process_HiC(selected_filename,matrix_resolution,stage_num,resolution,contact_threshold):
    ds_filename = [s.replace('process_data/Nagano_process/'+resolution+'_con'+contact_threshold+'/filter_true/stage'+str(stage_num), '') for s in selected_filename]
    ds_filename = [s.replace('.txt', '') for s in ds_filename]
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
    for key in raw_dic.keys():
        matrix_list=[]
        for i in range(raw_dic[key].shape[0]):
            Mh = raw_dic[key][i,:,:]
            matrix_list.append(Mh)
            row, col = np.where(Mh)
            coo = np.rec.fromarrays([row, col, Mh[row, col]], names='row col value'.split())
            processed_list[key][ds_filename[i]] = coo.tolist()
    path = 'process_data/Nagano_process/'+resolution+'_con'+contact_threshold+'/filter_true_no_inter'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    with open(path + '/true_list_stage'+str(stage_num)+'.pkl', 'wb') as f:
        pickle.dump(processed_list, f)        
    
def save_scihic_list(matrix_resolution,stage_num,resolution,contact_threshold):
    dicfile = open('process_data/Nagano_process/'+resolution+'_con'+contact_threshold+'/filter_true_no_inter/true_list_stage'+str(stage_num)+'.pkl', 'rb')     
    scihic_dic = pickle.load(dicfile)
    for chrom in scihic_dic.keys():
        for cell_path in scihic_dic[chrom].keys():
            path = 'process_data/Nagano_process/'+resolution+'_con'+contact_threshold+'/filter_true_no_inter/stage'+str(stage_num)
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            path = 'process_data/Nagano_process/'+resolution+'_con'+contact_threshold+'/filter_true_raw/stage'+str(stage_num)
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            # temp_path = 'process_data/Nagano_process/'+resolution+'_con'+contact_threshold+'/filter_true_no_inter/stage'+str(stage_num)+'/'
            temp_path = 'process_data/Nagano_process/'+resolution+'_con'+contact_threshold+'/filter_true_raw/stage'+str(stage_num)+'/'
            cell = cell_path.replace(temp_path,'')
            with open('process_data/Nagano_process/'+resolution+'_con'+contact_threshold+'/filter_true_no_inter/stage{s}/{c}'.format(c=cell,s=stage_num), 'a') as f:
                for i in range(len(scihic_dic[chrom][cell_path])):
                    temp_string = list(scihic_dic[chrom][cell_path][i])
                    temp_string.insert(1, chrom)
                    temp_string.insert(0, chrom)
                    temp_string[1]=temp_string[1]*matrix_resolution
                    temp_string[3]=temp_string[3]*matrix_resolution
                    f.write(" ".join(str(item) for item in temp_string))
                    f.write("\n")

if __name__ == '__main__':
    resolution="1mb"
    contact_threshold = "310000"
    matrix_resolution = 1000000
    stage1_true_cells_file = sorted(glob.glob("process_data/Nagano_process/"+resolution+"_con"+contact_threshold+"/filter_true/stage1/*"))
    stage2_true_cells_file = sorted(glob.glob("process_data/Nagano_process/"+resolution+"_con"+contact_threshold+"/filter_true/stage2/*"))
    stage3_true_cells_file = sorted(glob.glob("process_data/Nagano_process/"+resolution+"_con"+contact_threshold+"/filter_true/stage3/*"))
    stage4_true_cells_file = sorted(glob.glob("process_data/Nagano_process/"+resolution+"_con"+contact_threshold+"/filter_true/stage4/*"))
    stage_list = [stage1_true_cells_file,stage2_true_cells_file,stage3_true_cells_file,stage4_true_cells_file]
    # stage_list = [stage1_true_cells_file[0:2],stage2_true_cells_file[0:2],stage3_true_cells_file[0:2],stage4_true_cells_file[0:2]]
    for stage in range(len(stage_list)):
    # for i, stage in enumerate(stage_list):
        process_HiC(stage_list[stage],matrix_resolution, stage+1,resolution,contact_threshold)
        save_scihic_list(matrix_resolution,stage+1,resolution,contact_threshold)

    print('\n Finished filter out the inter-chromosomal interactions! ')