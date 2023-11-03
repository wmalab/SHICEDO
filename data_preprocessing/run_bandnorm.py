import pandas as pd
import numpy as np
import glob
import os

resolution="1mb"
contact_threshold = "310000"
stage_list=["stage1","stage2","stage3","stage4"]

for stage in stage_list:
    print('start change Low resolution list')
    df = pd.read_csv("process_data/Nagano_process/"+resolution+"_con"+contact_threshold+"/ds_9/norm/"+stage+".txt", sep="\t")
    df.insert(2, "chrom", df['chrom'], True)
    del df["diag"]
    grouped = df.groupby(['cell'])
    for key in grouped.groups.keys():
        temp_df = grouped.get_group(key)
        del temp_df["cell"]
        temp_df['BandNorm'] = np.log10(temp_df['BandNorm'] + 1)
        temp_name = key.replace('.txt', '')
        path = 'process_data/Nagano_process/'+resolution+'_con'+contact_threshold+'/ds_9/norm/'+stage+'/'
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)  
        np.savetxt(path+temp_name, temp_df.values,fmt="%s")
    print('finished process Low resolution norm !!!')
    
for stage in stage_list:
    print('start change true list')
    df = pd.read_csv("process_data/Nagano_process/"+resolution+"_con"+contact_threshold+"/filter_true_no_inter/filter_true_no_inter_norm/"+stage+".txt", sep="\t")
    df.insert(2, "chrom", df['chrom'], True)
    del df["diag"]
    grouped = df.groupby(['cell'])
    for key in grouped.groups.keys():
        temp_df = grouped.get_group(key)
        del temp_df["cell"]
        temp_df['BandNorm'] = np.log10(temp_df['BandNorm'] + 1)
        
        temp_name = key.replace('.txt', '')
        path = 'process_data/Nagano_process/'+resolution+'_con'+contact_threshold+'/filter_true_no_inter/filter_true_no_inter_norm/'+stage+'/'
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)  
        np.savetxt(path+temp_name, temp_df.values,fmt="%s")
    print('finished process true norm !!!')