import os
import pickle
import statistics
import shutil

def load_select_file_name(filename,contacts,threshold_num):
    with open(filename, "rb") as fp:   # Unpickling
        filename = pickle.load(fp)
    with open(contacts, "rb") as fp:   # Unpickling
        contacts = pickle.load(fp)
    contact_median = statistics.median(contacts)
    print('contact count median: ',statistics.median(contacts))
    print('contact count mean: ',statistics.mean(contacts))
    print('contact count min: ',min(contacts))
    print('contact count max: ',max(contacts))
    selected_filename=[]
    for i in range(len(contacts)):
        if contacts[i] >= threshold_num:
            temp = filename[i].replace('../test/data', 'single_cell/Nagano')
            selected_filename.append(temp)
    return selected_filename

def Run_separate_by_stage(selected_filename,root_path):
    stage1,stage2,stage3,stage4 = [],[],[],[]
    for name in selected_filename:
        temp = name.split('/1CDX_cells/')[1]
        stage = temp.split('.')[0]
        if stage == '1CDX1':
            stage1.append(name)
        elif stage == '1CDX2':
            stage2.append(name)
        elif stage == '1CDX3':
            stage3.append(name)
        elif stage == '1CDX4':
            stage4.append(name)    
    isExist = os.path.exists(root_path+'process_data/Nagano_process')
    if not isExist:
        os.makedirs(root_path+'process_data/Nagano_process')        
    with open(root_path+'process_data/Nagano_process/stage1_filename', "wb") as fp:  
        pickle.dump(stage1, fp)
    with open(root_path+'process_data/Nagano_process/stage2_filename', "wb") as fp:   
        pickle.dump(stage2, fp)    
    with open(root_path+'process_data/Nagano_process/stage3_filename', "wb") as fp:   
        pickle.dump(stage3, fp)        
    with open(root_path+'process_data/Nagano_process/stage4_filename', "wb") as fp:   
        pickle.dump(stage4, fp)  
    return stage1,stage2,stage3,stage4

if __name__ == '__main__':
    root_path=os.getcwd()+'/'
    cell_contact_threshold_num = 310000 #250000
    resolution='1mb'
    filename = root_path+"process_data/Nagano/filename"
    contacts = root_path+"process_data/Nagano/contacts"
    selected_filename = load_select_file_name(filename,contacts,cell_contact_threshold_num)
    separate_by_stage = True
    if separate_by_stage == True:
        stage1,stage2,stage3,stage4 = Run_separate_by_stage(selected_filename,root_path)
 
    path = root_path+'process_data/Nagano_process/'+resolution+'_con' + str(cell_contact_threshold_num) + '/filter_true/stage1/'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    filter_stage1_folders = [s.replace('single_cell/Nagano/1CDX_cells/', path) for s in stage1]
    filter_stage1_folders = [s.replace('/new_adj', '.txt') for s in filter_stage1_folders]

    path = root_path+'process_data/Nagano_process/'+resolution+'_con' + str(cell_contact_threshold_num) + '/filter_true/stage2/'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    filter_stage2_folders = [s.replace('single_cell/Nagano/1CDX_cells/', path) for s in stage2]
    filter_stage2_folders = [s.replace('/new_adj', '.txt') for s in filter_stage2_folders]
    
    path = root_path+'process_data/Nagano_process/'+resolution+'_con' + str(cell_contact_threshold_num) + '/filter_true/stage3/'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    filter_stage3_folders = [s.replace('single_cell/Nagano/1CDX_cells/', path) for s in stage3]
    filter_stage3_folders = [s.replace('/new_adj', '.txt') for s in filter_stage3_folders]
    
    path = root_path+'process_data/Nagano_process/'+resolution+'_con' + str(cell_contact_threshold_num) + '/filter_true/stage4/'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    filter_stage4_folders = [s.replace('single_cell/Nagano/1CDX_cells/', path) for s in stage4]
    filter_stage4_folders = [s.replace('/new_adj', '.txt') for s in filter_stage4_folders]
    
    for i in range(len(stage1)):
        temp=stage1[i].replace('single_cell', 'process_data')
        shutil.copyfile(temp, filter_stage1_folders[i])    
    for i in range(len(stage2)):
        temp=stage2[i].replace('single_cell', 'process_data')
        shutil.copyfile(temp, filter_stage2_folders[i])    
    for i in range(len(stage3)):
        temp=stage3[i].replace('single_cell', 'process_data')
        shutil.copyfile(temp, filter_stage3_folders[i])
    for i in range(len(stage4)):
        temp=stage4[i].replace('single_cell', 'process_data')
        shutil.copyfile(temp, filter_stage4_folders[i])
    
    print('Finished data filtering! ')