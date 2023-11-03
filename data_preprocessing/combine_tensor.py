import torch
import os 

def Combine_tensor(path):
    root_path= path+'stage1/'
    train_index1=torch.load(root_path+'train_index.pt')
    train_large_img1=torch.load(root_path+'train_large_img.pt')
    train_regular_img1=torch.load(root_path+'train_regular_img.pt')
    train_True_img1=torch.load(root_path+'train_true_img.pt')

    vali_index1=torch.load(root_path+'vali_index.pt')
    vali_large_img1=torch.load(root_path+'vali_large_img.pt')
    vali_regular_img1=torch.load(root_path+'vali_regular_img.pt')
    vali_True_img1=torch.load(root_path+'vali_true_img.pt')

    root_path=path+'stage2/'
    train_index2=torch.load(root_path+'train_index.pt')
    train_large_img2=torch.load(root_path+'train_large_img.pt')
    train_regular_img2=torch.load(root_path+'train_regular_img.pt')
    train_True_img2=torch.load(root_path+'train_true_img.pt')

    vali_index2=torch.load(root_path+'vali_index.pt')
    vali_large_img2=torch.load(root_path+'vali_large_img.pt')
    vali_regular_img2=torch.load(root_path+'vali_regular_img.pt')
    vali_True_img2=torch.load(root_path+'vali_true_img.pt')

    root_path=path+'stage3/'
    train_index3=torch.load(root_path+'train_index.pt')
    train_large_img3=torch.load(root_path+'train_large_img.pt')
    train_regular_img3=torch.load(root_path+'train_regular_img.pt')
    train_True_img3=torch.load(root_path+'train_true_img.pt')

    vali_index3=torch.load(root_path+'vali_index.pt')
    vali_large_img3=torch.load(root_path+'vali_large_img.pt')
    vali_regular_img3=torch.load(root_path+'vali_regular_img.pt')
    vali_True_img3=torch.load(root_path+'vali_true_img.pt')

    root_path=path+'stage4/'
    train_index4=torch.load(root_path+'train_index.pt')
    train_large_img4=torch.load(root_path+'train_large_img.pt')
    train_regular_img4=torch.load(root_path+'train_regular_img.pt')
    train_True_img4=torch.load(root_path+'train_true_img.pt')

    vali_index4=torch.load(root_path+'vali_index.pt')
    vali_large_img4=torch.load(root_path+'vali_large_img.pt')
    vali_regular_img4=torch.load(root_path+'vali_regular_img.pt')
    vali_True_img4=torch.load(root_path+'vali_true_img.pt')

    test_index1=torch.load(root_path+'test_index.pt')
    test_large_img1=torch.load(root_path+'test_large_img.pt')
    test_regular_img1=torch.load(root_path+'test_regular_img.pt')
    test_True_img1=torch.load(root_path+'test_true_img.pt')

    test_index2=torch.load(root_path+'test_index.pt')
    test_large_img2=torch.load(root_path+'test_large_img.pt')
    test_regular_img2=torch.load(root_path+'test_regular_img.pt')
    test_True_img2=torch.load(root_path+'test_true_img.pt')

    test_index3=torch.load(root_path+'test_index.pt')
    test_large_img3=torch.load(root_path+'test_large_img.pt')
    test_regular_img3=torch.load(root_path+'test_regular_img.pt')
    test_True_img3=torch.load(root_path+'test_true_img.pt')

    test_index4=torch.load(root_path+'test_index.pt')
    test_large_img4=torch.load(root_path+'test_large_img.pt')
    test_regular_img4=torch.load(root_path+'test_regular_img.pt')
    test_True_img4=torch.load(root_path+'test_true_img.pt')

    train_index = torch.cat([train_index1, train_index2, train_index3, train_index4], dim=0)
    train_large_img = torch.cat([train_large_img1, train_large_img2, train_large_img3, train_large_img4], dim=0)
    train_regular_img = torch.cat([train_regular_img1, train_regular_img2, train_regular_img3, train_regular_img4], dim=0)
    train_True_img = torch.cat([train_True_img1, train_True_img2, train_True_img3, train_True_img4], dim=0)
    
    vali_index = torch.cat([vali_index1, vali_index2, vali_index3, vali_index4], dim=0)
    vali_large_img = torch.cat([vali_large_img1, vali_large_img2, vali_large_img3, vali_large_img4], dim=0)
    vali_regular_img = torch.cat([vali_regular_img1, vali_regular_img2, vali_regular_img3, vali_regular_img4], dim=0)
    vali_True_img = torch.cat([vali_True_img1, vali_True_img2, vali_True_img3, vali_True_img4], dim=0)

    test_index = torch.cat([test_index1, test_index2, test_index3, test_index4], dim=0)
    test_large_img = torch.cat([test_large_img1, test_large_img2, test_large_img3, test_large_img4], dim=0)
    test_regular_img = torch.cat([test_regular_img1, test_regular_img2, test_regular_img3, test_regular_img4], dim=0)
    test_True_img = torch.cat([test_True_img1, test_True_img2, test_True_img3, test_True_img4], dim=0)

    path=path+'combined'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)  

    save_tensor_path = path +'/'+'train_index.pt'
    torch.save(train_index, save_tensor_path)
    save_tensor_path = path +'/'+'train_large_img.pt'
    torch.save(train_large_img, save_tensor_path)
    save_tensor_path = path +'/'+'train_regular_img.pt'
    torch.save(train_regular_img, save_tensor_path)
    save_tensor_path = path +'/'+'train_true_img.pt'
    torch.save(train_True_img, save_tensor_path)

    save_tensor_path = path +'/'+'vali_index.pt'
    torch.save(vali_index, save_tensor_path)
    save_tensor_path = path +'/'+'vali_large_img.pt'
    torch.save(vali_large_img, save_tensor_path)
    save_tensor_path = path +'/'+'vali_regular_img.pt'
    torch.save(vali_regular_img, save_tensor_path)
    save_tensor_path = path +'/'+'vali_true_img.pt'
    torch.save(vali_True_img, save_tensor_path)

    save_tensor_path = path +'/'+'test_index.pt'
    torch.save(test_index, save_tensor_path)
    save_tensor_path = path +'/'+'test_large_img.pt'
    torch.save(test_large_img, save_tensor_path)
    save_tensor_path = path +'/'+'test_regular_img.pt'
    torch.save(test_regular_img, save_tensor_path)
    save_tensor_path = path +'/'+'test_true_img.pt'
    torch.save(test_True_img, save_tensor_path)

