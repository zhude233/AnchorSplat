import torch
import gin
from .GS import SplatfactoDataset

@gin.configurable
def GS_collate_fn(data_list):
    return data_list

@gin.configurable
def build_trainloader(batch_size, num_workers, collate_fn, accumulate_step):
    with gin.config_scope('train_dataset'):
        train_dataset = SplatfactoDataset()
    assert batch_size % torch.cuda.device_count() == 0, 'Batch size should be divisible by the number of GPUs'
    assert batch_size % accumulate_step == 0, 'Batch size should be divisible by the number of accumulate steps'
    batch_size_per_gpu = int(batch_size / (torch.cuda.device_count()*accumulate_step))
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_per_gpu, num_workers=num_workers, 
                                             collate_fn=collate_fn)
    return dataloader

@gin.configurable
def build_testloader(batch_size, num_workers, collate_fn):
    test_nerfstudio_folder = gin.query_parameter('test_dataset/SplatfactoDataset.nerfstudio_folder')
    test_colmap_folder = gin.query_parameter('test_dataset/SplatfactoDataset.colmap_folder')

    assert type(test_nerfstudio_folder) == type(test_colmap_folder), 'test_nerfstudio_folder and test_colmap_folder should have the same type'
    if type(test_nerfstudio_folder) == str: #legacy
        test_folder_dict = {'default':{'nerfstudio_folder':test_nerfstudio_folder, 'colmap_folder':test_colmap_folder}}
    elif type(test_nerfstudio_folder) == dict:
        test_folder_dict = {}
        for key in test_nerfstudio_folder.keys():
            test_folder_dict[key] = {'nerfstudio_folder':test_nerfstudio_folder[key], 'colmap_folder':test_colmap_folder[key]}
    
    rt_dataloader = {}
    for test_dataset_name, nerfstudio_colmap_folder in test_folder_dict.items():
        with gin.config_scope('test_dataset'):
            test_dataset = SplatfactoDataset()
        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, 
                                                collate_fn=collate_fn)
        rt_dataloader[test_dataset_name] = dataloader
    return rt_dataloader


