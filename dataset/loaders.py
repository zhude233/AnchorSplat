import torch
import gin
from .gaussian_scene import GaussianSceneDataset

@gin.configurable
def gaussian_collate_fn(data_list):
    return data_list

@gin.configurable
def build_trainloader(batch_size, num_workers, collate_fn, accumulate_step):
    with gin.config_scope('train_dataset'):
        train_dataset = GaussianSceneDataset()
    assert batch_size % torch.cuda.device_count() == 0, 'Batch size should be divisible by the number of GPUs'
    assert batch_size % accumulate_step == 0, 'Batch size should be divisible by the number of accumulate steps'
    batch_size_per_gpu = int(batch_size / (torch.cuda.device_count()*accumulate_step))
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader

@gin.configurable
def build_testloader(batch_size, num_workers, collate_fn):
    with gin.config_scope('test_dataset'):
        test_dataset = GaussianSceneDataset()
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return {'default': dataloader}
