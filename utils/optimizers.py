import torch, gin 
from models.feature_predictor import FeaturePredictor
@gin.configurable
def build_3DGSoptimizer(gs_params, lr_dict, optimizer_type, optimizer_params):
    params_lr = []
    for param in gs_params:
        lr = lr_dict.get(param, lr_dict['base'])
        params_lr.append({'params': gs_params[param], 'lr': lr})
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(params_lr, 
                                     lr = lr_dict['base'],
                                     **optimizer_params)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(params_lr, 
                                    lr = lr_dict['base'])
    else:
        raise NotImplementedError
    return optimizer

@gin.configurable
def build_optimizer(model, 
                    lr_dict: gin.REQUIRED, 
                    optimizer_type: gin.REQUIRED,
                    optimizer_params):  
    params_lr = []
    if type(model) == FeaturePredictor:
        if hasattr(model, 'point_embeddings'):
            embedding_lr = lr_dict.get('embedding', lr_dict['base'])
            params_lr.append({'params': model.point_embeddings.parameters(), 'lr': embedding_lr})
        if model.backbone_type != 'empty':
            params_lr.append({'params': model.backbone.parameters(), 'lr': lr_dict['backbone']})
        for feature in model.features_outputhead.keys():
            lr = lr_dict.get(feature, lr_dict['base'])
            params_lr.append({'params': model.features_outputhead[feature].parameters(), 'lr': lr})
    else:
        for param in model.parameters():
            lr = lr_dict.get(param, lr_dict['base'])
            params_lr.append({'params': param, 'lr': lr})

    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(params_lr, 
                                     lr = lr_dict['base'],
                                     **optimizer_params)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(params_lr, 
                                    lr = lr_dict['base'])
    else:
        raise NotImplementedError
    return optimizer

@gin.configurable
def build_scheduler(optimizer, schedule, total_step, warmup_step=0):
    if schedule == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    elif schedule == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1-step/total_step)
    elif schedule == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_step)
    elif schedule == 'exponential':
        raise ValueError
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exponential_gamma)
    else:
        raise NotImplementedError
    if warmup_step > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: step/warmup_step)
        lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, lr_scheduler], optimizer)
    return lr_scheduler
