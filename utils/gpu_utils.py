import torch

def move_to_device(data, device):
    if data is None:
        return None
    elif isinstance(data, (list, tuple)):
        return [move_to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        return data.to(device)