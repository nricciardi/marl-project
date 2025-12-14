import random
import numpy as np
import torch
import os


def set_global_seed(seed: int):
    
    # 1. Python & Numpy
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 3. OS-level
    os.environ["PYTHONHASHSEED"] = str(seed)