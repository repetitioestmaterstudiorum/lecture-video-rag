import os
import random
import numpy as np
import torch

# ---


def set_seeds(seed_value: int = None):
    if seed_value is None:
        seed_value = np.random.randint(0, 10000)

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed value set: {seed_value}")

    return seed_value
