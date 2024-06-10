import numpy as np
import torch
from torch.utils.data import default_collate


# use this because we want different size of (N cars) in a batch
def collate_fn_dict(in_batch):
    """Custom collate_fn that returns a dict of lists

    Args:
        in_batch: Batch containing a list of dictionaries

    Returns:
        Batch containing a dict of lists
    """

    def recursive_numpy_to_tensor(np_data):
        if isinstance(np_data, np.ndarray):
            np_data = torch.from_numpy(np_data)
        elif isinstance(np_data, tuple):
            np_data = [recursive_numpy_to_tensor(x) for x in np_data]
        elif isinstance(np_data, list):
            np_data = [recursive_numpy_to_tensor(x) for x in np_data]
        elif isinstance(np_data, dict):
            for key in np_data:
                np_data[key] = recursive_numpy_to_tensor(np_data[key])
        return np_data

    in_batch = recursive_numpy_to_tensor(in_batch)
    out_batch = dict()
    for key in in_batch[0]:
        out_batch[key] = [x[key] for x in in_batch]
    return out_batch
