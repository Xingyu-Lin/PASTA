import torch
import numpy as np
import random


class VArgs(object):
    def __init__(self, vv):
        for key, val in vv.items():
            setattr(self, key, val)

def set_ipdb_debugger():
    import sys
    import ipdb
    import traceback

    def info(t, value, tb):
        if t == KeyboardInterrupt:
            traceback.print_exception(t, value, tb)
            return
        else:
            traceback.print_exception(t, value, tb)
            ipdb.pm()

    sys.excepthook = info

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_to_array(l, tensor=False, axis=0):
    """Concatenate list and save index"""
    if axis == 0:
        list_idx = [len(ele) for ele in l]
    else:
        list_idx = [ele.shape[1] for ele in l]

    if not tensor:
        l = np.concatenate(l, axis=axis)
    else:
        l = torch.cat(l, dim=axis)
    return l, list_idx


def array_to_list(arr, list_idx, axis=0):
    """Concatenate array to list"""
    ret = []
    cnt = 0
    for num in list_idx:
        if axis == 0:
            ret.append(arr[cnt:cnt + num])
        elif axis == 1:
            ret.append(arr[:, cnt:cnt + num])
        else:
            raise NotImplementedError
        cnt += num
    return ret
