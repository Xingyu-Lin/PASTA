import torch


class LogDict(object):
    def __init__(self):
        self.dict = None

    def log_dict(self, d):
        if self.dict is None:
            self.dict = {}
            for key, val in d.items():
                self.dict[key] = [val]
        else:
            for key, val in d.items():
                self.dict[key].append(val)

    def agg(self, reduction='sum', numpy=False):
        assert reduction in ['sum', 'mean']
        ret = {key: sum(self.dict[key]) for key in self.dict.keys()}
        if reduction == 'mean':
            for key in self.dict.keys():
                ret[key] /= len(self.dict[key])
        if numpy:
            for key in ret.keys():
                if isinstance(ret[key], torch.Tensor):
                    ret[key] = ret[key].item()  # Assume scalar
        return ret
