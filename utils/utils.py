from pathlib import Path 
import os 


import torch
import torch.distributed as dist
from transformers.tokenization_utils_base import BatchEncoding

def batch_to_cuda(batch):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()
        if isinstance(batch[key],  BatchEncoding):
            for sub_key in batch[key]:
                if isinstance(batch[key][sub_key], torch.Tensor):
                    batch[key][sub_key] = batch[key][sub_key].cuda()
    return batch

def batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        if isinstance(batch[key],  BatchEncoding):
            for sub_key in batch[key]:
                if isinstance(batch[key][sub_key], torch.Tensor):
                    batch[key][sub_key] = batch[key][sub_key].to(device)
    return batch


class MetricMonitor:
    def __init__(self,):
        self.metrics = {} 

    def add_with_step(self, name, value, step):
        if name not in self.metrics:
            self.metrics[name] = {step: value}
        else:
            self.metrics[name].update({step: value})
    
    def __str__(self):
        _metrics = self.metrics 
        names = list(_metrics.keys())
        fmt_str = "step\t" + "\t".join(names) + "\n"

        f_name = names[0]
        for step in _metrics[f_name]:
            fmt_str += f"{step:4>}"
            for name in names:
                val = _metrics[name].get(step, None)
                if val is not None:
                    fmt_str += f"\t{val:.3f}"
                else:
                    fmt_str += "\tNULL"
            fmt_str += "\n"
        
        return fmt_str

    def write_to_file(self, output_path):
        p = Path(output_path)
        if not os.path.exists(p.parent):
            os.mkdir(p.parent)

        with open(output_path, "w") as f:
            f.write(str(self))

def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0