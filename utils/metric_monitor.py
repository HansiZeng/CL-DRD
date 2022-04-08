from pathlib import Path 
import os 

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


class MetricMonitor_old:
    def __init__(self,):
        self.metrics = {} 

    def add_with_epoch(self, name, value, epoch):
        if name not in self.metrics:
            self.metrics[name] = {epoch: value}
        else:
            self.metrics[name].update({epoch: value})
    
    def __str__(self):
        _metrics = self.metrics 
        names = list(_metrics.keys())
        fmt_str = "Epoch\t" + "\t".join(names) + "\n"

        f_name = names[0]
        for epoch in _metrics[f_name]:
            fmt_str += f"{epoch:4>}"
            for name in names:
                val = _metrics[name].get(epoch, None)
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
