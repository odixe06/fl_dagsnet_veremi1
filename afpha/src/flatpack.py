"""Flat packing of a DAGSNet state_dict.

Two reasons, both practical:

* A state_dict has 192 entries. Putting one per client on a multiprocessing queue
  means ~19k shared-memory file descriptors per round at 100 clients, which
  exhausts the process limit. One tensor per client does not.
* Aggregation becomes a weighted vector sum instead of a per-key Python loop.

Layout is fixed and parameters come first, so `vec[:n_params]` is exactly the
learnable-parameter block the proximal drift is measured over.
"""
import torch


class Template:
    """Key order, shapes and slices for one architecture."""

    def __init__(self, model):
        sd = model.state_dict()
        param_names = list(dict(model.named_parameters()))
        float_keys = param_names + [k for k, v in sd.items()
                                    if v.is_floating_point() and k not in set(param_names)]
        int_keys = [k for k, v in sd.items() if not v.is_floating_point()]

        self.float_keys, self.int_keys = float_keys, int_keys
        self.float_shapes = [tuple(sd[k].shape) for k in float_keys]
        self.int_shapes = [tuple(sd[k].shape) for k in int_keys]
        self.float_sizes = [sd[k].numel() for k in float_keys]
        self.int_sizes = [sd[k].numel() for k in int_keys]
        self.n_params = sum(sd[k].numel() for k in param_names)
        self.n_float = sum(self.float_sizes)
        self.n_int = sum(self.int_sizes)
        self.int_dtypes = [sd[k].dtype for k in int_keys]

    def flatten(self, sd):
        f = torch.empty(self.n_float, dtype=torch.float32)
        i = 0
        for key, size in zip(self.float_keys, self.float_sizes):
            f[i:i + size] = sd[key].detach().reshape(-1).to(torch.float32)
            i += size
        n = torch.empty(self.n_int, dtype=torch.int64)
        i = 0
        for key, size in zip(self.int_keys, self.int_sizes):
            n[i:i + size] = sd[key].detach().reshape(-1).to(torch.int64)
            i += size
        return f, n

    def unflatten(self, f, n):
        out, i = {}, 0
        for key, size, shape in zip(self.float_keys, self.float_sizes, self.float_shapes):
            out[key] = f[i:i + size].reshape(shape).clone()
            i += size
        i = 0
        for key, size, shape, dt in zip(self.int_keys, self.int_sizes, self.int_shapes,
                                        self.int_dtypes):
            out[key] = n[i:i + size].reshape(shape).to(dt).clone()
            i += size
        return out


def weighted_mean(vecs, weights):
    """Sample-weighted mean in float64: 100 float32 additions lose bits."""
    acc = torch.zeros(vecs[0].numel(), dtype=torch.float64)
    for v, w in zip(vecs, weights):
        acc.add_(v.to(torch.float64), alpha=w)
    return acc.to(torch.float32)


def elementwise_max(vecs):
    """num_batches_tracked is a counter; a weighted mean of it is meaningless."""
    out = vecs[0].clone()
    for v in vecs[1:]:
        torch.maximum(out, v, out=out)
    return out
