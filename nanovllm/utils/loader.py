import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Use token-wise replacement to avoid accidental substring matches
                parts = weight_name.split('.')
                for k, (v, shard_id) in packed_modules_mapping.items():
                    if k in parts:
                        param_name = '.'.join((v if p == k else p) for p in parts)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # Direct load path: prefer module-aware loaders to handle TP shards
                    param = model.get_parameter(weight_name)
                    module_path = weight_name.rsplit('.', 1)[0]
                    try:
                        module = model.get_submodule(module_path)
                    except AttributeError:
                        module = None

                    loaded_weight = f.get_tensor(weight_name)

                    if isinstance(module, (MergedColumnParallelLinear, QKVParallelLinear, ColumnParallelLinear)):
                        # Use ColumnParallelLinear logic to shard-copy fused weights
                        ColumnParallelLinear.weight_loader(module, param, loaded_weight)
                    elif isinstance(module, (RowParallelLinear, ReplicatedLinear)):
                        module.weight_loader(param, loaded_weight)
                    else:
                        # Fallback to parameter's own loader or direct copy
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        try:
                            weight_loader(param, loaded_weight)
                        except TypeError:
                            # In case a 3-arg loader is bound, fall back to direct copy
                            default_weight_loader(param, loaded_weight)

