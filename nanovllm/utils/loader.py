"""从 safetensors checkpoint 加载普通、packed 与 TP-sharded parameters。"""

import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """无特殊分片/打包规则时直接复制完整 tensor。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """逐 tensor 流式加载 checkpoint，避免先在 CPU 拼出完整 state_dict。

    parameter 上若挂有 ``weight_loader``，则由具体 layer 决定如何选 TP shard；
    packed mapping 还会把多个 checkpoint tensors 写入同一个运行时 parameter。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 例如 q_proj -> qkv_proj，并把 shard_id="q" 交给 QKV loader。
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 非 packed 参数仍可能拥有 TP-aware loader，例如 embedding/o_proj。
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
