"""SwiGLU 使用的 fused-style SiLU-and-multiply 激活。"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入最后一维按 [gate | up] 对半切分，输出为 SiLU(gate) * up。
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
