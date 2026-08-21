"""Temperature sampling；使用 exponential race 实现 categorical sample。"""

import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits: [batch, vocab]，temperature: [batch]。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # 若 E_i ~ Exp(1)，argmax(p_i / E_i) 服从 categorical(p)，等价于 Gumbel-Max。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
