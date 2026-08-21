"""推理引擎的集中配置。

学习重点：这里的参数大致分为三类：调度容量、KV cache 容量和并行/执行模式。
它们最终会同时约束“一轮能算多少 token”和“GPU 上能常驻多少条序列”。
"""

import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    """Nano-vLLM 的最小配置集合。

    ``max_num_batched_tokens`` 是单个 engine step 的 token budget；
    ``max_num_seqs`` 是同一轮最多调度的 sequence 数；
    ``num_kvcache_blocks`` 在模型 warmup 后根据剩余显存反推，而不是由用户直接指定。
    """

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # 当前实现直接从本地 checkpoint 目录加载权重，不接受 Hugging Face repo id。
        assert os.path.isdir(self.model)
        # FlashAttention paged KV cache 的实现要求这里按 256 对齐。
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 用户设置不能越过模型配置声明的最大位置长度。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
