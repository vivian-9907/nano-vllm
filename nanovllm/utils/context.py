"""一次 model forward 的 Attention 元数据。

这些字段不是模型参数，而是 scheduler/model runner 为当前 batch 临时构造的信息。
用进程内全局 Context 可避免修改每个 decoder layer 的 forward 参数列表。
"""

from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """prefill 和 decode 共用的数据容器；未使用字段保持 None/0。"""

    is_prefill: bool = False
    # Prefill varlen metadata：长度均为 batch+1 的累积 token offsets。
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 每个新 token 的 K/V 应写入的扁平物理 cache slot。
    slot_mapping: torch.Tensor | None = None
    # Decode 时每条 sequence 的完整上下文长度，shape=[batch]。
    context_lens: torch.Tensor | None = None
    # logical block -> physical block 的 padded 映射，shape=[batch, max_blocks]。
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    """供 Attention/LM head 读取当前 forward metadata。"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """ModelRunner 在进入模型前设置一次当前 batch metadata。"""
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """forward 完成后清空引用，防止下一批读取旧 tensor。"""
    global _CONTEXT
    _CONTEXT = Context()
