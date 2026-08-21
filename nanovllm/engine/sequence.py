"""单条生成请求在调度器中的可变状态。

Sequence 是串起 tokenizer、scheduler、block manager 和 model runner 的核心对象。
要特别区分：``num_tokens`` 是逻辑序列长度，``num_cached_tokens`` 是已经执行
forward 并写入 KV cache 的前缀长度，``num_scheduled_tokens`` 是本轮即将计算的长度。
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """请求生命周期：等待分配/重算、已进入 decode、已经结束。"""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """一条 prompt 及其后续生成 token 的完整状态。"""

    # BlockManager 和 ModelRunner 必须共享同一个 block size；引擎初始化时会覆盖它。
    block_size = 256
    # 单进程内单调递增的请求 ID，用于最终按提交顺序整理输出。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        # 已计算且可从 KV cache 读取的 token 数；prefix cache 命中时可从 0 直接跳到若干整块。
        self.num_cached_tokens = 0
        # scheduler 为当前 engine step 分配的 token 数，postprocess 后会清零。
        self.num_scheduled_tokens = 0
        # prefill 可以一次处理多个 token；decode 通常每条 sequence 每轮只处理最后一个 token。
        self.is_prefill = True
        # logical block i -> physical KV-cache block_id。
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已经追加到 prompt 后面的生成 token 数。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """当前逻辑序列覆盖的 block 数，向上取整。"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个逻辑 block 中已经使用的 slot 数。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """返回第 i 个逻辑 block 对应的 token；最后一块可能不满。"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """commit 一枚采样 token；其 KV 会在下一次 decode forward 中写入。"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """为 TP worker 间的 pickle 通信压缩状态。

        prefill worker 需要完整 token slice；decode worker 只需要最后一个 token。
        temperature 等参数只在 rank 0 采样，因此没有被发送给其他 rank。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        # 子 worker 只重建本轮模型执行所需的最小 Sequence 视图。
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
