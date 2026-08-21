"""请求调度：把 waiting/running queues 转换成一次 GPU model step。

当前策略有两个阶段：只要本轮能调度 prefill，就立即返回 prefill batch；只有没有
prefill 可运行时才调度 decode。因此它并不是 vLLM V1 的 decode-first 混合调度，
但已经支持“首条请求”的 chunked prefill。
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """同时受 sequence 数、token budget 和可用 KV blocks 约束的调度器。"""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """新请求先进入 FIFO waiting queue。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """选择下一轮请求，返回 ``(sequences, is_prefill)``。

        一个返回 batch 不会混合 prefill 与 decode。这简化了 ModelRunner 的输入准备，
        代价是长 prefill 可能阻塞已在运行的 decode 请求。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ---- Prefill 阶段 -------------------------------------------------
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 第一次调度：先查询 prefix cache 命中和 KV 空间，再建立 block table。
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 被切块的 prefill 已有 block table，只需计算尚未进入 KV cache 的 suffix。
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 若 batch 已有其他请求，本实现不允许再塞入一个无法完整容纳的 prefill；
            # 只有 batch 中第一条 sequence 可以吃掉剩余 budget，形成 chunked prefill。
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # prompt 全部计算完成后进入 RUNNING。此轮还会从最后一个 prompt hidden
                # state 采样首个 completion token，下一轮才以该 token 执行 decode。
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # ---- Decode 阶段 --------------------------------------------------
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # append_token 可能刚跨入新 block。空间不足时从队尾抢占其他请求；
            # 被抢占请求会释放全部 KV，之后从 waiting 重新 prefill（recompute）。
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # decode 中每条 sequence 每轮只计算 last_token，并采样一个新 token。
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 保持本轮被调度 sequence 在 running 队列前部及其原有顺序。
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """采用 recompute preemption：释放全部 KV，并回到 waiting queue 头部。"""
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """提交一次 model step 的状态变化和采样结果。"""
        for seq, token_id in zip(seqs, token_ids):
            # 先登记刚写满的 blocks，再推进 cached-token 水位。
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # chunked prefill 尚未覆盖完整 prompt：本轮 logits 不能作为最终 next-token
                # 结果使用，sequence 留在 waiting，下一轮继续处理 suffix。
                continue
            # 新 token 只进入逻辑序列；它的 K/V 将在下一次 decode forward 时写入 cache。
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
