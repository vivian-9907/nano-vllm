"""Paged KV cache 的物理 block 分配器与 prefix cache 索引。

Sequence 只保存逻辑 token 顺序；BlockManager 把每个逻辑 block 映射到预分配的
GPU KV-cache block。物理 block 可以不连续，因此序列增长时无须搬迁整段 KV cache。
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """一个物理 KV block 的 CPU 侧元数据。

    ``ref_count`` 支持多个相同前缀的 sequence 共享同一物理 block；``hash`` 和
    ``token_ids`` 共同验证 prefix cache 命中，后者用于防御 hash collision。
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        # 新分配的 block 先由一个 sequence 独占，写满后才会生成可复用 hash。
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """管理固定数量的预分配 KV blocks，不直接分配 GPU tensor。"""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        # free/used 只描述当前是否被活跃 sequence 引用；free block 仍可能保留旧 hash，
        # 因而可在尚未被覆盖时被 prefix cache “复活”。
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """构造链式 block hash：H_i = hash(H_{i-1}, tokens_i)。

        同一 token block 出现在不同前缀后时不会被错误视为同一个完整前缀。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从 free list 取一个物理 block，并清除其可能遗留的 prefix 索引。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """把引用数已归零的 block 放回 free list；保留 hash 以供缓存复用。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """返回可命中的完整 prefix blocks 数；空间不足返回 -1。

        只扫描 ``num_blocks - 1``：最后一块可能未满，而未满 block 会继续被写入，
        不能安全地作为不可变前缀共享。
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 命中的 block 若仍在 used 集合中只需增加引用；若已 free，则仍需从
            # free list 中重新占用，因此只有前一种情况能减少“新占用”的 block 数。
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为新 sequence 建立完整 block table，并挂接可复用的前缀 blocks。"""
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # 活跃前缀共享：多个 sequence 的 block table 指向同一个物理 block。
                block.ref_count += 1
            else:
                # 缓存 block 已无活跃引用但尚未被覆盖，从 free list 中重新激活。
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放一条 sequence 对所有物理 blocks 的引用。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查追加最新 token 是否需要且能够分配新 block。

        scheduler 在采样后已经把 token append 到 Sequence。因此长度模 block_size 为 1
        表示刚进入一个新逻辑 block，下一轮 forward 前必须为它补一个物理 block。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """若 sequence 刚跨过 block 边界，为最后一个逻辑 block 分配物理空间。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """将本轮新写满的 blocks 登记进 prefix cache。

        只有越过完整 block 边界后才建立 hash；部分 block 仍可能在下一轮被修改。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
