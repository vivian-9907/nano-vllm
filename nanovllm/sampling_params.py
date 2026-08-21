from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """每条 sequence 独立持有的采样与停止参数。"""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        # Sampler 使用 temperature sampling；该最小实现刻意不支持 temperature=0 的 greedy 路径。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
