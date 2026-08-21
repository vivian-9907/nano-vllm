from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """对外公开的 API 名称。

    当前没有增加额外逻辑：用户调用 ``LLM.generate``，实际执行的是父类
    :class:`LLMEngine` 中的请求入队和 engine step 循环。
    """

    pass
