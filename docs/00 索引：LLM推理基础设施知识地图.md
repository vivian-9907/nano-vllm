# nano-vLLM 学习与推理基础设施知识地图

这组文档按“源码调用 → 框架数据链路 → 推理生态 → 服务与硬件”组织。可以按编号顺序阅读，也可以从本页按问题定位专题。

## 按问题查卡片

| 我想弄清的问题 | 先看哪张卡 | 关键词 |
| --- | --- | --- |
| `example.py` 怎样进入调度循环 | [从 example.py 进入 nano-vLLM 调度循环](<./01 从example.py进入nano-vLLM调度循环.md>) | LLMEngine、Sequence、Scheduler、Prefill、Decode |
| Scheduler 的结果怎样进入 GPU Kernel | [推理框架与 Attention Kernel 的数据链路](<./02 推理框架与Attention Kernel的数据链路.md>) | ModelRunner、block table、slot mapping、FlashAttention |
| 常见推理优化仓库分别位于哪一层 | [推理优化仓库分层卡片](<./03 推理优化仓库分层卡片.md>) | Engine、Kernel、KV 传输、集群控制面 |
| 请求怎样进入服务，P/D 如何调度 | [LLM 推理服务：PD 部署与调度](<./04 LLM推理服务：PD部署与调度.md>) | client、API server、router、scheduler、Prefill、Decode、SLA、尾延迟 |
| GPU、节点、机柜、交换机是什么层级 | [GPU 集群硬件与 RDMA 网络层级](<./06 GPU集群硬件与RDMA网络层级.md>) | GPU、CPU、DRAM、SSD、NIC/HCA、PCIe、NVLink、IB、RoCE、RDMA |
| NIC 到交换机之间到底接了什么 | [GPU 集群网络物理层](<./07 GPU集群网络物理层：NIC、光模块与线缆.md>) | DAC、ACC、AOC、光模块、光纤、端口、距离 |
| A100/H100/H200/B200 有什么区别 | [NVIDIA GPU 架构、芯片与量化格式速查](<./05 NVIDIA GPU架构、芯片与量化格式速查.md>) | Ampere、Hopper、Blackwell、HBM、L2、FP8、FP4、AWQ、GPTQ |
| NVL72、NVSwitch、机架级 NVLink 是什么 | [NVL72：机架级 NVLink 系统](<./08 NVL72：机架级NVLink系统.md>) | GB200、GB300、NVLink Fabric、scale-up、scale-out、NVSwitch Tray |

## 三条建议阅读路径

### 从 nano-vLLM 源码出发

```text
example.py 与 Scheduler
  → ModelRunner 与 Attention Kernel 数据链路
      → 推理优化仓库分层
          → PD 部署与调度
```

适合当前项目的学习主线：先看单机引擎内部，再扩展到 Serving 和集群。


### 从 PD 推理服务出发

```text
PD 部署与调度
  → GPU 集群硬件与 RDMA 网络层级
      → NIC、光模块与线缆（物理层细节）
      → NVL72（特殊的机架级 scale-up 系统）
```

适合先理解“一个请求怎么跑”，再逐层下钻到硬件和网络。

### 从 GPU 硬件出发

```text
NVIDIA GPU 代际速查
  → GPU 集群硬件与 RDMA 网络层级
      → NVL72
  → PD 部署与调度（回到服务应用）
```

适合先区分芯片、板卡、服务器和机架，再理解它们如何服务 LLM 推理。

## 一张图串起来

```mermaid
flowchart LR
    C["客户端请求"] --> S["API Server / Router"]
    S --> P["Prefill GPU"]
    P -->|"KV Cache<br/>IB/RoCE + GPUDirect RDMA<br/>或 NVLink"| D["Decode GPU"]

    G["GPU 架构与显存<br/>H100 / H200 / B200"] --> P
    G --> D
    H["节点 / NIC / 交换机<br/>硬件与网络层级"] --> P
    H --> D
    W["线缆 / 光模块<br/>网络物理层"] --> H
    N["NVL72<br/>机架级 NVLink Fabric"] -.特殊系统形态.-> H
```

## 这组卡片的边界

- **总览卡**回答“它是什么、位于哪一层、和相邻概念有什么关系”。
- **专题卡**保留必要的数量级、路径图和官方资料，不堆实现代码。
- **源码解读卡**沿 `example.py → Scheduler → ModelRunner → Attention` 展开；基础设施卡负责服务、硬件和网络概念。

