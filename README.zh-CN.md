<p align="center">
  <a href="README.md">English</a> &nbsp; · &nbsp; <strong>简体中文</strong> &nbsp; · &nbsp; <a href="README.es.md">Español</a>
</p>

<p align="center">
  <img src="assets/migro-launch.svg" alt="MIGRO — 模型不断演进，记忆持续相伴。跨语言模型的持久记忆。" width="1600">
</p>

<p align="center">
  <a href="https://zoumingxi.github.io/MIGRO/">
    <img src="assets/migro-demo-entry.svg" alt="体验 MIGRO demo：先做决定，再找回上下文，看看答案如何改变。点击即可在浏览器中体验，无需安装。" width="1600">
  </a>
</p>

<p align="center">
  <strong>跨语言模型的持久记忆</strong><br>
  让积累的经验拥有共同的存储底座，让使用它的模型拥有可适配的接口。
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2611.15315"><strong>论文 ↗</strong></a> &nbsp; · &nbsp;
  <a href="https://zoumingxi.github.io/MIGRO/"><strong>在线体验 ↗</strong></a> &nbsp; · &nbsp;
  <a href="#记忆连续性的挑战">项目概览</a> &nbsp; · &nbsp;
  <a href="#系统设计">系统架构</a> &nbsp; · &nbsp;
  <a href="#实验结果">实验结果</a> &nbsp; · &nbsp;
  <a href="#已发布组件">代码</a> &nbsp; · &nbsp;
  <a href="docs/system_overview.md">系统说明</a> &nbsp; · &nbsp;
  <a href="docs/performance_protocol.md">评测协议</a>
</p>

---

更换一个智能体的底层模型，也许只需要一个下午。积累值得保留的历史，可能需要几个月。

这些历史承载着决策背后的上下文：用户重视什么，哪些约束仍然有效，计划经历过怎样的调整，以及当前对话之前发生过什么。模型不断演进，如何让智能体继续使用这些经验，就成了一个系统问题。

**MIGRO 是一种让积累的经验跨语言模型宿主持续可用的记忆架构。** 它将持久记忆底座与各个骨干模型读取记忆的接口分开设计，把写入、存储和跨模型访问放在同一个连续性问题中处理。

> **我们的目标：模型升级之后，智能体仍能站在已经积累的经验之上。**

## 记忆连续性的挑战

长期运行的智能体，与支撑它的模型，更新节奏并不相同。历史仍在增长，部署却可能已经更换了骨干模型、分词器或模型家族。每次变化都会带来一个实际问题：**哪些内容应该直接延续，哪些部分必须重新构建？**

MIGRO 从三个层面回答这个问题：

|系统关注点|设计目标|
|---|---|
|**记忆效用**|从积累的历史中找回证据，并用于回答问题。|
|**持久化占用**|紧凑地表示内容和检索地址，分别核算持久状态与共享部署资源。|
|**宿主连续性**|通过宿主专用的读取接口，将已有状态连接到接收端骨干模型。|

这些目标共同决定了架构和评测方式。更小的存档仍然要准确恢复状态；成功连接宿主之后，记忆仍然要能提供有用证据；比较迁移效果时，也必须说明实际执行的是哪一种操作。

## 系统设计

<p align="center">
  <img src="assets/migro-system.svg" alt="MIGRO 参考架构：源宿主写入观察到的经验；持久记忆保存证据、语义地址和编码状态；transport 与校准接口将其连接到接收宿主。" width="1600">
</p>

<p align="center">
  <a href="assets/joint_frontier.png"><img src="assets/joint_frontier.png" alt="记忆联合前沿：Qwen 4B 向 Mistral 7B 交接后继续写入，对比回答质量、私有记忆占用及相对重建保留的质量。" width="1600"></a>
</p>

<p align="center">
  <strong>带上已有经验，继续写入新的经历。</strong><br>
  Qwen 4B → Mistral 7B 交接后继续写入：回答质量、私有记忆占用，以及相对于目标端重建保留的质量。
</p>

### The Handoff · 交互式 demo

Leah 刚更换助手，厨房供应商就发来确认请求。最新邮件里有新的报价、图纸和送货方案；而她的预算上限、现场尺寸和楼宇限制，仍留在四月的往来邮件中。现在由你接手：先做出自己的判断，再找回早先的上下文，重新考虑这个决定。随后，可以对比五种记忆条件下模型实际生成的答案。第二个案例则根据发布提交，重现了我们自己遇到的 demo 下载支持问题。

[![The Handoff — Qwen 到 Mistral 的已记录会话](demo/preview.webp)](https://zoumingxi.github.io/MIGRO/)

[**打开交互式 demo ↗**](https://zoumingxi.github.io/MIGRO/) · [下载离线 demo（.zip）](https://github.com/zOUMINGXI/MIGRO/raw/refs/heads/main/demo/MIGRO-Handoff.zip) · [运行自己的会话](demo/README.md#run-your-own-session)

使用浏览器即可打开在线 demo，无需下载或安装。你可以查看已记录的模型运行结果、切换记忆条件，并检查检索出的证据。要根据自己的输入生成新答案，请使用上方链接中的本地运行环境。

### 持久记忆底座，可适配的宿主连接

MIGRO 将存储的经验，与向特定宿主呈现这些经验的机制分开。

**写入保留与真实历史的联系。** 原生参考写入使用源宿主，根据观察到的内容构建记忆。持久化载荷和元数据保留存储内容与其来源经验之间的对应关系。

**寻址和编码承担不同的任务。** 语义地址用于选择证据。Anchor + Residual 表示与仅 Anchor、完整语义地址两种对照进行比较。持久化编码单独评测，包括源 token ID、地址几何关系和元数据的精确恢复。

**读取接口适配接收宿主。** 词汇 transport 将源表示连接到目标词表和嵌入接口；协议允许时，通过校准适配宿主连接。这样的分解让宿主相关的适配成为系统中明确的一部分。

图示描述的是研究架构。已发布代码包含持久状态后端、精确编码内核和 benchmark 入口。共享检查点和数据集单独提供；详见[运行环境配置](docs/runtime.zh-CN.md)。

[架构与组件约定](docs/system_overview.md) · [实现细节](docs/implementation.md) · [消融配置](configs/ablations)

### 迁移状态，或重建历史

迁移协议区分两种具体操作：

|操作|持久状态|目标端工作|
|---|---|---|
|**State Transfer**|复用已有记忆状态。|将状态连接到接收宿主；允许少量读取接口校准，并单独报告。|
|**Full-History Rebuilding**|根据完整原始历史重新构建记忆。|执行目标端构建流程；需要时包含针对该模型的记忆训练。|

**State Transfer** 指状态迁移，**Full-History Rebuilding** 指完整历史重建。标签对应具体运行：一种方法可以同时拥有状态迁移结果，以及在目标端独立重建的对照结果。重建记忆本身，并不意味着微调骨干模型。

[迁移操作与校准预算](docs/migration_protocol.md)

## 实验结果

一次升级应当保留有用历史，也为下一段经验留出空间。现有结果覆盖记忆效用、更大规模宿主、迁移后继续写入和持久化占用。下方展示已完成的评测组；[带日期的结果索引](results/README.md)记录了协议与待完成项目。

### 记忆效用

| 方法 | LoCoMo | LongMemEval-S | MemOps | BEAM-1M | MAB · 主结果 |
|---|---:|---:|---:|---:|---:|
| MIGRO | 54.13 | 68.40 | 66.30 | 55.83 | 35.16 |
| PlugMem | 41.89 | 43.20 | 35.09 | 34.11 | 7.76 |
| Mem0 | 52.12 | 56.80 | 52.13 | 52.43 | 32.82 |
| TiMem | 49.13 | 48.40 | 50.66 | 51.90 | 18.90 |
| A-MEM | 36.20 | 35.10 | 39.78 | 33.25 | 16.72 |
| LSTMem | 40.64 | 26.00 | 52.59 | 28.44 | 6.56 |
| TransMem | 45.85 | 37.40 | 62.51 | 28.76 | 22.80 |
| Doc-to-LoRA | 14.96 | 12.80 | 15.35 | 30.20 | 11.62 |

分数均采用 0–100 刻度。LoCoMo 使用自定义的全题综合指标（1,986 题）；LongMemEval-S 使用辅助准确率（500 题）；MemOps 使用纵向回答准确率（2,006 题）；BEAM 使用类别聚合指标（700 题）。MAB 使用自定义的四能力宏平均（2,071 题）。这些指标不能相互替代。

**独立的 MAB 预算敏感性实验：** 在相同的 2,000 源 token 上限下，MIGRO 为 **35.14**，TransMem 为 **22.88**；在 8,000 上限下，分别为 **40.30** 和 **28.99**。这些保留的预算敏感性实验，与论文 MAB 主结果表分开统计。其他方法保留原生读取路径；相同源 token 上限不代表推理计算量相同。

[Benchmark 汇总 · CSV](results/performance_snapshot.csv) · [JSON](results/performance_snapshot.json) · [评分定义](docs/performance_protocol.md)

### 更大的宿主，已有的记忆

|接收端 Qwen 宿主|MIGRO · 从 4B 迁移状态|MIGRO · 完整历史重建|
|---|---:|---:|
|8B|55.52|55.68|
|14B|60.37|60.32|
|32B|62.33|61.83|

另一条异构链路 **Qwen 4B → Ministral 14B → Gemma 26B-A4B** 在两个接收宿主上的得分为 **57.41 → 66.28**，采用 2,000 源 token 预算，不增加校准样例。两次迁移复用同一份不可变状态，中间不写入新内容。Gemma 26B-A4B 表示 MoE 的总参数量／激活参数量。这条链路与下方历史上的 Phi／Mistral 实验不同。

[同宿主扩展](results/quality_scaling.csv) · [迁移汇总](results/migration_snapshot.csv)

<details>
<summary><strong>全部方法 · 同宿主规模扩展与 Qwen 迁移</strong></summary>

| 宿主 | MIGRO | PlugMem | Mem0 | TiMem | A-MEM | LSTMem | TransMem | Doc-to-LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | 54.13 | 41.89 | 52.12 | 49.13 | 36.20 | 40.64 | 45.85 | 14.96 |
| Qwen3-8B | 55.68 | 34.86 | 54.88 | 46.19 | 44.70 | 19.70 | 46.39 | 12.77 |
| Qwen3-14B | 60.32 | 40.10 | 54.36 | 50.25 | 40.88 | 16.58 | 49.83 | 12.30 |
| Qwen3-32B | 61.83 | 44.66 | 57.16 | 49.24 | 45.38 | 14.35 | 48.03 | 13.18 |

| 方法 | 操作 | Qwen 8B | Qwen 14B | Qwen 32B |
|---|---:|---:|---:|---:|
| MIGRO | State Transfer | 55.52 | 60.37 | 62.33 |
| PlugMem | State Transfer | 44.41 | 46.10 | 49.15 |
| Mem0 | State Transfer | 53.18 | 57.07 | 57.17 |
| TiMem | Full-History Rebuilding | 46.19 | 50.25 | 49.24 |
| A-MEM | Full-History Rebuilding | 44.70 | 40.88 | 45.38 |
| LSTMem | Full-History Rebuilding | 19.70 | 16.58 | 14.35 |
| TransMem | Full-History Rebuilding | 46.39 | 49.83 | 48.03 |
| Doc-to-LoRA | Full-History Rebuilding | 12.77 | 12.30 | 13.18 |

TransMem 的 Qwen 重建数值，是论文迁移对比中使用的同宿主参考结果。

</details>

### 交接之后，继续写入

在 Qwen 4B → Mistral 7B 的继续写入实验中，MIGRO 追加后半段历史后得分为 **45.13**，只迁移不追加为 **36.94**，目标端完整历史重建为 **45.19**。三种条件都按同一份最终完整历史评分。八种方法均分别报告旧历史、新历史和跨边界问题的结果。

<details>
<summary><strong>全部方法的继续写入结果</strong></summary>

| 方法 | 仅前半段记忆 | 继续写入 | 完整历史重建 |
|---|---:|---:|---:|
| MIGRO | 36.94 | 45.13 | 45.19 |
| PlugMem | 33.46 | 37.38 | 36.95 |
| Mem0 | 35.59 | 43.89 | 44.37 |
| TiMem | 32.30 | 37.17 | 40.92 |
| A-MEM | 29.67 | 34.01 | 37.87 |
| LSTMem | 11.34 | 11.15 | 11.16 |
| TransMem | 9.60 | 7.04 | 7.04 |
| Doc-to-LoRA | 11.04 | 10.98 | 11.32 |

所有条件都使用最终完整历史的标签，并在 Mistral 接收端评分。LSTMem 和 Doc-to-LoRA 使用在目标端构建的前缀；TransMem 在追加时重建其记忆视图。其他方法则扩展继承的前缀状态。

</details>

### 跨模型家族访问

论文沿 Qwen 4B → Mistral 7B → Phi 链路展开。接收宿主上的 LoCoMo 分数，对应各方法表中标注的操作。

| 方法 | 操作 | Mistral 7B | Phi |
|---|---:|---:|---:|
| MIGRO | State Transfer | 45.69 | 49.77 |
| PlugMem | State Transfer | 38.60 | 36.30 |
| Mem0 | State Transfer | 44.06 | 44.72 |
| TiMem | Full-History Rebuilding | 41.22 | 28.03 |
| A-MEM | Full-History Rebuilding | 37.73 | 29.57 |
| LSTMem | Full-History Rebuilding | 11.31 | 16.84 |
| TransMem | Full-History Rebuilding | 7.04 | 18.79 |
| Doc-to-LoRA | Full-History Rebuilding | 11.32 | 14.07 |

[迁移汇总 · CSV](results/migration_snapshot.csv) · [JSON](results/migration_snapshot.json) · [操作与接口](docs/migration_protocol.md)

<details>
<summary><strong>评测细节与解读</strong></summary>

LoCoMo 综合指标是在实验后选定的。它对全部 1,986 题赋予相同权重：包括类别 1–4 的 1,540 个题目分数，以及类别 5 的 446 个题目分数。它不是官方 LoCoMo token F1。不同宿主的各行，也不构成严格控制条件下的模型能力排名。

MIGRO 的 AB 参考状态包含 Qwen 写入的前缀，以及随后由 Phi 继续写入的内容。Phi 使用选定的 step-0 恒等接口。Mistral 的读取接口包含 131,072 个参数，在 64 个独立的公开人工 QA 样例上校准，执行 64 次优化器更新。骨干模型和共享记忆核心保持冻结；LoCoMo 答案不作为校准目标。

PlugMem 和 mem0 的 State Transfer 行复用它们在 Qwen 上构建的源状态，其源端轨迹与 MIGRO 的 AB 参考不同。要讨论迁移损失，需要匹配目标骨干模型的重建参考。

LongMemEval-S 和 MemOps 使用已记录的辅助判分协议。BEAM 按类别评分，在需要时使用辅助判分。这些结果来自研究评测框架，并非官方排行榜复现。各 baseline 保留其声明的原生读取路径；推理计算量没有统一。

</details>

### 紧凑记忆，更低的增长成本

质量与字节占用在同一批被评测状态上配对统计。LoCoMo 存储量对十个对话状态求和；BEAM 存储量按每份百万 token 历史统计。

| 方法 | LoCoMo | 独立占用 MB / 10 个状态 | BEAM MB / 每份历史 | 500K–1M 字节/token |
|---|---:|---:|---:|---:|
| MIGRO | 54.13 | 1.115 | 1.857 | 1.809 |
| PlugMem | 41.89 | 92.078 | 82.871 | 79.608 |
| Mem0 | 50.45 | 160.671 | 125.499 | 123.093 |
| TiMem | 47.22 | 10.851 | 6.166 | 6.178 |
| A-MEM | 35.29 | 23.340 | 23.191 | 23.639 |
| LSTMem | 40.64 | 1.871 | 4.371 | 4.370 |
| TransMem | 45.96 | 1.200 | 7.092 | 7.778 |
| Doc-to-LoRA | 14.96 | 283.140 | 56.625 | 28.313 |

MIGRO 每份百万 token 历史平均占用 **1.857 MB**。当必需资源由 **1,000 份独立记忆**共享时，计入固定部署资源后，平均每份记忆为 **2.154 MB**。共同的增长统计窗口为 **500K–1M 源 token**。

<details>
<summary><strong>共享资源估算 · 100–100,000 份独立记忆</strong></summary>

必需资源只存储一份并共享时，每份百万 token 记忆的平均占用（MB）：

| 方法 | 100 份记忆 | 1,000 | 10,000 | 100,000 |
|---|---:|---:|---:|---:|
| MIGRO | 4.829 | 2.154 | 1.887 | 1.860 |
| PlugMem | 84.237 | 83.008 | 82.885 | 82.873 |
| Mem0 | 127.018 | 125.651 | 125.514 | 125.500 |
| TiMem | 7.532 | 6.303 | 6.180 | 6.167 |
| A-MEM | 24.557 | 23.328 | 23.205 | 23.193 |
| LSTMem | 4.775 | 4.411 | 4.375 | 4.371 |
| TransMem | 24.292 | 8.812 | 7.264 | 7.109 |
| Doc-to-LoRA | 73.465 | 58.309 | 56.794 | 56.642 |

</details>

[存储汇总 · CSV](results/storage_summary.csv) · [JSON](results/storage_summary.json)

### 机制检验

实验约定将完整系统的问题，与单个组件的问题分开。

|研究问题|受控对比|配置|
|---|---|---|
|记忆系统能否提供有用证据？|完整记忆、BM25、近期历史和无记忆对照，并以目标端原生重建作为参考。|[核心迁移](configs/ablations/core_transfer.json)|
|宿主接口贡献了什么？|边界组合与冻结校准的交叉对照。|[Transport 与校准](configs/ablations/transport_composition.json)|
|地址空间与质量如何权衡？|Anchor + Residual、仅 Anchor 和原始完整地址。|[地址表示](configs/ablations/addresses.json)|
|持久化编码节省了什么？|完整无损编码选择、既有字面编码候选和未压缩存档。|[持久化编码](configs/ablations/persistence.json)|

论文报告了以下匹配对照。LoCoMo 数值使用局部完整机制参考值 54.21，与主表中的 54.13 分开统计。

| 机制 | 完整机制 | 简化对照 |
|---|---:|---:|
| 证据选择 | 54.21 | 50.48 关闭宿主关联; 44.82 语义 Top-K |
| 语义地址 | 54.21 · Anchor + Residual | 51.73 · 仅 Anchor |
| Mistral 边界组合 | 45.48 | 39.98 关闭边界组合 |
| 存档编码 | 0.870 MB | 0.953 MB 既有无损编码; 2.381 MB 未压缩 |

与 float32 相比，Anchor + Residual 将地址表字节数减少 **91.1%**，质量相差 **0.46 分**。核心迁移对照也优于 BM25、近期历史窗口和无记忆条件；报告中直接读取相对 BM25 的增益，在 **Phi 上为 11.74 分**，在 **Mistral 上为 3.93 分**。

## 已发布组件

仓库提供实现代码和评测流程的入口。

|组件|作用|源码|
|---|---|---|
|**持久状态**|共享 Anchor 与 Residual、精确内容绑定、保存／恢复及继续写入。|[`migro.memory`](src/migro/memory.py)|
|**精确编码**|字面与因果内容编码候选、地址／存档编码，以及原生内核。|[`migro.backend`](src/migro/backend)|
|**Benchmark 运行**|增量 MAB、匹配读取预算、格式感知评分与迁移后继续写入。|[运行实验](docs/experiments.md)|
|**Transport**|词表感知的分段 transport、边界组合与词汇映射。|[`migro.transport`](src/migro/transport.py)|
|**校准**|残差接口与以证据为条件的宿主接口。|[`migro.calibration`](src/migro/calibration.py)|
|**参考写入**|基于观察历史的冻结宿主因果写入。|[`migro.native_writer`](src/migro/native_writer.py)|
|**观察内容处理**|确定性格式化与时间标注。|[`migro.observations`](src/migro/observations.py)|
|**记忆对照**|可移植的 BM25 与近期窗口 baseline。|[`migro.controls`](src/migro/controls.py)|
|**配对分析**|匹配评测组对比与对话级 bootstrap 区间。|[`migro_eval.ablations`](src/migro_eval/ablations.py)|

### 安装

使用 Python 3.10 或更高版本。结果汇总和可移植文本对照使用标准库；transport 与 analysis 可选依赖安装对应模块所需的软件包。

```bash
python -m pip install -e .
python -m pip install -e '.[transport,analysis]'
```

要使用记忆后端和 benchmark 入口，请安装 `.[runtime,benchmarks]`、编译随附内核，并配置共享模型资源。[运行环境配置](docs/runtime.zh-CN.md) · [运行实验](docs/experiments.md) · [组件约定](docs/implementation.md)

### 结果汇总与配对分析

```bash
# 展示仓库中的汇总结果。
python scripts/render_results.py

# 汇总已经完成评分的 LoCoMo 评测组。
python scripts/summarize_locomo.py /path/to/scored_records.json --output summary.json

# 查看配对消融分析接口。
python scripts/summarize_ablations.py --help

# 运行组件与结果汇总检查。
python -m unittest discover -s tests -v
```

汇总工具读取已评分的记录，默认要求完整评测组。逐题评分仍由上游完成。可选依赖不可用时，对应测试会跳过。

[汇总输入格式](docs/reporting_tools.md) · [配对分析用法](docs/implementation.md#paired-analysis)

### 仓库结构

```text
assets/                 视觉标识与系统图
src/migro/              Transport、校准、写入与对照
src/migro_eval/         已评分记录汇总与配对分析
configs/ablations/      组件级实验约定
docs/                   架构、协议与实现说明
results/                精选论文汇总表（CSV / JSON）
scripts/                结果汇总与分析入口
tests/                  组件与结果汇总检查
```

## 发布状态

仓库正在逐步完善开源发布内容。目前包含共享 Anchor 记忆后端、精确编码源码、宿主接口、MAB 数据准备／生成／评分与 MIGRO 继续写入入口，以及实验约定和汇总工具。共享核心与接口检查点已在[运行资源 Release](https://github.com/zOUMINGXI/MIGRO/releases/tag/runtime-assets-20261112)中提供，并附有组件来源和许可声明；[运行环境配置](docs/runtime.zh-CN.md)给出了下载与验证命令。骨干模型和数据集需单独获取。完整 baseline 适配器与其余 benchmark 编排仍在整理中，当前发布尚不是覆盖全部实验的独立复现包。

[精选结果汇总](results/README.md)与当前论文保持一致，覆盖主要 benchmark、同宿主规模扩展、迁移和匹配状态存储。论文中仍待完成的对比不填写最终分数。详细实验数据和运行产物留待后续发布。

MIGRO 自有代码和运行资源采用 [Apache-2.0](LICENSE) 许可。随包上游组件和外部模型，保留[第三方声明](THIRD_PARTY_NOTICES.md)中列出的各自许可。

[实现范围](docs/implementation.md) · [发布范围](docs/release_scope.md) · [第三方参考](docs/third_party.md)
