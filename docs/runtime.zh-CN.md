<p align="center">
  <a href="runtime.md">English</a> &nbsp; · &nbsp; <strong>简体中文</strong> &nbsp; · &nbsp; <a href="runtime.es.md">Español</a>
</p>

# 运行 MIGRO

共享运行资源包包含冻结的记忆核心、编码表、分词器，以及 Phi／Mistral 读取接口，并附有组件来源与许可声明。MIGRO 自有代码和资源采用 Apache-2.0；随包上游组件保留原有许可。

## 安装

已验证的环境为 Linux、Python 3.10、C++17 编译器和 PyTorch 2.5.1。请根据硬件选择合适的 PyTorch 构建版本。CUDA 验证使用以下配置：

~~~bash
git clone https://github.com/zOUMINGXI/MIGRO.git
cd MIGRO
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[runtime,benchmarks]' -c constraints/runtime-tested.txt
~~~

如果仅在 CPU 上运行记忆操作，将上方 CUDA 安装源替换为 `https://download.pytorch.org/whl/cpu`，安装 CPU 版 PyTorch。

## 下载并验证共享资源

~~~bash
export MIGRO_ASSETS="$PWD/migro-assets"
python scripts/setup_runtime.py --assets-dir "$MIGRO_ASSETS"
python scripts/verify_runtime.py --assets-dir "$MIGRO_ASSETS"
~~~

配置命令会下载公开资源包，并编译三套 C++ 编码内核。验证命令会加载实际记忆核心和两个接口检查点，依次执行观察写入、状态保存与重载、迁移、追加观察、压缩，以及原始绑定和元数据恢复。

[下载资源包](https://github.com/zOUMINGXI/MIGRO/releases/tag/runtime-assets-20261112) · [组件来源与许可](../THIRD_PARTY_NOTICES.md) · [来源清单](../PROVENANCE.json)

`ASSETS.json` 列出所需载荷文件。检查点导出保留张量数值、数据类型，以及已解析的接口作用域；本地路径、训练配置和实验记录已移除。

## 生成新答案

生成式 demo 需要外部骨干模型和重排序模型。使用配置脚本从发布方下载：

~~~bash
python scripts/setup_runtime.py --assets-dir "$MIGRO_ASSETS" --skip-assets \
  --models qwen mistral reranker
MIGRO_DEVICE=cuda python demo/run.py --case demo/starter.json --output trace.json
MIGRO_DEVICE=cuda python demo/serve.py --trace trace.json
~~~

打开 `http://127.0.0.1:18764`。生成的会话记录包括源宿主回答、迁移状态后的回答、继续写入、仅保留后续消息，以及完整历史重建。可以在本地页面中编辑会话并生成新答案。

两个骨干模型需要数 GB 的模型存储空间。完整的 Qwen 4B → Mistral 7B demo 已在 H800 80GB 上验证。公开浏览器 demo 也提供已记录会话的回放。

使用 Phi 接口时，可增加 `--models phi reranker`。模型通过 `huggingface_hub` 下载；发布方要求认证时，使用其通常的认证配置。

如果已有模型目录，请在导入运行时模块之前配置：

~~~bash
export MIGRO_QWEN_MODEL=/path/to/Qwen3-4B-Instruct-2507
export MIGRO_MISTRAL_MODEL=/path/to/Mistral-7B-Instruct-v0.3
export MIGRO_PHI_MODEL=/path/to/Phi-3-mini-4k-instruct
export MIGRO_RERANKER_MODEL=/path/to/ms-marco-MiniLM-L6-v2
~~~

## Python API

~~~python
from migro.memory import MemoryStore, load_core

core = load_core("cpu")
store = MemoryStore.create(core)
store.observe([{
    "speaker": "Ada", "role": "observed_human", "body": "I moved to Kyoto.",
    "session": "session_1", "date": "1 January, 2026", "dialogue_id": "s1:0"
}])
store.save("state.bin")
receiver = MemoryStore.load(core, "state.bin").handoff()
~~~

`MemoryRuntime.observe()` 提供冻结宿主上的原生写入，需要配置对应宿主模型。相对资源名称用于保持状态存储格式。`MIGRO_NATIVE_DIR` 可指定生成内核的位置，`MIGRO_PROFILES` 可选择其他读取配置 JSON。

## 检查

~~~bash
MIGRO_TEST_ASSETS=1 python -m unittest discover -s tests -v
python -m pip check
~~~

下载包内附有 `ASSETS.json`、`PROVENANCE.json`、`NOTICE` 和许可文件。CPU 状态测试与真实模型生成检查见[运行验证记录](../assets/runtime/validation.json)。

[返回项目首页](../README.zh-CN.md)
