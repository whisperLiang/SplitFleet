# 统一分割联邦学习框架

SplitFleet 将任务、深度学习后端、模型分割和联邦训练分开：任务适配器描述一次模型调用和训练目标，后端适配器管理模型状态与张量编码，TorchLens 2.34.1 负责捕获计算图与执行分割，Flower 负责客户端选择、轮次和聚合。

## 安装与版本

项目使用 Python 3.11。`pyproject.toml` 和 `uv.lock` 指向本地修正版 `torchlens-2.34.1-1-py3-none-any.whl`，包版本仍为 2.34.1，wheel 构建编号为 1。原始 `torchlens-2.34.1-py3-none-any.whl` 保留为可校验的构建输入；源码补丁和重建方式见 [TorchLens 补丁说明](../patches/torchlens/README.md)。修复在依赖包中生效，无需运行时替换内部函数。

```bash
uv sync --extra dev --reinstall-package torchlens
uv run --no-sync python -c "import torchlens; print(torchlens.__file__); assert torchlens.__version__ == '2.34.1'"
```

需要运行全部框架和真实模型检查时：

```bash
uv sync --extra dev --extra integration --extra multibackend --extra experiment
```

基础安装仍需要 PyTorch；其他框架按 extra 安装，框架适配器按需导入。

## 模块边界

| 模块 | 职责 |
| --- | --- |
| `splitfleet.tasks` | `ModelInputs`、`TaskBatch`、任务损失与样本数 |
| `splitfleet.backends` | PyTorch、TensorFlow、JAX、Paddle、tinygrad 的状态、优化器和张量编码 |
| `splitfleet.split_engine` | 可注册的分割引擎协议及 TorchLens 实现 |
| `splitfleet.autosplit` | 切点诊断、选择、运行时、动态批次范围及缓存 |
| `splitfleet.transport` | 无 pickle 的边界、目标与梯度协议 |
| `splitfleet.client` / `splitfleet.server` | Flower 轮次、前缀/后缀执行与 SplitFed 聚合 |
| `splitfleet.validation` | 与未分割模型比较的任务验证矩阵 |

“多后端”指在各自框架中执行完整的前缀/后缀训练。它不意味着把同一个模型的 PyTorch 前缀直接连接到 TensorFlow 后缀，也不自动把不同模型结构的参数进行联邦平均。

可在不导入 TensorFlow 等大型框架的情况下发现依赖：

```python
from splitfleet.backends import BACKEND_ADAPTERS

for backend in BACKEND_ADAPTERS.availability():
    print(backend.name, backend.available, backend.missing_modules, backend.install_extra)
```

`available` 只表示可发现依赖；动态库能否加载、具体模型是否能捕获、某个切点是否支持训练，仍由运行时和验证决定。`tensorflow` 是 `tf` 的别名。自定义适配器可以通过 `BACKEND_ADAPTERS.register(...)` 注册；新框架还必须有对应的分割引擎支持。

## 细粒度分割

切点可以位于模块内部的算子之前或之后，包括 functional 算子与分支汇合处。跨切点的所有依赖张量构成边界，因此残差或多分支模型可能需要传输多个张量。

```python
import torch
from torch import nn
from splitfleet.autosplit.torchlens_backend import TorchLensSplitBackend

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
backend = TorchLensSplitBackend()
backend.trace(model, torch.randn(2, 4), boundary="50%", dynamic_batch=(1, 8))
report = backend.split_points(diagnose=True).as_dict()
candidates = backend.enumerate_candidates(kinds=("before", "after"))
handle = backend.repartition(candidates[0].boundary)
x = torch.randn(3, 4)
output = handle.backend.run_suffix(handle.backend.run_prefix(x))
torch.testing.assert_close(output, model(x))
```

使用 `before:<节点名>`、`after:<节点名或模块路径>` 或 `50%` 选择位置。显式切点不满足约束时返回错误，只有 `boundary="auto"` 才允许搜索其他位置。候选报告保留不支持的切点及原因。终端切点、仅含整型或形状信息的边界、不可微算子、数据依赖的 Python 路径，不应被当作已通过训练验证的切点。

候选的 `estimated_payload_bytes` 和 `max_payload_bytes` 约束按 TorchLens 原生捕获批次计算，批次记录在候选描述的 `payload_batch_size` 中。动态捕获通常使用 `B=1`；固定形状捕获保留给定形状。估算通过形状程序保留 `B * 10` 等派生维度，统计所有跨切点张量，不包含协议元数据和目标标签；实际训练通信量由轮次的 `upload_bytes`、`download_bytes` 记录。

TorchLens 2.34.1 用 `batch_axes` 描述批次语义，例如 `{"/kwargs/input_ids": 0, "/kwargs/attention_mask": 0}`。路径相对于模型调用的 `args` 和 `kwargs`，数值是批次所在的维度；例如 `[T, B, C]` 的位置输入可以指定 `{"/args/0": 1}`。参数树和其他静态张量不应声明为批次输入。

SplitFleet 的 `dynamic_batch=(min, max)` 负责设备间约定的允许范围；只有原生 TorchLens 批次探测也通过时，运行时才支持范围内的批次变化。扩大范围不会绕过原生拒绝。列表长度、序列长度、图像尺寸、控制流或训练/评估路径发生变化时，可能需要重新捕获。

需要固定输入形状时显式传入 `batch_axes={}`。这与省略参数不同：省略时 TorchLens 推断批次轴，空字典表示没有动态批次轴，按提供的样本形状捕获。

```python
import torch
from torch import nn
from splitfleet.autosplit import prepare_torchlens_runtime

model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 2)).train()
sample = torch.randn(3, 4)
handle = prepare_torchlens_runtime(model, sample, boundary="50%", batch_axes={})
x = torch.randn(3, 4)  # 输入结构、批次和其他维度均与捕获样本一致
output = handle.backend.run_suffix(handle.backend.run_prefix(x))
torch.testing.assert_close(output, model(x))
```

Flower 中，策略和每个客户端都应传入相同的 `batch_axes={}` 并提供形状一致的 `sample_inputs`。数据加载器必须维持捕获形状；不足一个批次的尾批次需要在加载数据时处理，或另行捕获运行时。`partial_batch_policy="skip"` 只处理动态批次窗口之外的批次，不能自动修复固定形状不匹配。`TaskBatch.num_examples` 仍按真实样本数计数，不依赖是否声明了批次轴。

当前版本的训练检查保留了以下原生限制：测试中输入为 `[B, C]` 的 `BatchNorm1d` 和 DeepLab 池化分支的 BatchNorm 在标准 `B=1` 捕获时无法训练；回退到 `B=2` 后若没有独立探测，运行时必须拒绝外推到 `B=3`。Swin 的训练批次探测在测试配置下未通过数值校验，也必须拒绝动态回放。对应训练用例以 `batch_axes={}` 显式捕获实际输入形状完成验证；训练模式的限制不应推断为评估模式也一定失败。

## 统一任务接口

`ModelInputs(args=(...), kwargs={...})` 显式表达模型调用。元组表示多个位置参数；图像列表必须保留为单个参数，例如 `ModelInputs(args=(images,))`。`TaskBatch` 携带 inputs、targets 和 num_examples，避免将图像通道数、框的数量或参数矩阵维度误当样本数。内置图像分类、文本分类和语义分割适配器按标签或掩码的首维计数；显式传入的 `TaskBatch.num_examples` 保持不变。

| 任务 | 适配器 | 默认目标 |
| --- | --- | --- |
| 图像分类 | `ImageClassificationTask` | 稀疏交叉熵 |
| 文本分类 | `TextClassificationTask` | 保留 tokenizer 关键字输入，标签独立传递，稀疏交叉熵 |
| 目标检测 | `DetectionTask` | 模型原生标量损失字典求和，或显式检测 criterion |
| 语义分割 | `SemanticSegmentationTask` | 像素交叉熵，支持 ignore_index 和辅助输出 |
| 实例分割 | `InstanceSegmentationTask` | 检测接口加每个实例的 mask 与原生 mask loss |

默认分类/分割交叉熵约定 logits 为 `[N, C, ...]`，类别轴在第 1 维。采用 channels-last 的模型需要转换输出或提供 `loss_fn`。默认损失支持已注册的五种张量框架；任务可通过自定义 `prepare_batch` 和 `loss` 扩展。直接调用底层 split runtime 时必须显式传入 `loss_fn`，不会根据输出和标签猜测 MSE。

```python
from splitfleet.tasks import TextClassificationTask

text_task = TextClassificationTask()
batch = text_task.prepare_batch({
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels,
})
# batch.inputs.args == ()
# batch.inputs.kwargs 包含 input_ids 和 attention_mask
```

Flower 策略和客户端使用同一个任务定义：

```python
from splitfleet.client.autosplit_split_client import AutoSplitSplitLearningClient
from splitfleet.server.strategy import AutoSplitStrategy
from splitfleet.tasks import ImageClassificationTask

image_task = ImageClassificationTask()
strategy = AutoSplitStrategy(
    model=model,
    sample_inputs=sample_images,
    task=image_task,
    boundary="50%",
    aggregation_policy="splitfed",
    dynamic_batch=(1, 32),
    optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)
client = AutoSplitSplitLearningClient(
    model=model,
    sample_inputs=sample_images,
    task=image_task,
    train_data=train_loader,
    evaluate_data=validation_loader,
    optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
)
```

检测数据使用 `(images, targets)`，targets 为逐图字典列表，每个字典包含 `[N, 4]` 的 boxes 和 `[N]` 的 labels；实例分割额外包含 `[N, H, W]` 的 masks。空目标允许保留为零长度张量传输。`model_loss=True` 时模型接收 images 和 targets，训练输出应是标量损失字典；外部 criterion 模式使用 `DetectionTask(model_loss=False, loss_fn=...)`。检测模型在评估时常返回预测框而非损失，需要明确提供 `evaluation_loss_fn`，不能拿预测均值当检测损失。训练损失和推理预测使用不同图时应分别捕获。

JAX 函数式模型将参数树作为第一个输入，客户端通过 `functional_update_fn` 应用返回梯度。其他框架通过 `optimizer_fn` 创建对应原生优化器。SplitFed 要求客户端模型结构和状态清单一致。

## 验证范围

验证分为三层：

1. 协议和联邦轮次回归：张量形状、元数据、批次窗口、训练/评估模式、梯度回传、客户端/服务端参数重组。
2. 无需下载的合成任务验证：在小型分类、文本、检测和分割网络上比较完整输出、任务损失、参数梯度和一次 SGD 更新，并记录每个切点的结果。
3. 真实架构验证：torchvision 分类/分割/检测头、timm、Hugging Face，以及可选大型检测模型。检测头回放测试不等同于完整检测器的 mAP 或实例分割质量验证。

通过合成数据上的数值一致性不代表已经完成真实数据集的准确率、mAP、mIoU、通信收益或收敛性实验。新增模型和数据集应运行同样的一致性检查，再进行多客户端、多轮性能实验。

基础和后端检查：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tests --ignore=tests/integration
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tests/integration/test_torchlens_optional_frameworks.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tests/integration/test_all_split_nodes_training.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest tests/integration/test_torchlens_real_task_matrix.py
```

多阶段非连续设备放置、跨框架混合切分以及所有未知模型的无条件分割不在当前实现范围内。对合法计算图的算子级分割、任务统一接口和跨后端一致性验证才是可检查的支持边界。

## 一条命令运行任务矩阵

```bash
uv run --no-sync python -m splitfleet.validation \
  --backends torch jax --all-nodes \
  --output results/unified_task_validation.json
```

此命令需要安装 JAX extra，例如先运行 `uv sync --extra dev --extra jax`；只安装基础依赖时使用 `--backends torch`。

也可运行 `examples/unified_task_validation.py`，参数相同。默认仅检查 PyTorch，并从每个模型中采样至多 4 个切点；`--all-nodes` 检查所有枚举的 before/after 切点。每个后端在独立进程中执行，默认每个进程限制为一个计算线程，可通过 `--timeout` 限制每个后端的运行时间。

JSON 保存每个任务、每个切点的结果、误差和能力诊断。退出码 0 表示全部通过，1 表示失败，2 表示存在不支持或仅部分完成的项目。任务完整数值等价矩阵当前覆盖 PyTorch 和 JAX；TensorFlow、Paddle、tinygrad 的支持由独立原生训练/回放和全节点测试验证，不能将两者写成五后端×五任务均已完整验证。

本次合成验证（seed=2026，TorchLens 2.34.1）的通过数量：

| 后端 | 图像分类 | 文本分类 | 检测 | 语义分割 | 实例分割 | 总切点数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch | 8 | 24 | 12 | 6 | 14 | 64 |
| JAX | 8 | 26 | 10 | 6 | 12 | 62 |

每个通过项同时检查完整输出、任务损失、所有参数梯度、一次 SGD 更新，以及激活和边界梯度的序列化往返。检测 fixture 使用每图一个监督对象的分类与框回归，实例分割额外使用 mask BCE；这些是用于验证训练机制的小模型，不能代表 Faster R-CNN、Mask R-CNN 或真实数据集上的最终精度。
