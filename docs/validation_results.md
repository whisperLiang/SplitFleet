# TorchLens 2.34.1 重构验证记录

验证日期：2026-09-22 至 2026-09-23。此记录验证功能和数值一致性，不报告真实数据集精度、联邦收敛速度或通信收益。

## 环境

| 组件 | 本次安装版本 |
| --- | --- |
| Python | 3.11.15 |
| TorchLens | 2.34.1，仓库本地修正版 wheel，构建编号 1 |
| PyTorch / torchvision | 2.11.0 / 0.26.0 |
| TensorFlow | 2.21.0 |
| JAX | 0.6.2 |
| PaddlePaddle | 3.3.1 |
| tinygrad | 0.13.0 |
| Flower | 1.29.0 |
| timm / transformers | 1.0.27 / 5.8.1 |

原始构建输入 `torchlens-2.34.1-py3-none-any.whl` 的 SHA-256：

```text
118fe87e092838664f2daf25c41df6de26b2758e4eeb7445ef636daf9e304993
```

已安装 `torchlens-2.34.1-1-py3-none-any.whl`，其 SHA-256 为 `8f01c47d480059e092f99907ae93d6ed94ae28b0a7f6903191c4d124f23f2e79`。从环境实际导入并核对 `torchlens.__version__` 和构建来源记录。旧版 2.31.0 wheel 已删除，原始 2.34.1 wheel 保留为可校验的补丁构建输入；源码补丁与重建步骤见 [TorchLens 补丁说明](../patches/torchlens/README.md)。重复构建哈希一致，wheel 的所有 `RECORD` 校验值匹配；`uv lock --check` 和 `uv pip check --python .venv/bin/python` 均通过。

## 最终结果

| 检查 | 结果 | 本地记录 |
| --- | --- | --- |
| 默认 CPU 回归（含实验模块） | 360 通过，9 按配置跳过，0 失败；284.15 秒 | `results/pytest_tinygrad_fix_full.log`、`results/pytest_tinygrad_fix_full.xml` |
| CPU→GPU、GPU→GPU 分割训练 | 2 通过，0 失败 | `results/pytest_tinygrad_fix_cuda_final.log`、`results/pytest_tinygrad_fix_cuda_final.xml` |
| PyTorch/JAX 五类任务数值矩阵 | 10 组合、126 切点通过，0 失败 | `results/unified_task_validation.json` |
| 静态检查 | `git diff --check`、Python 编译检查通过 | 本地命令 |

完整回归覆盖真实架构的分类、文本、分割与检测头，以及任务适配器、Flower 轮次、跨进程协议和五后端原生训练检查。tinygrad 的固定输入用例验证全部 24 个非终端切点实际更新参数；解码张量先物化为独立缓冲区，避免惰性复制图导致数值或梯度错误。CUDA 两项在 CPU 回归中计入跳过，随后单独通过，不应重复计为未验证。

本次审查修复增加了 JAX 双 batch Flower 轮次与原生 SGD 的参数及损失对照、参数树和时间维优先输入的真实样本计数，以及 reshape 边界的通信量和约束检查。通信量测试同时覆盖固定形状、动态批次与 JAX 参数透传。

## 原 9 项跳过用例的显式补验

逐项启用后的结果为 **9 项通过，0 项未完成**。以下通过项均检查了实际执行结果，失败、错误和跳过数均为零；机器可读明细见 `results/skipped_validation_summary.json`。tinygrad 的依赖缺陷修复及额外数值对照见下文。

| 用例 | 结果与范围 | 耗时 | 本地日志 |
| --- | --- | --- | --- |
| PyTorch ResNet-18 | 89/89 个切点完成训练及双客户端聚合 | 53.65 秒 | `results/pytest_skipped_resnet_torch.log` |
| TensorFlow ResNet-18 | 530/530 个切点完成训练及双客户端聚合 | 344.58 秒 | `results/pytest_skipped_resnet_tf.log` |
| JAX 卷积残差变体 | 78/78 个切点完成训练及双客户端聚合；stax 实现不含 BatchNorm | 24.06 秒 | `results/pytest_skipped_resnet_jax.log` |
| Paddle ResNet-18 | 169/169 个切点完成训练及双客户端聚合 | 74.62 秒 | `results/pytest_skipped_resnet_paddle.log` |
| tinygrad 18 层标量残差结构 | 119/119 个切点完成训练及双客户端聚合；并非完整卷积 ResNet-18 | 248.57 秒 | `results/pytest_tinygrad_fixed_final.log` |
| YOLOv8n | 固定 B=2 推理数值一致、边界传输往返通过 | 5.57 秒 | `results/pytest_skipped_yolo.log` |
| RF-DETR Nano | 固定 B=2 推理及边界传输通过；eval 模式下反向传播并实际更新参数 | 16.42 秒 | `results/pytest_skipped_rfdetr.log` |
| CPU→GPU、GPU→GPU | 两个跨设备训练用例通过 | 2.66 秒 | `results/pytest_tinygrad_fix_cuda_final.log` |

补验修正了测试接入方式：固定输入显式设置 `batch_axes={}`；逐切点选择使用唯一 `canonical_id`；YOLO 的预热 anchors/strides 声明为非持久缓冲区。检测模型的动态批次支持不在通过范围内。RF-DETR 使用输出张量构造的测试损失，不能据此宣称已验证检测器原生训练目标或真实数据集效果。

测试入口现在传播子进程的跳过状态，导入或构造失败不再被宽泛异常处理转为跳过；ResNet 穷举测试也删除了所有节点未完成训练时改测 `50%` 切点的 fallback，并要求每个可训练切点都实际完成训练。对应隔离入口回归为 4 项通过。

原 tinygrad 阻塞来自 TorchLens 对共享 UOp 图的递归展开。原模型具有 143 个唯一 UOp，按原始字符串格式计算，单条最终签名约 3.11 GiB，全部节点签名累计规模约 14.33 GiB；这是理论字符串规模，不是实测峰值内存。修正版使用每次调用局部缓存的迭代 DAG 遍历，结构签名固定为 71 字节，设备重写也只处理每个可达节点一次。固定形状捕获跳过动态形状重写，动态 batch 路径仍有独立回归测试。

数值验证还发现列表内残差层的参数遗漏及错误缓冲区绑定。修正版补齐容器状态发现，并按准确的捕获张量绑定参数；namedtuple 参数路径与 tinygrad 原生状态字典一致，转置、reshape、切片视图在初始状态和加载新状态后均能正确回放。原始八个残差块、18 个参数和全部 119 个操作切点均保留；所有切点都完成前后缀数值比较、边界传输、双客户端训练和聚合，没有不支持或不可微切点。额外集成测试逐个对照全部 18 个参数的梯度和一次 SGD 更新，与原生 tinygrad 训练一致。

可复现诊断：[tinygrad_signature_cost.py](diagnostics/tinygrad_signature_cost.py)，对照结果为 `results/tinygrad_signature_cost_fixed.json`。脚本计算原始格式长度并直接测量修复后的签名，不分配巨大字符串；严格逐切点训练通过的证据为 `results/pytest_tinygrad_fixed_final.xml`。修复通过可重建的依赖 wheel 提供，无运行时 monkey patch 或 fallback。

```bash
# 重现已通过的 7 个 CPU 重型用例；CUDA 两项使用下方独立命令。
SPLITFLEET_RUN_RESNET18_ALL_NODES=1 SPLITFLEET_RUN_HEAVY_REAL_MODELS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES='' \
uv run --no-sync pytest tests/integration/test_resnet18_all_backends_all_nodes.py \
  tests/integration/test_torchlens_real_detection_optional.py \
  -ra -s --tb=short --disable-warnings

DEV=CPU DEBUG=0 uv run --no-sync python docs/diagnostics/tinygrad_signature_cost.py
```

## 可复现命令

完整 CPU 回归包括 `tests` 与 `experiments/resource_adaptive_splitfed/tests`；可选框架使用子进程隔离，避免多个框架的原生库在同一进程内冲突。

```bash
mkdir -p results
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES='' \
uv run --no-sync pytest -ra --tb=short --disable-warnings \
  --junitxml=results/pytest_tinygrad_fix_full.xml > results/pytest_tinygrad_fix_full.log 2>&1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync pytest tests/unit/test_torchlens_split_engine.py -k gpu \
  -ra --tb=short --disable-warnings --junitxml=results/pytest_tinygrad_fix_cuda_final.xml \
  > results/pytest_tinygrad_fix_cuda_final.log 2>&1

uv run --no-sync python -m splitfleet.validation \
  --backends torch jax --all-nodes --output results/unified_task_validation.json
```

## 任务矩阵

任务矩阵 10 个后端/任务组合、126 个 before/after 切点通过，无失败或未支持项。每个切点比较完整输出、任务损失、所有参数梯度和一次 SGD 更新，激活与边界梯度均经过实际序列化往返。

| 后端 | 图像分类 | 文本分类 | 检测 | 语义分割 | 实例分割 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch | 8 | 24 | 12 | 6 | 14 |
| JAX | 8 | 26 | 10 | 6 | 12 |

该矩阵使用小型合成任务。其他三个后端的原生回放、训练、参数更新和任务目标有独立检查，不能据此宣称已完成五后端 × 五任务的完整端到端矩阵。详细能力边界与接入示例见 [统一框架说明](unified_framework.md)。

## 跳过项与限制

- 默认关闭五个跨后端 ResNet-18 全节点穷举测试，需显式设置 `SPLITFLEET_RUN_RESNET18_ALL_NODES=1`；五个均已补验通过，模型范围见上表。
- 默认关闭 YOLO、RF-DETR 两个大型可选测试，需设置 `SPLITFLEET_RUN_HEAVY_REAL_MODELS=1`；两项已在固定形状下补验通过。
- CPU 命令屏蔽 CUDA，因此两个跨设备测试在该次运行中跳过，随后使用独立 GPU 命令验证。
- BN1d、DeepLab 和 Swin 的特定训练配置保留原生动态批次拒绝行为，固定形状训练通过 `batch_axes={}` 显式捕获，未修改 TorchLens 私有实现以绕过探测。

`results/` 是本地生成目录，不随 Git 默认跟踪；复现命令会重新生成日志、JUnit XML 和 JSON 明细。
