# ResNet50 共用模型、数据与验证组件

本包的模型状态契约、CIFAR 数据处理和收敛指标被当前真实设备后端实验及数值回归直接使用。
训练入口见[六客户端真实设备实验](../resnet50_backends/README.md)，当前 PyTorch
100 客户端、50 轮配对结果见[论文证据说明](../../paper/README.md)。

| 模块 | 保留用途 |
|---|---|
| `model_contract`、`torch_model` | 完整 ResNet50-V1.5、参数/BN 状态和规范初始化 |
| `data` | CIFAR 输入、确定性分区和增强 |
| `convergence` | 完整预算与预先指定的连续窗口收敛判据 |
| `gate_torch`、`staged_reference` | 原生训练与切分数值验证 |
| `run`、`workloads`、`native`、`hardware` | 可配置虚拟训练与设备映射，保留其功能回归 |
| `campaign`、`queue`、`protocol`、`concurrency`、`efficiency`、`summarize`、`status` | 配置化虚拟实验的预算、执行和结果检查 |

先前本机 50 客户端、100 轮、双种子的准备批次未启动，已按作者要求删除；
该计划不属于当前论文的实测结果。创建额外虚拟实验时使用新输出目录和显式配置，
不复用该已删除批次。`status` 要求显式提供 `--root`。

当前论文的六客户端后端实验使用独立的
`experiments.resnet50_backends.campaign`，不由本包的虚拟实验队列调度。
原生后端模型夹具位于 `tests/integration/resnet50_reference/`。

删除内容、哈希和保留范围见[清理收据](../../paper/evidence/manuscript_cleanup_20261023/receipt.json)。
