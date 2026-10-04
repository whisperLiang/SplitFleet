# 四任务真机实验

本目录只保留这次完整边缘模型实验需要的训练、准入、profile、汇总与统计代码。正式比较包括 FedAvg、FedProx、固定25%/50%/75% SFL 和 SplitFleet；每个模型两种子、十轮，共48项作业。

| 任务 | 模型 | 数据集 | 客户端 | batch | Adam lr |
|---|---|---|---:|---:|---:|
| 图像分类 | ResNet-50 | CIFAR-10 | 6 CPU/GPU | 4 | 1e-4 |
| 文本分类 | BERT-base | AG News | 3 GPU | 1 | 2e-5 |
| 目标检测 | RF-DETR Nano | VOC2007 | 6 CPU/GPU | 1 | 1e-5 |
| 语义分割 | DeepLabV3-ResNet50 | Oxford-IIIT Pet | 6 CPU/GPU | 2 | 1e-4 |

每项作业使用240个训练样本、200个评估样本，完整本地epoch、全员参与、保留尾批，每轮训练后评估一次。固定与自适应SFL使用相同的状态所有权交换政策。

本地文件：完整协议 `plans/four_task_study.md`，时间与质量结果 `paper/evidence/four_task_training_comparison_20261003.md`，论文 `paper/README.md`。这些文件不随代码仓库上传。

安装依赖：

```bash
uv sync --extra dev --extra experiment --extra integration
```

入口及职责：

| 入口 | 职责 |
|---|---|
| `physical_multitask.py` | 模型与数据包、FL/SFL客户端及服务器 |
| `orchestrate_physical_multitask.py` | 六方案执行、远程源码核对、同步屏障 |
| `edge_model_admission.py` | 原生训练、分割训练和数值/资源准入 |
| `profile_split_execution.py` | 协调端切点成本profile |
| `collect_device_split_profiles.py` | 各Orin客户端成本profile |
| `run_edge_standard_study.py` | 准入、profile与逐模型训练队列 |
| `www2027_study.py` | 冻结源码、配对种子执行与分析 |
| `summarize_physical_multitask.py` | 完整性核对与物理作业汇总 |
| `common/workload_training.py` | 共享批次处理、训练及任务评估 |

核验保留实验，无须重新训练或改写原始结果：

```bash
.venv/bin/python paper/evidence/verify_four_task_study.py
```

复现原始作业须使用保留的冻结源码及输入计划；清理后的工作区源码具有新的身份，不能替代原实验的源码快照。原始部署、数据分区、检查点、profile及逐作业日志都保存在正式队列中，见完整协议。
