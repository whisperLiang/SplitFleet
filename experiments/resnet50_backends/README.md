# ResNet50：三个真实 Orin 上的后端收敛验证

本入口补充 TensorFlow、JAX 和 Paddle 的训练正确性证据，使用
`orin140`、`orin118`、`orin238` 三台物理设备和本机协调器。每台 Orin 的 CPU、GPU
各启动独立进程，固定为 `orin140_cpu/gpu`、`orin118_cpu/gpu`、`orin238_cpu/gpu`
共六个逻辑客户端。客户端与服务器硬件沿用论文图 3 的 ResNet-50 部署：
服务器为本机 `192.168.66.205` 的 **RTX A6000 GPU0**，三个后端均执行真实 GPU
后缀计算，执行容量 **C=1**；六个客户端各有独立后缀副本。客户端通过 SSH
反向隧道连接实际 Flower 与 SplitFleet server-model gRPC 服务。
历史 Orin238 CPU 服务器入口及对应试跑已在
[本次清理](../../paper/evidence/manuscript_cleanup_20261023/receipt.json)中移除。
当前本机 A6000 任务以独立进程运行，可跨终端断连持续计算；主机重启仍会中断，
状态查询会明确识别，不将其记作完成或自动重试。

正式预算是完整 ResNet50-V1.5（23,528,522 个可训练参数）、CIFAR-10、50 个
完整联邦轮次、每轮六个客户端全部参与，各执行一个本地 epoch。默认每客户端
7,500 张训练图像、batch size 16，保留最后 12 张的短批次。每方法训练样本呈现
共 2,250,000 次、本地 SGD 更新共 140,700 次。固定 5,000 张验证图像；完整
10,000 张官方测试图像仅用于训练结束后的最终模型和最佳验证模型评估。

当前优先完成各后端 **SplitFleet**，默认 `--methods splitfleet`，暂不运行 FedAvg。
需要对照时可显式指定 `--methods splitfleet fedavg`，使用新输出目录。
两方法的协议共用 seed 233 的
初始化、互不重叠且按类均衡的六个分区、由 sample/client/round/seed 固定的随机增强与
批次顺序、SGD（momentum=0、weight decay=0）及 0.05→0.0005 的 50 轮 cosine
学习率。FP32，关闭 AMP/TF32。单 seed 曲线是描述性结果，不提供统计等价结论。

## 准入和判定

训练前，在协调器和每一台 Orin 的 CPU、GPU 上分别检查三个残差 stage 末端切点。对比
原生完整模型与切分执行的实际 logits、CE、161 个参数梯度、SGD 参数增量和
320 个状态张量；53 个 BatchNorm 计数器必须恰好增加一次。浮点容差固定为
`atol=1e-5, rtol=1e-4`，整数完全一致。激活和边界梯度经过正式序列化接口往返。

新训练默认使用 SplitFleet 的实测 Greedy，为六个 CPU/GPU 客户端分别从合法操作切点
目录中选择分割点，并使用每轮计算、实际链路和服务器服务时间反馈更新成本。
初始化保留 B=2 的小型图捕获，独立使用实际训练批次进行不改变持久模型状态的校准。
客户端使用自身冻结分区的首批数据，服务器使用冻结的客户端 0 分区，因此小预算验证集
无需达到训练批大小。每个新选中的切点在客户端和服务器上，按冻结协议中的完整批次
和短尾批大小接受数值准入（默认正式预算为 B=16、B=12；整除时只检查一种大小，
本地数据不足一个批次时检查实际样本数）；汇总使用同一组大小校验回执。准入
计算单独计时，不归入前缀计算或通信残差。每轮保存独立选点回执、校准和实际执行切点。
调用现有 `AutoSplitStrategy`、`AutoSplitSplitLearningClient`、独立后缀副本和
owned-state SplitFed 聚合。浮点状态按样本加权平均，整数状态取最大值。
历史 stage2 固定实验单独保留；仅显式 `--placement fixed_stage2` 可创建该对照。
本实验用于验证各后端真实训练的正确性和收敛。按真实 CPU/GPU 放置分别记录，
不把不同后端的时间作硬件受控速度排名。
切点的规划预算为 256 MiB，与现有传输 envelope 上限一致。JAX 的目录显式
排除常驻后缀参数的重复传输；边界梯度成本按 JAX VJP 的浮点／复数输入语义
计算，排除在后缀消费的常驻参数梯度，并按实际批大小解析形状。完整原生 ABI
及常驻状态契约继续校验。

JAX ResNet50 对每个实际批次形状缓存原生后缀梯度和前缀 VJP 的 Jaxpr，后续批次
由 JAX 原生解释器逐算子执行，避免重复 Python 图回放和自动微分追踪。缓存不启用
整图 JIT 或融合；Orin GPU 继续设置 `JAX_DISABLE_JIT=1`。权重、输入、标签和返回
梯度均为每次调用的动态参数，BN 状态仍只在真实训练步完成后提交一次。客户端
Greedy 校准与正式执行一样排除常驻后缀参数梯度；JAX 前缀计时等待设备结果完成。

每轮必须收到六个指定 CPU/GPU 客户端的完整更新，真实状态张量设备、样本数、
批次数及数据哈希必须符合
冻结协议。任一训练进程失败即停止该方法，保留失败和已完成轮次。缺失后端、
跳过项、部分训练不能计作验证通过。准入失败不会放宽容差。

沿用验证准确率 70% 的操作性收敛判据：最后连续五轮达到目标，且最后五轮平均
验证 CE 低于前五轮。未达到目标如实标记；训练完整完成与达到收敛目标分别报告。

## 执行

```sh
.venv/bin/python -m experiments.resnet50_backends.campaign prepare \
  --output results/resnet50_backends_orin_NEW \
  --data-root data/cifar-10-batches-py --methods splitfleet

.venv/bin/python -u -m experiments.resnet50_backends.campaign execute \
  --root results/resnet50_backends_orin_NEW
```

所有后端均要求三个 CPU 和三个 GPU 客户端，不允许用重复 CPU 进程替代 GPU。
CPU 进程通过 `CUDA_VISIBLE_DEVICES=-1` 隔离 GPU，JAX 进一步固定 `JAX_PLATFORMS`。
部署前应核实本机和 Orin 的框架环境；GPU 依赖或数值准入失败会保留失败记录。
TensorFlow 使用保留 JetPack GPU 构建的 Python 3.10 环境；其他环境和版本记录在
每个设备的准入回执中。实验分发当前源码和
TorchLens wheel 到独立路径，并核验所有 Python 源文件哈希。
`campaign.json` 保存图 3 原始部署文件及其 SHA256；启动时核对 GPU0 型号，
`server_hardware.json` 记录实际 GPU UUID。每轮日志每 25 个完成批次报告一次客户端进度。

JAX CPU 使用 0.6.2；Orin JAX GPU 使用 0.5.1 和 JetPack 自带 CUDA/cuDNN，并关闭 GPU autotuning；
通用 ARM CUDA wheel 中的 autotuning 辅助内核缺少 Orin 所需的 sm_87 代码。
Orin JAX GPU 同时设置 `JAX_DISABLE_JIT=1`，保留原生 eager 归约顺序；其状态和
计算仍绑定真实 GPU。GPU 配置启用 CPU 设备用于原生 debug callback，不作为模型回退。
Paddle 在 Orin 上使用社区 JetPack 构建 `3.4.0.dev20260119`；
本机协调器使用 `3.3.1`。实际版本均记录在回执中，分别
接受完整模型与切分模型的严格数值准入。
该构建来源为 [JetPack wheel 发布页](https://github.com/SidaWang12/paddle3.0-jetson-jetpack6-build/releases/tag/0.0.0)。
新增依赖均放在独立 overlay，未替换既有 JetPack 或框架环境。

准备、执行尝试和失败输出使用新目录。`campaign.json` 固定源码、设备和方法
顺序；`status.json` 记录当前阶段。每轮模型保存到
`runs/BACKEND/METHOD/checkpoints/`，`latest_checkpoint.json` 原子更新，运行时
`server.json` 和六个客户端日志保留已完成轮次。检查点记录轮末模型；当前入口
不自动断点续训或重试。
训练采用 `latest_and_best` 保留策略：每轮写入
指标和模型状态 SHA256，保留最新轮检查点及最佳验证模型。指标与检查点先
同步到磁盘，再公布原子更新的指针。训练中断不会伪装成已完成。

小预算检查可设置 `--train-samples 360 --validation-samples 10 --test-samples 10
--rounds 2 --batch-size 16`，同时检查每台设备的 CPU/GPU 进程驻留。训练样本数必须为 60
的倍数。小预算始终标记为 smoke，不可作为 50 轮收敛结果。

请求的方法完整训练后生成 `summary/curves.csv`、`summary/summary.json` 和
`summary/convergence.png/.svg/.pdf`。每列对应一个后端，上下分别呈现原始验证
准确率和验证 CE；无插值、拟合、外推或虚构误差带。仅运行 SplitFleet 时独立报告
其真实训练和收敛结果，不要求 FedAvg 文件，也不生成配对完成或方法差异结论。
仅在两方法都完成相同预算时才核验完整配对。

```sh
.venv/bin/python -m pytest tests/test_resnet50_physical_backends.py -q
```

当前状态入口为 `results/resnet50_backends_current.json`。修复后的正式队列仅包括
Paddle、JAX 的 SplitFleet，每项 50 轮；仍使用三台 Orin 的六个 CPU/GPU
逻辑客户端以及 A6000 GPU0、C=1 后缀服务器。
TensorFlow SplitFleet 的已完成 50 轮结果保留在
`results/resnet50_backends_orin6_a6000_50rounds_20261022_v7`，不重跑。

当前执行顺序：

1. `resnet50_backends_orin6_paddle_greedy_50rounds_20261023_v14`：逐客户端选点的正式训练。
2. `resnet50_backends_orin6_jax_greedy_20261023_smoke_v15`：修正边界梯度成本后的实机准入。
3. `resnet50_backends_orin6_jax_greedy_50rounds_20261023_v15`：准入通过后执行正式训练。

Paddle 的固定 stage2 三轮记录保留为性能诊断对照，目录为
`resnet50_backends_orin6_paddle_fixed_50rounds_20261023_v12`；该任务已停止。
当前论文图 6 使用的历史 v8 截面、TensorFlow 50 轮原始结果及其冻结代码保留。
已取消的队列、重复准入和不再使用的早期试跑已删除，路径、哈希和删除前状态
见清理收据；历史元数据中出现的这些路径仅用于说明执行沿革。

TensorFlow 的前缀修复在
`resnet50_backends_orin6_tensorflow_prefix_20261023_smoke_v12` 通过
21 个实机切点及六客户端两轮训练；按作者要求仅验证修复有效，不重跑 50 轮。
Paddle 最小前沿、JAX 常驻权重与梯度成本的诊断分别见
`resnet50_backends_diagnostics_20261023_v1`、
`resnet50_backends_diagnostics_20261023_v2` 和
`resnet50_backends_perf_20261023_v3/ANALYSIS.md`。
`profile_split`、`audit_frontier`、`cpu_prefix_probe` 是这些保留诊断的复核工具，
不参与正式训练启动。

读取当前本机服务器的实时状态：

```sh
.venv/bin/python -m experiments.resnet50_backends.status \
  --root results/resnet50_backends_orin6_paddle_greedy_50rounds_20261023_v14
```
