# PyTorch ResNet50：50 个虚拟客户端的收敛与用时对比

本实验对比 **SplitFleet（CoSplit-UCB）与 FedAvg**，固定 PyTorch 后端、完整
ResNet50-V1.5（23,528,522 个可训练参数）、**50 个虚拟客户端、100 轮**。
使用 CIFAR-10、从头训练、FP32。每轮全部客户端参与，各训练 900 张图像，
完整覆盖 45,000 张训练图像；每客户端 batch size 为 16，包含最后 4 张的短批次。
每方法每 seed 共 4,500,000 个样本呈现、285,000 次本地 SGD 更新。

使用配对 seeds 233、234，共 4 次独立的 100 轮训练：233 的 FedAvg → SplitFleet，
然后 234 的 SplitFleet → FedAvg。两种方法共享初始化、IID 类别均衡分区、按
sample/client/round/seed 固定的数据增强、SGD（momentum=0、weight decay=0）
及 0.05 → 0.0005 的 100 轮 cosine 学习率。保留原始 64×64 输入预处理。
旧多后端、多客户端数量和 10 轮预跑阶段已移除。

每轮在固定 5,000 张验证图像上记录 CE、top-1 和验证结束时的实际累计耗时。
最终模型及验证集选择的最佳模型都在完整 10,000 张测试图像上评估。测试集
不参与目标选择、学习率调整或最佳检查点选择。

## 判定与展示

冻结协议时预先指定验证准确率目标，默认 **70%**。本实验将“收敛”操作性定义为：
最后连续 5 轮验证准确率均达到目标，且最后 5 轮平均验证 CE 低于前 5 轮。
这是有限训练预算内的判据，不能解释为数学上的收敛证明。

达到目标所需时间取**首次连续 5 轮达到目标时，第 5 次验证结束的实测时刻**，
不插值、不外推；未达到目标记为缺失。100 轮总训练时间使用同一起点的独立
wall clock，覆盖运行时初始化、模型构建、Ray 启动、捕获、CoSplit 校准和决策、
训练、RPC、状态同步、聚合、每轮验证、检查点和运行时清理。解释器启动、
实验模块导入、冻结输入验证和最后两次测试集评估在该时钟之外。

汇总同时输出每个 seed 的原始值、配对速度比 `FedAvg 秒数 / SplitFleet 秒数`、
时间减少百分比和配对 log-speedup 的 95% t 区间。只有两方法在两个 seed 中均
满足上述收敛判据，且总训练 speedup 区间下界大于 1，才将
`both_converged_and_faster` 标为 true。最终测试准确率差异及 ±1pp 区间判断
另行报告。两个 seed 的区间精度有限。完整执行成功与研究目标成立分别记录；
更慢、未达目标或差异较大都会如实输出。

## 硬件

保留相同的真实异构硬件分配：10 个客户端在 GPU0，40 个客户端在单核 CPU
亲和性上运行；GPU1 用于协调器和 SplitFleet 后缀。两方法使用相同的设备映射、
Worker 数、资源配额和网络设置。FedAvg 在客户端训练完整模型，SplitFleet
执行真实前缀/后缀计算和 gRPC 边界梯度传输。

默认 **13 个实际 Ray Worker** 调度全部 **50 个逻辑客户端**；可在准备时指定
固定 Worker 数。每个逻辑客户端保留独立后缀状态；后缀驻留和执行槽等于 Worker
数。Worker 数和 GPU fraction 是调度资源配置，不代表 50 个客户端同时计算，
也不代表限制了 GPU 算力。报告保留实测并发及排队时间。

训练失败时停止，不自动重试或调整 Worker 数。失败运行的部分输出按已有清理
策略删除，原因保留于队列；重新执行应使用新的输出目录。框架其他后端的数值
回归夹具位于 `tests/integration/resnet50_reference/`，不参与此实验调度。

## 准备与执行

在仓库根目录运行，使用已有 CIFAR-10 数据与归档（`data/cifar-10-batches-py`
及 `data/cifar-10-python.tar.gz`）。

```sh
uv sync --extra simulation --extra experiment --extra dev

# 仅生成当前实验的冻结输入；不加载旧多后端协议
.venv/bin/python -m experiments.resnet50_quality.protocol \
  --target-accuracy 0.70 --output data/resnet50_torch50/protocol

# CPU 编号需属于当前主机允许的 affinity；这里沿用 CPUs 16–31
.venv/bin/python -m experiments.resnet50_quality.hardware \
  --workers 13 --cpu-cores 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --execution-cpu-cores 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --output data/resnet50_torch50/hardware.json

.venv/bin/python -m experiments.resnet50_quality.campaign \
  --protocol data/resnet50_torch50/protocol \
  --hardware data/resnet50_torch50/hardware.json \
  --data-root data/cifar-10-batches-py --output results/resnet50_torch50

# 执行 4 个配对运行，完成后自动汇总和出图
.venv/bin/python -m experiments.resnet50_quality.campaign \
  --execute results/resnet50_torch50/campaign.json

.venv/bin/python -m experiments.resnet50_quality.status --root results/resnet50_torch50
```

训练从冻结的当前源码子进程执行，保留源码与依赖哈希；不复用历史训练结果。
更改准确率目标或硬件配置需要在首次执行前生成新的冻结输入和输出目录。

## 输出

`summary/` 包含：

- `convergence_comparison.png` / `.svg`：验证准确率–轮数、验证 CE–轮数、
  验证准确率–实际时间，以及完整 100 轮训练用时。按轮数图显示均值和两 seed
  范围；按时间图逐 seed 展示，保留各自实际时钟。
- `paired_validation_curves.csv`：每个 seed、每方法、每轮原始指标与实际累计秒数。
- `pytorch_validation_curves.csv`：按轮数汇总的验证曲线。
- `time_to_target.csv`：各 seed 的目标达成、连续窗口、达到目标所需时间和收敛判断。
- `backend_summary.csv` / `summary.json`：准确率差异、完整训练速度比、达到目标速度比、
  区间及目标是否成立。只有 PyTorch 一项。

单次运行保留 `round_metrics.csv`、客户端/后缀训练记录、检查点、测试集指标及
完整预算校验。仅完整的四个配对运行进入结论。

```sh
.venv/bin/python -m pytest tests/virtual -q
```
