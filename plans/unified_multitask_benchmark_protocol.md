# SplitFleet 统一多任务基准协议 v1

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-23T00:00:00+08:00
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1
- Upstream Dependencies: SplitFleet commit `5c1e771`; unified task validation schema v1

> 本文是待执行的预注册式实验计划，不是结果报告。数值门槛是判定规则，
> 不是预期结果；未达到门槛时必须保留原始结果并将相应假设报告为不支持。

## 研究问题与可证伪假设

### RQ1：统一性是否是可验证的运行时能力？

统一性不以“代码中存在适配器”计数，而以下列四项同时成立为准：

1. 同一公开 `TaskSpec` 契约覆盖图像分类、文本分类、目标检测和语义分割；
2. 同一分割引擎能枚举模型捕获图中的合法算子级 before/after 切点；
3. 每个声称支持训练的切点都通过完整输出、任务损失、全部参数梯度、一步更新和双向线格式对照；
4. 不支持的模型、控制流、形状或切点被明确拒绝，而不是回退到另一个切点后记为成功。

**H1（功能主假设）**：对每个纳入的后端×任务×模型组合，所有被运行时标记为
`training_supported=true` 的测试切点均通过数值对照；任何失败都会否定该组合的完整支持声明。

### RQ2：逐客户端、逐轮自适应切点是否改善异构联邦训练？

**H2（系统主假设）**：在共享初始模型、数据分区、客户端采样、优化器、批预算和
资源轨迹的配对 seed 中，`cosplit_ucb` 相对 `best_global_fixed`：

- 平均轮次时间至少降低 15%；
- p95 客户端完成时间至少降低 20%；
- OOM/超时完成率不劣于基线；
- 同时满足 H3 的任务质量非劣性。

H2 是合取门槛。只降低时延但损害学习质量，或只在一个方便的固定切点上取胜，
都不能支持“整体优越”。

### RQ3：系统收益是否以学习质量为代价？

**H3（质量主假设）**：最终任务指标的配对差值
`SplitFleet - baseline` 的 95% bootstrap 置信区间下界不低于预设非劣性界限：

| 任务 | 主指标 | 非劣性界限 |
| --- | --- | ---: |
| 图像分类 | accuracy | -0.005 |
| 文本分类 | macro-F1 | -0.005 |
| 目标检测 | mAP@0.5 | -0.010 |
| 语义分割 | mIoU | -0.010 |

这些界限在正式结果生成前冻结；pilot 只能用于修复契约和测量故障，不能据其调宽界限。

## 实验对象

| 任务 | 数据集 | 默认模型 | 主指标 | 次指标 |
| --- | --- | --- | --- | --- |
| 图像分类 | CIFAR-10 | ResNet-18/GroupNorm | accuracy | macro-F1 |
| 文本分类 | AG News | 小型 Transformer 或 TextCNN | macro-F1 | accuracy |
| 目标检测 | Pascal VOC 2007 | SSDLite320 或同规模自定义检测器 | VOC07 11 点 mAP@0.5 | all-point mAP、完成率 |
| 语义分割 | Oxford-IIIT Pet | LR-ASPP-MobileNetV3 或 U-Net | mIoU | Dice |

VOC07 的 11 点 AP 依据 [PASCAL VOC 挑战论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/PascalVOC_IJCV2009.pdf)；
极小 pilot 子集只验证管线，不与完整测试集的公开榜单比较。

正式矩阵至少使用 PyTorch 完成四任务收敛实验。TensorFlow、JAX、Paddle 和 tinygrad
作为后端能力矩阵单独报告；只有实际完成数据集级收敛实验的组合才计入跨后端效果结论。

## 方法矩阵

经典基线的原始方法依据分别是 [FedAvg（AISTATS 2017）](https://proceedings.mlr.press/v54/mcmahan17a.html)、
[FedProx（MLSys 2020）](https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)
和 [SplitFed（AAAI 2022）](https://ojs.aaai.org/index.php/AAAI/article/view/20825)。
这里比较的是明确冻结的实现和资源设置，不把论文原报告的收益当成本项目实验结果。

| 类别 | 方法 | 作用 |
| --- | --- | --- |
| 经典 FL | FedAvg | 完整模型本地训练与样本加权聚合 |
| 经典 FL | FedProx | 在 FedAvg 上加入冻结的 `mu=0.01` 近端项 |
| 静态 SFL | fixed-early/middle/late | 三个预先冻结的静态切点 |
| 静态 SFL | best-global-fixed | 只由独立 profile 集选择的最佳统一切点 |
| 异构 SFL | static-heterogeneous | 按设备档位固定切点，不随轮次变化 |
| CoSplit-UCB 消融 | cosplit_ucb_no_global_solver | 每客户端独立选择，用于量化联合 solver 的贡献 |
| 提议方法 | cosplit_ucb | cooperative online learning、共享服务器排队与安全探索 |
| 上界（非基线） | oracle | 资源阶段首轮穷举，仅作 regret 参照，不参加优越性检验 |

SplitFed V1 的同步、双侧更新与轮末聚合由静态 SFL 组覆盖。V2/V3 改变服务器更新顺序
或客户端参与机制，会同时改变优化算法和系统调度，不纳入 H2 主比较；如实现，应作为
标明探索性的次级实验，不能替换 `best-global-fixed` 主基线。

## 公平性与控制变量

- 每个 seed 的方法共享初始模型哈希、数据内容哈希、Dirichlet 分区哈希、客户端采样顺序、
  本地 epoch、批大小、优化器和评估集。
- profile、pilot 与 confirmatory 结果使用不同 run-id 前缀和目录；profile 数据不得来自正式训练结果。
- 同一任务内各方法消耗相同的最大本地 batch 数和样本预算。失败记录为失败，不插值、不填零。
- `best-global-fixed` 只能由 profile/calibration 数据选择；不能从最终结果倒推。
- 硬件时延采用同机配对执行并轮换方法顺序；多主机实验保存时钟、版本、GPU、内存和网络记录。
- 轮次与客户端不是独立统计样本。确认性推断的统计单位是 seed。

## 样本量与执行顺序

- 每任务、每方法至少 8 个配对 seed，正式目标为 10 个。精确双侧符号翻转检验在
  5 对 seed 全部同向时最小 p 值仍为 0.0625；若同一任务/指标族要对三个主基线
  做 Holm 校正，8 对全同向的最小校正 p 值约为 0.0234。
- 默认 100 联邦轮；若任务采用不同轮数，必须以相同的样本更新预算为对照并在协议中先行冻结。
- 每个任务先执行 1 个 pilot seed；pilot 不进入统计汇总。
- 正式执行顺序按 seed 分块并在块内随机化方法顺序，降低温度、缓存和时间漂移偏差。

## 统计分析

1. 对每个任务×基线计算配对 seed 差值、相对差值、Cohen's \(d_z\) 和确定性
   percentile bootstrap 95% CI（20,000 次）。
2. 两侧显著性使用精确配对符号翻转检验；超过 20 对时采用固定 seed 的 100,000 次 Monte Carlo。
3. 在每个任务×指标族内使用 Holm 校正控制 family-wise error。
4. 质量指标按 H3 做非劣性判断；系统指标同时报告绝对值与比例，不只报告 p 值。
5. 预先报告每个失败、OOM、超时与缺失项；不做结果插补。

### “优越性”判定

- **支持**：H2 的全部系统门槛成立，H3 对所有四任务成立，并且无未解释的完成率下降。
- **部分支持**：只在部分资源阶段或部分任务上成立；逐项限定结论。
- **不支持**：任一主质量非劣性失败，或总体系统门槛未达到。

## 预期产物

| 产物 | 路径 | 成功标准 |
| --- | --- | --- |
| 功能矩阵 | `results/unified_task_validation.json` | 机器可读、无 failed；unsupported 单列 |
| 原始实验记录 | `results/cosplit_ucb/<run_id>/` | 通过严格 validator，无插补 |
| 配对统计 | `results/cosplit_ucb/aggregated/paired_comparisons.csv` | 含 effect、CI、精确 p、Holm 与非劣性字段 |
| 复现报告 | `docs/experiment_validation_report.md` | Material Passport + 11/11 fallacy scan |

## 可复现入口

```bash
# 功能正确性：四个目标场景；当前矩阵还额外覆盖实例分割
uv run --no-sync python -m splitfleet.validation \
  --backends torch jax --all-nodes \
  --output results/unified_task_validation.json

# CoSplit-UCB 在线 smoke 不要求完整离线 profiling
uv run --no-sync python -m experiments.cosplit_ucb.run_suite \
  --config experiments/cosplit_ucb/configs/smoke.yaml \
  --run-prefix smoke --resume

uv run --no-sync python -m experiments.cosplit_ucb.aggregate_results \
  --results-root results/cosplit_ucb \
  --output results/cosplit_ucb/aggregated
```

## 当前证据边界

截至本协议创建时，仓库已有合成任务数值等价与真实架构回放证据，但尚无四任务×多方法×
多 seed 的完整确认性结果。因此当前只能声称“统一机制已通过所列正确性测试”，不能声称
“在四任务上统计显著优于经典 FL/SFL”。完整训练结束并通过上述合取门槛后才能升级该结论。
