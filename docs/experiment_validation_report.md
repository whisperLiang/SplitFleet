# SplitFleet 多任务实验验证记录（2026-09-23）

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-23T00:00:00+08:00
- Verification Status: ANALYZED
- Version Label: validation_v1
- Source: 本工作区代码与 `results/` 原始记录

## 验证结论

当前已验证统一框架的多任务执行与数值正确性，但**没有证据支持“在四任务上优于经典 FL/SFL”**。
原因是 CIFAR-10、AG News、VOC 2007、Oxford-IIIT Pet 虽已跑通真实数据，
但尚未完成配对多 seed、充分轮次和异构设备训练；本轮性能数据只有单 seed 的小规模 pilot。

### 2026-09-23 三台真实 Orin 部署 smoke（失败，未计入有效 run）

四客户端物理协议中的 Windows 节点 `192.168.66.136` 当时不可达，故按用户同意
改用 `192.168.66.140`、`.118`、`.238` 三台 Orin 和服务器 `.205` 执行单轮、
CPU、每客户端两批的极小合成模型 SplitFed smoke。三客户端均成功到达服务器，
但都在训练前报错：`TorchLens split id mismatch: prepared after:linear_2_3,
expected after:linear_2_3:1`。三个 SSH 客户端进程退出码均为 1；服务器报告
“3 suffix result(s) but no client update”，经用户授权只终止本次服务器 PID，
退出码 143。服务器未写出结果 JSON，因此此尝试不属于下表 48 个有效 run。

只读环境核对显示：三台 Orin 的 TorchLens 均为 2.31.0，服务器为 2.34.1；
三台客户端的 `autosplit_split_client.py` 和 `torchlens_backend.py` 哈希相同，
但与服务器当前文件哈希不同。入口 `real_device_splitfed.py` 哈希在四台主机相同。
版本/实现漂移与本次切点 ID 差异相符；在隔离环境核实兼容性并统一运行时前，
不自动重试，不把此失败归因于内存或显存不足。此 smoke 使用合成模型，
即使修复后通过，也不能替代四任务真实数据的物理性能验证。

### 2026-09-23 三台真实 Orin 部署 smoke（修复后通过）

按用户要求同步三台客户端至 GitHub `main` 提交 `5c1e771`，在设备虚拟环境中安装
仓库自带的修正版 TorchLens 2.34.1 和 NumPy 1.26.4 后，原样重跑此前核准的
`experiments.real_device_splitfed` 单轮、CPU、每客户端 2 批、批量 2 的合成模型
smoke。服务器和三台客户端进程均以退出码 0 结束；服务器记录 3 条有效客户端更新、
3 条后缀结果、0 条 fit 失败，实际逻辑 ID 为 `orin140`、`orin118`、`orin238`。
每台客户端处理 4 个样本，三台报告的切点均为 `after:linear_2_3:1`，
未再出现先前的切点 ID 错误。服务器测得单轮实验总耗时 13.08 秒，
这里不是与其他 FL/SFL 方法可比的性能数字。

原始结果为 `results/real_device_splitfed/three_orin_pilot_20260923.json`，
大小 3768 字节，SHA-256 为
`c9bd2f51489f8c3b611a69a54ce740113f64ed8d53f99526e1bce00ba9cec6df`。
结果 JSON 已核验：客户端身份、更新与后缀结果数量、批次数、样本数、有限损失值、
切点 ID 和失败列表均符合本次 smoke 的完成条件。实验使用 CPU 和合成数据；
设备上的旧 CIFAR-10 数据已按用户要求删除。这只验证三设备物理部署链路，
不验证 GPU 性能、四任务真实数据的效果、可扩展性或相对 FL/SFL 的优越性。

| 证据层 | 实测结果 | 可以支持的结论 |
| --- | --- | --- |
| 完整默认回归 | 本轮 391 项收集、退出码 0；统计分组补丁后新增的重点测试另行通过 | 新接口与既有框架回归兼容；重型/GPU 用例仍按原测试开关跳过 |
| 合成任务切点矩阵 | PyTorch/JAX，10 个组合、126 个切点通过，0 failed/unsupported | 已测试切点的输出、损失、梯度、一步 SGD 与线格式等价 |
| 新参考模型更新对照 | 4 个任务的两批次 SplitFed 与全本地参数更新对齐；6 个新增重点测试通过 | 四任务新模型能够真实完成前后缀反向传播 |
| 四任务方法矩阵 | 4 任务×4 方法×1 seed×2 轮，共 16 个 fixture run，经严格校验 | 方法管线接通；fixture 不提供现实任务效果证据 |
| 四个实际数据集 | CIFAR-10、AG News、VOC 2007、Oxford-IIIT Pet 各 4 方法×1 seed×1 轮，共 16 run，经严格校验 | 四种任务的真实数据加载和训练管线接通 |
| 三轮真实 pilot | 四任务各 4 方法×1 seed×3 轮，64 个训练样本、32 个测试样本，各客户端最多 2 batch，共 16 run，经严格校验 | 策略发生切点切换，FedProx 多批次目标接通；未验证收敛 |
| 原 RA-SplitFed CIFAR pilot | 4 方法×1 seed×1 轮，2 客户端，各 1 batch，经原 validator 通过 | 原资源自适应实验器支持新增 FedProx 对照 |

`results/unified_multitask_summary_final.json` 纳入 48 个最终格式的有效 run，
包含 36 个单 seed 方法对照；全部 `paired_n=1`，故置信区间与显著性检验字段为 `null`。
历史中间格式 run 按 `--include-prefix` 排除，未被纳入该汇总。
首次 AG News pilot 错取文件前 64 行，训练样本全属一类；该批
`verified_agnews_real_*` run 已由跨全数据索引、四类 macro-F1 固定分母的
`serial_pilot_real_*` run 取代，原始记录保留但不进入最终汇总。

四任务 fixture 上，提议方法与各基线的最终主指标差值均为 0；同机本地执行中，
自适应切分相对 FedAvg 的总墙钟时间更长。这些计时包括每客户端 TorchLens 捕获与
本地序列化，不反映真实网络或边端/服务器分离部署。三轮 pilot 的四任务主指标在
四方法之间也完全相同；其中检测 mAP@0.5 为 0，明确说明尚未收敛。
SplitFleet 在四任务三轮中均记录了从 50% 到 25%/75% 的切点变化；但两个逻辑客户端
在同一 CPU 上顺序执行，首个客户端的捕获/缓存时延可能触发切换，不能解释为异构
硬件上减少 straggler 的证据。单轮 pilot 更不足以估计收敛或方法差异。
原 RA-CIFAR pilot 的最终 accuracy 均为 0.109375。

三轮真实 pilot 的原始单 seed 对照如下。四方法的主指标在每一行相同；下表只列
FedAvg 与 SplitFleet 的总本机墙钟时间，单位秒，不能当作分布式系统时延：

| 任务 | 三轮主指标（四方法相同） | FedAvg | SplitFleet |
| --- | ---: | ---: | ---: |
| CIFAR-10 accuracy | 0.09375 | 0.281 | 0.890 |
| AG News macro-F1 | 0.17155 | 0.365 | 1.059 |
| VOC07 11 点 mAP@0.5 | 0 | 0.483 | 1.173 |
| Oxford Pet mIoU | 0.19420 | 0.601 | 1.250 |

该 pilot 没有显示 SplitFleet 的本机时间优势；应保留这个负结果。

## 身份与可复现性

- 新 benchmark 在每个 run 保存初始模型、Dirichlet 分区和实际转换后数据内容 SHA-256，
  严格校验器比较同任务同 seed 方法间的三项身份。汇总器还按客户端数、批量、
  本地 batch 预算、训练/测试样本数、学习率、候选切点和采样方式分组，拒绝跨协议配对。
- 原 RA-CIFAR 的四方法 pilot 共享相同 `initial_model_hash`、`partition_hash` 和
  `client_sampling_hash`；未将其与新 benchmark 的数值混算。
- AG News CSV 的来源、行数和原始文件 SHA-256 记录于
  `data/ag_news_csv/source_manifest.json`。Oxford Pet 图片/标注归档的 MD5 分别
  是 `5c4f3ee8e5d25df40f4fd59a7f44e54c`、`95a8c909bbe2e81eed6a22bccdf3f68f`；
  VOC 2007 trainval/test 归档的 MD5 分别是 `c52e279531787c972589f7e41ab4ae64`、
  `b6e924de25625d8de591ea690078ad9f`，均与 torchvision 列值匹配。
  下载中断留下的 Pet 部分归档已单独保留，未用于实验；失败 run 不计入验证结果。
- 本轮数值对照可复现；墙钟时间受本机负载、缓存与安装版本影响，不作跨机器复现判定。
- 一次 AG News split pilot 在与重型回归并行时出现原生进程崩溃；随后独立重跑通过。
  不能仅凭时间重合归因于内存压力，也不能把重试结果写作零失败。
  后续矩阵由 `experiments.unified_multitask.run_matrix` 串行启动，每个 run
  使用独立进程且先校验再继续；下载、训练与完整回归也错开执行。

## 统计解释和 11 项谬误扫描

覆盖：11/11。当前没有正式的多 seed 效应估计、p 值或可信的非劣性判断。

| 项 | 本轮检查结果 |
| --- | --- |
| Simpson 悖论 | 尚无按设备/资源阶段分层的正式结果；不能把 pilot 平均数推广到所有设备。 |
| 生态谬误 | 同机逻辑客户端的结果不能推断真实 Jetson/Windows 设备表现。 |
| Berkson 偏差 | 只保留成功 run 会造成选择偏差；汇总显式列出无效 run。 |
| 碰撞变量偏差 | 不能条件化在成功完成率上后只比较幸存客户端速度；正式报告需同时给失败率。 |
| 基准率忽略 | 检测 mAP 需披露类频次与空类处理；当前 fixture 不能代表 VOC 类分布。 |
| 回归均值 | 当前无以极端性能筛选客户端后的前后对照；正式阶段需保持冻结的分区和设备档位。 |
| 幸存者偏差 | 校验器拒绝缺轮次、缺客户端记录；正式实验仍需保留失败日志。 |
| 多重搜索效应 | 四任务×多方法×多指标会产生多重比较；统计脚本按任务/指标族做 Holm 校正。 |
| 分叉路径 | 协议预先固定主比较、指标与非劣性界限；新增任务族不得追溯改写既有 v2 结论。 |
| 相关不等于因果 | 同机 pilot 的时延差不构成异构硬件上的系统收益因果证据。 |
| 反向因果 | 调度器以历史时延更新切点，需区分“慢导致换切点”与“切点导致变快”。 |

## 尚待完成的确认性实验

依据 [`plans/unified_multitask_benchmark_protocol.md`](../plans/unified_multitask_benchmark_protocol.md)，
需要四任务真实训练集与固定测试集、至少八个配对 seed、预先冻结的轮次/样本预算、
异构设备或受控资源轨迹，以及每 run 的完整校验。只有当系统时间/完成率门槛和
任务质量非劣性同时成立，才能作整体“优越性”结论。
