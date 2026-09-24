# 统一分割联邦学习框架：多任务扩展与真机对照实验

## 目标

把 SplitFleet 从「单任务（CIFAR-10 图像分类）真机 SplitFed」扩展为
**统一分割联邦学习框架**：多后端、任意模型任意细粒度切点、四类任务场景，
并在真实设备上对照经典 FL 与 SFL 方法证明其优越性。

## 已核实的现状（不重写既有工作）

框架核心已具备，约 12k 行：

- 任意切点：TorchLens 驱动，`autosplit/planner.py` + `split_engine/`
- 多后端：`backends/` 已注册 torch，README 记录 TF / JAX / Paddle / tinygrad 可选组
- SplitFed 聚合、per-client 后缀副本、能力感知放置、动态 batch 窗口
- 真机实验框架（在 `.136`）：`orchestrate.py` 324 行、`run.py` 1623 行、
  `native_prefix.py` 1040 行、4 个 protocol yaml、A1–A10 修正案、`validate_run.py`
- 拓扑已与目标一致：136/140/118/238 为客户端，205 为后缀 + 聚合服务端

缺口正是本计划要补的三项：

1. **任务覆盖**：仅图像分类。`model_data.py:25` 对非 resnet18 直接 `raise ValueError`；
   `metrics.py` 只有 `classification_metrics`
2. **基线**：仅 `fedavg_full_local`。缺 SplitFed V1/V2/V3 与 FedProx
3. **数据**：五台机器 `data/` 只有 CIFAR-10（341M）

## 硬约束（实测）

| 主机 | 角色 | 算力 / 内存 | 备注 |
|---|---|---|---|
| 205 | 服务端 | 2× RTX A6000 48GB，torch 2.11+cu130，Py3.11 | `.venv` |
| 136 | 客户端 | Windows，CPU | 代码基准；shell 非 POSIX |
| 140 | 客户端 | Orin NX，**7.4GB 统一内存 / 可用 4.2GB / 磁盘 11GB**，6 核 | **短板** |
| 118 | 客户端 | Orin NX，15.3GB / 可用 10.8GB / 磁盘 17GB，8 核 | |
| 238 | 客户端 | Orin NX，15.3GB / 可用 10.9GB / 磁盘 17GB，8 核 | |

三台 Jetson：aarch64、JetPack R36.4.4、torch 2.5.0a0+nv24.08、venv 名为
`.venv-real-device`（非 `.venv`）、无 git、为部署副本。

`.140` 的 7.4GB 统一内存与 11GB 磁盘是全局上限 —— 决定了检测/分割必须用轻量数据集。

## 阶段 0：解除 205 阻塞（先做，其余全部依赖）

205 现在跑不了 server：`splitfleet/server/app.py:13` 与
`client/grpc/connection.py:19` 都 `from flwr.common.address import parse_address`，
而两台机器的 flwr 都是 1.29.0，该模块已不存在（实测 `ModuleNotFoundError`）。
`.136` 已修：新增 `splitfleet/common/address.py` 自实现 `parse_address`。

按你的选择 `.136 → 205`：

1. 从 `.136` 取回 11 个已修改文件 + 未跟踪的 `splitfleet/common/address.py`、
   `experiments/physical_cosplit_ucb/`、`experiments/real_device_splitfed.py`
2. 校验 205 的 `import splitfleet.server.app` 与 `client.grpc.connection` 通过
3. 在 205 上提交，使其成为唯一基准；此后统一由 205 下发到边端
4. 边端同步改用「排除 `.venv-real-device` / `data/` / `*.tar.gz`」的 rsync，
   并按边端 venv 名调用 `./.venv-real-device/bin/python`

205 工作树当前干净（`efd9364`），此步可 `git checkout .` 完整回退。

## 阶段 1：学术方案设计（协议定位由它决定）

你选择「先用学术 skill 出方案再定协议形态」，因此本阶段的产出是后续冻结形式的输入，
我不预先敲定 v3 还是 A11。

调用 `academic-paper` 的 plan 模式（非 deep-research，避免不必要的检索开销），产出：

- 研究问题与假设族（主 / 次假设分离）
- 「统一性」的可测化定义 —— 这是核心论点，必须给出可证伪的操作化指标，
  而非仅陈述工程覆盖面
- 四类任务的对照设计与统计方案（配对 seed 级效应、非劣性边界）
- 新增任务与新增基线相对既有 v2 冻结协议的方法学定位建议

既有 v2 协议已冻结、预注册，A1–A10 均在结果产出前冻结并自评 post-hoc 属性。
新增三类任务 + 四个基线属于新的假设族，我倾向独立冻结 v3 而非挂 A11 修正案，
但按你的决定，以 skill 输出为准。

## 阶段 2：任务族抽象（框架层）

在 `splitfleet/tasks/` 新增任务族注册表，把「模型 + 数据 + 损失 + 指标 + 切点候选」
收敛为一个协议对象，替代 `model_data.py` 里硬编码 resnet18 的分支：

```
splitfleet/tasks/
  base.py          TaskSpec: build_model / datasets / loss_fn / metrics / candidate_cuts
  registry.py      按名注册与解析
  image_cls.py     CIFAR-10 + resnet18 / wide_resnet50_2（迁移既有实现）
  text_cls.py      AG News + 轻量 transformer 或 TextCNN
  detection.py     VOC2007 + 精简 SSD / FCOS 变体
  segmentation.py  Oxford-IIIT Pet + UNet / LR-ASPP
```

指标层扩展 `metrics.py`：分类 accuracy / macro-F1，检测 mAP@0.5，分割 mIoU。

**检测与分割的已知技术风险**（需在阶段 2 先做 spike 验证，不确定就先降级）：

- torchvision 检测模型签名为 `model(images, targets)`，输入为变长 list、
  输出为 dict；README 明确「SplitFleet adapter 不支持 keyword-input tracing」。
  缓解：固定分辨率（如 320×320）批张量入口的 wrapper，在 wrapper 内部构造 list，
  使 tracing 面对的仍是单一位置张量；dict 输出只在服务端算损失，不跨边界
- FPN 多尺度特征意味着边界不止一个张量。`PlacementConstraint.max_frontier_size`
  默认 1，planner 已把它传给 `max_boundary_count`（`planner.py:231`），
  多张量边界机制上支持，但需实测确认，并相应放宽 `max_payload_bytes`
- 分割模型（UNet / LR-ASPP）输入输出较规整，风险低于检测

先用一个最小 spike 脚本在 205 上验证「检测模型能否被 TorchLens 追踪并在 layer 边界切开」。
若 spike 失败，退到自定义骨干的检测头实现，而不是放弃检测任务。

## 阶段 3：基线方法实现

按你勾选的四项，在 `run.py` 的方法分派与 `ProfileGuidedPlacementPolicy.METHODS`
中扩展。既有 `TaggedFedAvgClient`（纯 FL 路径）与 `TaggedPhysicalClient`
（分割路径）两条客户端已就位，新基线沿用同一记录与校验管线：

- **SplitFed V1**：客户端并行训练 prefix，prefix 与 suffix 各自 FedAvg。
  SFL 主对照，最接近本框架而不含资源自适应
- **SplitFed V2**：服务端不聚合，按客户端顺序更新 suffix
- **SplitFed V3**：仅聚合部分客户端
- **FedProx**：在 FedAvg 客户端目标上加近端项，处理 non-IID 漂移
- **FedAvg**：沿用既有 `fedavg_full_local`

同一 seed 内所有非 oracle 方法必须共享初始模型哈希、分区哈希、客户端顺序、
batch 预算、优化器与评估集 —— 沿用 `run.py` 既有的身份校验，不新造一套。

## 阶段 4：数据准备

按你的选择用轻量数据集，全四任务、四台客户端全参与：

| 任务 | 数据集 | 体量 | 落位 |
|---|---|---|---|
| 图像分类 | CIFAR-10 | 341M | 已在五台机器上 |
| 文本分类 | AG News | ~30M | 需下发 |
| 检测 | VOC2007 trainval+test | ~900M | 需下发，`.140` 可容纳 |
| 分割 | Oxford-IIIT Pet | ~800M | 需下发 |

合计约 2GB，`.140` 剩余 11GB 可容纳。分区沿用既有 Dirichlet(α=0.5) 与
`partition_manifest` 哈希机制，检测/分割按图像级划分。

数据下发脚本需对每个数据集记录内容哈希，确保五台机器数据同一。

## 阶段 5：真机实验

`orchestrate.py` 已实现 SSH 拉起边端 + PowerShell 拉起 Windows 客户端的完整流程，
扩展其参数以接受 `--task`，其余复用。

执行顺序遵循既有 README 的证据分层与阶段约束：

1. 五台主机环境/仓库/数据/时钟/GPU/链路记录
2. 每客户端每候选切点 profile（`profile_` 前缀，不得混入确认性结果）
3. 单 seed pilot 验证契约与记录（`pilot_` 前缀）
4. 冻结实现修正
5. 四类任务 × 全部方法 × seeds 1–5（每 run 校验后再进下一 seed）
6. 配对 seed 级效应、置信区间、准确率非劣性判定

`.140` 内存最紧，检测/分割须先在其上单独做一次内存压力验证，
必要时对该客户端降 batch —— 框架的动态 batch 窗口正是为此设计。

## 阶段 6：验证与产出

- 每阶段代码变更后在 205 跑 `pytest tests/`，边端跑导入与 smoke
- `validate_run.py` 对每个 run 做记录完整性校验
- `aggregate_results.py` / `plot_results.py` 汇总
- 结果交 `academic-paper-reviewer` 做方法学审查

## 需要你知道的两个判断

**「优越性」的可证伪表述**。四类任务全覆盖是工程贡献，但审稿人会问统一性本身
带来了什么可测收益。真正能支撑「优越」的是：在同等精度下降低边端峰值内存与
训练时延，或在异构设备上取得更优的公平性/完成率。既有 `metrics.py` 已有
`jain_fairness_index`，方向是对的。阶段 1 的学术设计需要把这一点定死。

**检测任务是本计划最大的技术不确定项**。它同时触及 keyword-input tracing
限制和多张量边界两个已知短板。阶段 2 的 spike 就是为了尽早暴露它，
而不是等到阶段 5 才发现跑不通。

