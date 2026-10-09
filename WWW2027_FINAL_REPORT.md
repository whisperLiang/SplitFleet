# SplitFleet 当前完成记录

2026年10月9日按当前中英文正文及附录，删除5,510个未使用文件，共137,055,226字节（130.71 MiB），包括旧试跑、独立通信试验、额外层级oracle、未执行计划、中间分析及重复代码快照。4,409个删除文件为Python源码，主要来自闲置快照及重复依赖；当前实验包另删除未被使用的 `online_calibration_admission.py`。正式五种子CNN仍需的旧部署配置和权重解析器、在线队列依赖的准入/输入/文献资料均保留。当前 `results/`约1.67 GiB。[结果索引](results/README.md)和[清理收据](paper/evidence/unused_experiment_cleanup_receipt.json)记录具体范围。

15,321个保留的结果、源码和论文文件与清理前逐字节一致，既有代码修改也保持原样。历史训练重分析的JSON及六份CSV一致，在线重分析的全部数值及九份其他输出一致，动机重分析的JSON及六份CSV一致；在线摘要仅分析脚本版本字段与旧记录不同，该版本差异在清理前已存在，原摘要未改动。核验五后端25项训练及305项原始回执hash、保留符号链接、当前证据索引和中英文PDF/Word导出均通过。完整CPU回归782通过、19跳过，退出码0；本次未重新运行真实设备训练。

已实现完整模型切点验证，并更新中英文论文、LaTeX、PDF和Word。图3采用约180×61 mm的横向布局，将真实YOLO26训练依赖、全部边界状态及验证分母放在同一图中。图4/5为在线/历史训练，附录图6–8延续原证据。

RQ1使用冻结PyTorch执行器，接入ResNet-50、BERT-base、RF-DETR Nano、DeepLabV3、MobileNetV3-Large、六层DistilBERT及YOLO26n；三个新增架构没有手写分区或执行器改动。全部5,884个算子前/后边界中3,796个准入，3,604个通过第一步，3,603个完成全部计划检查。区别来自一个BERT压力边界在第二步失败。

BERT/DistilBERT分别保留133/60个Adam参数增量检查失败，失败步的输出、损失和梯度检查通过。20条预选最大前沿轨迹中15条完成十步，失败保留于计划分母。七项同主机生产gRPC单步检查均通过。MobileNetV3的batch二→一复用通过，DistilBERT在batch二已偏离，无法将失败单独归于尾批形状变化。新证据仅覆盖一次初始化/输入轨迹下的已测PyTorch路径，不建立任意模型、后端或设备的训练等价性。

正式完整结果在 `results/rq1_execution_20261009_final/`，包含冻结源码/依赖、输入hash、所有切点检查、十步失败轨迹及生产RPC记录。[实验说明](experiments/EXECUTION_COVERAGE.md)、[简要结果](paper/evidence/rq1_execution/summary.json)及 `paper/figures/evaluation_execution_cuts_source.csv`保留精确分母与失败位置。边界站点不是独立重复，等价before/after站点可能对应同一分割。

上一轮清理删除96,423个过期计划、重复论文备份、旧图/生成器及闲置实验文件，按文件大小合计73.87 GiB。[产物清理收据](paper/evidence/cleanup_receipt.json)记录当时的完整性检查。动机分析重新绑定到字节完全相同的证据缓存，数值及PNG完全不变。

本轮按用户要求删除119个闲置输入bundle、79个已完成训练的完整权重及7,962个字节码文件。删除前核验全部79个检查点的有限值/模型哈希，并记录每个删除文件的SHA256；其中48个历史检查点复用文件hash完全一致的既有核验。释放35.75 GiB，`results/`从37.59 GiB降至1.84 GiB。全部20,381个保留的结果/源码文件及所有图表数据、图片保持字节完全相同。仅保留四个RQ1 client-0输入bundle和实际PPO策略文件，合计约953 MiB；原始JSON/CSV、失败、运行配置、冻结源码及依赖快照不变。[存储清理收据](paper/evidence/storage_cleanup_receipt.json)记录精确路径、删除前核验和当前状态。

统计分析和图表仍可重建；七模型RQ1仍有原始输入。历史训练重跑需要重新准备bundle，删除后的最终权重不能直接加载复评。`refresh_evidence.py --use-checkpoint-receipts`明确使用删除前核验，不声称重新检查已删张量。

完整CPU回归759通过/19跳过，退出码0；可选及CUDA依赖检查在此回归中跳过，七个完整模型另完成GPU实测。[测试收据](paper/evidence/rq1_execution/test_receipt.json)和原始日志保留。最初测试错误地假设各图至少两个压力切点，现改为从实际非空目录选取中间项；该修正不改变正式实验。

[中文PDF](paper/zh/splitfleet_zh.pdf)、[英文PDF](paper/en/splitfleet.pdf)、[中文Word](paper/docx/SplitFleet_ZH.docx)、[英文Word](paper/docx/SplitFleet_EN.docx)。导出检查核验图号、数值/来源、引用顺序、原生公式、Word图片和编译日志；当前核验收据为 `paper/evidence/rq1_execution/export_validation.json`。作者科学复核仍待完成，`submission_ready=false`。

本轮存储清理核验：48项历史作业的证据JSON和六份CSV逐字节一致；在线重分析的所有数值/原始来源hash及九份表格、报告一致，仅当前分析代码hash更新；动机重分析JSON/CSV一致。重新生成六幅评估PNG和九份源CSV，与清理前逐字节一致。新增存储核验测试2项通过；中英文PDF/Word导出检查通过，英文正文八页、总十二页，中文十七页。

后续存储默认已落实到代码：PyTorch多任务/放置/自适应及多后端服务器仅记录结果、失败和最终内存有限值/哈希核验，完整任务权重需`--save-model`。v3输入描述文件共享模型及数据，层级变体只改元数据；物理矩阵停止作业后默认清理自有输入，`--keep-input-bundles`保留以供调试、重跑或RQ1。独立准备的共享输入及PPO策略保留；新冻结子进程禁写字节码。115项相关测试通过，四个原有RQ1输入的哈希和模型/数据检查通过，各服务器与矩阵CLI保留选项核验通过。旧执行快照与原始实验记录未修改。[存储策略验证](paper/evidence/storage_policy_validation.json)记录当前源码hash与测试日志。
