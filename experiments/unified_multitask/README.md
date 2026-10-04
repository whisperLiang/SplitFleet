# 四任务数据与模型适配

此目录是物理实验的数据、模型支持库：`data.py`提供数据集适配、任务指标、采样及内容哈希；`edge_models.py`提供完整预训练 ResNet-50、BERT-base、RF-DETR Nano、DeepLabV3-ResNet50；`fetch_ag_news.py`准备AG News源文件。批次处理、训练与评估共用 `experiments/common/workload_training.py`。

训练使用CIFAR-10、AG News CSV、VOC2007和Oxford-IIIT Pet。BERT配置与tokenizer固定到同一检查点版本，抽样前去除规范化后与评估集完全相同的训练文章。VOC保留difficult标记；子集mAP采用VOC07十一点AP，未出现非difficult标注的类别不计入均值。Nano采用384像素的[0,1]输入、50queries和原生匹配损失，六方案配置一致。

本地四任务协议 `plans/four_task_study.md` 和[物理训练入口](../README.md)描述正式实验；各作业的模型、数据、分区、初始状态和最终检查点身份均可核验。协议保存在本地，不随代码仓库上传。
