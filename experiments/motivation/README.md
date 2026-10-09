# DeepLabV3-ResNet50 动机实验

当前动机图使用三台Orin上的 `cpu1–3`、`gpu1–3`：面板(a)呈现本地训练成本、掉队者和推导等待；面板(b)比较Local与25%、50%、75%静态切点。`analyze_deeplab.py`只读分析P04已有记录，不启动训练。

唯一当前分析位于 [paper/evidence/deeplab_motivation/analysis](../../paper/evidence/deeplab_motivation/analysis)。保留八项完整作业、两个原始种子、全部十轮与480条客户端轮次记录，逐文件校验原始hash。Oxford-IIIT Pet、320×320、FP32、batch=2、Adam 1e-4及固定A6000后缀沿用P04。

每轮成本为客户端就绪等待结束后的fit秒数除以实际batch数；SFL包含前缀、后缀及激活/梯度RPC，不含联邦状态传输、聚合和评估。先在每种子每轮计算 `tau=max(t_i)`、`w_i=tau-t_i`，再平均轮次和种子。灰色等待是共同起点、等工作量比较的推导差值，原记录不提供实测同步等待。分区26–57样本、末批大小及同机CPU/GPU并发仍是限制。

Local/25%/50%/75%的平均最大成本为8.22/1.91/3.05/13.13 s每批，静态切点的配对变化为−76.7%/−62.9%/+59.8%；平均隐含等待为3.96/0.27/0.56/2.21 s每批。75%的等待高于25%/50%，仍低于Local。所有六种配置的已测最小成本均在25%；不推断设备特有最优切点或自适应优势。

从仓库根目录生成当前图和论文：

```bash
.venv/bin/python paper/figures/generate_deeplab_motivation.py \
  --summary paper/evidence/deeplab_motivation/analysis/summary.json
.venv/bin/python paper/build.py
```

如需从原始回执重新分析，输出必须是尚不存在的新目录：

```bash
.venv/bin/python -m experiments.motivation.analyze_deeplab \
  --output /tmp/deeplab_reanalysis
```

当前来源由 `paper/figures/motivation_current.json`及摘要SHA-256绑定，导出hash在 `motivation_figure_manifest.json`。重分析不会自动替换当前来源。
