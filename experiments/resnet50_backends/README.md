# ResNet50 physical backend campaign

The retained physical campaign evaluates TensorFlow and Paddle on three Orin devices. Each Orin runs one CPU client and one GPU client, for six logical clients, while the server uses GPU 0 of the NVIDIA RTX A6000 with concurrency one.

The formal protocol uses 45,000 training examples, 5,000 validation examples, 10,000 test examples, 50 rounds, seed 233, Greedy per-client placement, and batch size 16 (469 batches per client, including a tail batch of 12). Each round evaluates the validation set and writes the latest and best checkpoints. Training admission may be skipped only when the campaign receipt records that choice explicitly.

The current retained results are selected through `results/resnet50_backends_current.json`. Historical interrupted attempts and backend implementations are excluded from the current queue and are not pooled into progress.

Prepare a retained campaign with:

```bash
.venv/bin/python -m experiments.resnet50_backends.campaign prepare \
  --output results/resnet50_backends_orin6_<backend>_50rounds \
  --batch-size 16 --rounds 50 --methods splitfleet \
  --placement greedy
```

Run it with:

```bash
.venv/bin/python -u -m experiments.resnet50_backends.campaign execute \
  --root results/resnet50_backends_orin6_<backend>_50rounds \
  --backends tensorflow paddle
```
