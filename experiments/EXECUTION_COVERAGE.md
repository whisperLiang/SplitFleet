# Full-model execution coverage

`analysis/execution_coverage.py` enumerates each model's complete before/after catalog, checks native training against the admitted split paths, and keeps rejected, refused and numerically failed sites separate. It uses the frozen executor through existing task adapters; the added architectures have no handwritten partitions. The current run covers seven full PyTorch architectures, not five-backend YOLO support.

Every admitted site receives a paired first Adam step. The widest eligible frontier in each nonempty graph third is selected before outcomes for ten steps; RF-DETR has two eligible thirds. Each cut restores the same initialization, with independent client/server storage, continuous Adam state within the trajectory, matched RNG and retained dropout. Native/native repeat controls pass for all seven models. Inputs are the first ten distinct source examples from retained client-0 training bundles, batch one. Server bundles contain only trace batches and are unsuitable here.

The checks cover outputs, loss, gradients, parameter increments, persistent/nonpersistent buffers, Adam state and exact activation/gradient wire values. Floating checks use rtol=2e-4, atol=2e-6 on each quantity. Wire snapshots are compared before optimizers can mutate aliased parameter views. Structures, integers and transmitted values are exact. The same-host RPC probe uses the production gRPC protobuf service and tail model; it does not measure remote Orin equivalence. A separate B2→B1 sequence reuses one capture for MobileNetV3/DistilBERT, starting from the common post-control model state with fresh Adam.

| Model | P/A/E | Ten-step | RPC | B2→B1 |
|:--|:--|:--|:--|:--|
| resnet50 | 242/242/350 | 3/3 | passed | not_tested |
| bert | 609/742/760 | 0/3 | passed | not_tested |
| rfdetr | 1016/1016/2126 | 2/2 | passed | not_tested |
| deeplab | 424/424/620 | 3/3 | passed | not_tested |
| mobilenet_v3 | 280/280/466 | 3/3 | passed | passed |
| distilbert | 348/408/410 | 1/3 | passed | failed |
| yolo26 | 684/684/1152 | 3/3 | passed | not_tested |

P/A/E means sites passing all planned checks / admitted / enumerated. Before/after sites can represent equivalent cuts; these counts are not independent observations. There are 5,884 sites, 3,796 admitted, 3,604 first-step passes and 3,603 passes of all planned checks. One BERT pressure site passes the first step and fails the second. The 193 failed sites are Adam parameter-increment mismatches; loss/gradient checks pass at their failed steps. Fifteen of twenty ten-step trajectories complete. All seven RPC first steps pass. DistilBERT's B2→B1 sequence already differs at B2, so this result does not isolate a tail-shape cause.

Four models restore recorded pretrained weights. MobileNetV3-Large, six-layer DistilBERT and YOLO26n use seeded random initialization at full standard size; this is a reuse check, not a pretrained-quality or blind holdout claim. YOLO26 uses its native dual-head training graph and matching loss. Torch 2.11.0, Torchvision 0.26.0, Transformers 5.8.1, RF-DETR 1.6.5.post2 and Ultralytics 8.4.51 are recorded in the frozen runtime manifest.

Reproduce with the repository's `integration` extra and the recorded dependency versions. Use a fresh output directory; the runner never overwrites previous attempts:

```bash
.venv/bin/python -u -m experiments.analysis.execution_coverage \
  --suite-bundles paper/evidence/rq1_execution/bundle_catalog.json \
  --output results/execution_coverage_repeat --steps 10
```

The suite freezes source/config/dependency identities and runs two GPU lanes. Results and failure traces are in `results/rq1_execution_20261009_final/`; `paper/evidence/rq1_execution/summary.json` is a compact source-bound summary. Figure generation reads these results without training.

The workspace validator now restores all named buffers in place after capture
and before each independent cut/RPC trajectory, including non-persistent buffers
omitted from `state_dict`. Tail-batch capture uses the same restoration; buffer
and Adam state remain continuous within each trajectory. This fixes false
failures for models whose forward modifies non-persistent buffers. The retained
seven-model results continue to identify their original executed validator
snapshot. Export checks verify that complete snapshot and the unchanged
framework executor, and report both validator hashes when the workspace version
has advanced.

Storage pruning retains exactly the four client-0 inputs named in the bundle
catalog, together with the separate actual PPO actor. Other historical input
copies and completed training weights have been removed; raw metrics, failure
records, frozen sources and figure data remain unchanged. This RQ1 command still
has its original inputs. Historical end-to-end training needs newly prepared
bundles; checkpoint checks are recorded before deletion.

For newly prepared physical matrices, inputs are shared v3 descriptors and the
matrix removes its own inputs when all worker processes stop. Add
`--keep-input-bundles` if they will also feed an execution-coverage run. Existing
v2 inputs remain readable; this suite's four retained original inputs and frozen
execution records are unchanged. Final task weights in new server runs require
`--save-model`; numerical results and failures are always recorded.
