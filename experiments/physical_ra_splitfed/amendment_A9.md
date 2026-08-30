# Protocol amendment A9: prefix-only synchronization and native persistent execution

- Date: 2026-08-12
- Trigger: inspection of the completed A8 timing records showed that the
  SplitFed implementation transferred the complete model in both directions
  and replayed TorchLens prefix graphs in the measured path.
- Classification: post hoc systems-engineering experiment, reported separately
  from the A8 algorithm comparison.

The optimized path preserves the A8 topology, selected `wide_resnet50_2`
architecture, CIFAR-10 partition, optimizer, batch budget, evaluation, resource
phases, and candidate cuts. It changes only the implementation of split
execution and parameter ownership:

1. A client receives and returns only the named state entries in its selected
   prefix. The experiment uses absolute prefix parameters (`prefix_parameters`),
   not a compressed delta.
2. The complementary suffix state never crosses the client link. One suffix
   replica per Flower client remains resident on fy205 across rounds and is
   refreshed from the server-side aggregated suffix state.
3. Prefix and suffix execute explicit torchvision ResNet modules. TorchLens
   discovery, graph capture, and graph replay are absent from the timed path.
4. Each client constructs and prewarms all three persistent prefix callables
   before joining the Flower server. `torch_compile` is an optional fail-closed
   executor; `persistent_eager` is the portable primary executor.
5. Each per-client prefix and suffix update is reassembled into one complete
   logical model before sample-weighted SplitFed aggregation. Central
   evaluation therefore continues to consume a complete global model.

Before formal optimized training, a fresh nine-round native profile must run
the frozen schedule `stem,layer2,layer4` three times. Placement selection for
optimized methods may use only that native profile, not the A8 TorchLens
profile. Results measure the effect of this implementation change and cannot
retroactively alter the A8 conclusions.

## Frozen native profile selection

The nine-round physical profile
`profileA9_native_prefix_candidates_seed1_20260812` completed with 36 client
records, 36 persistent-suffix records, and zero failures. Its validation report
is valid and has SHA-256
`f08299e77137f8b2e9856cf7669493d774249a18f9366b2281a0208f27d1ae1a`.
Excluding the first occurrence of each cut, steady round-time means were
1.968144 s for `stem`, 5.667616 s for `layer2`, and 73.378649 s for `layer4`.
The frozen `best_global_fixed` cut is therefore `stem`. Per-client steady fit
time freezes `stem` for `win136` and `orin118`, and `layer2` for `orin140` and
`orin238`. These choices were recorded before any formal A9 training run.

## Native scheduler timing semantics

After the fixed-cut A9 run and before any resource-adaptive A9 run, a scheduler
audit found that the legacy profile reader treated client telemetry that was
already accumulated per round as if it were per batch. The native-only reader
now uses those counters once and adds the observed coordination residual
(physical round time minus the slowest client fit) to represent Flower prefix
serialization, transfer, reassembly, and aggregation. Legacy TorchLens profile
interpretation is unchanged. The derived scheduler report
`validation_report_native_cost_v2.json` is valid and has SHA-256
`94b4e9d25b64beea57448f34207770714a6e061f6d7bd2e05d56e0970fb23232`.

## Optional compiled-prefix feasibility

After the two formal persistent-executor runs, the optional
`torch_compile`/`aot_eager` path was attempted on all four physical clients as
`pilotA9_torch_compile_prefix_feasibility_seed1_20260812`. The attempt failed
before `orin140` connected: its NVIDIA PyTorch compile worker could not import
`triton_key` from the user-level Triton package. The run completed zero rounds,
used no fallback, was stopped after the fail-closed client exit, and is not
valid experimental evidence. The persistent native executor remains the only
four-device-validated A9 execution mode.
