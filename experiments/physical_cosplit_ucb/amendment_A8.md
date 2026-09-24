# Protocol amendment A8: post hoc large-model stress experiment

- Date: 2026-08-10
- Trigger: the valid seed-1 ResNet-18 FedAvg run was faster than all valid
  SplitFed variants, motivating the user-requested test of whether the original
  model was too small to expose a split-compute advantage
- Timing: after ResNet-18 outcomes were inspected and before any large-model
  candidate profile or training outcome existed
- Classification: separate exploratory stress experiment; not a modification
  of the original ResNet-18 confirmatory matrix

The large-model experiment preserves CIFAR-10, GroupNorm, the four physical
clients, fy205 server, seed-specific Dirichlet partitions, batch size 16, four
batches per client per round, SGD without momentum, centralized evaluation,
and the fail-closed validity rule.

Before profiling or formal training, the implementation may run feasibility
checks that do not produce model updates for inference. Candidate architectures
are ordered by parameter count: `wide_resnet50_2`, `resnet101`, then `resnet50`.
The first (largest) candidate that completes full-local and `stem`, `layer2`,
and `layer4` split-training checks on every physical client without OOM or an
unsupported graph contract is selected. Selection cannot use comparative
round-time or accuracy outcomes. Failed candidates and their diagnostics are
retained.

After selection, the architecture and source manifest are frozen. A fresh
nine-round physical candidate profile selects the large-model
`best_global_fixed` and `static_heterogeneous` placements; the ResNet-18 profile
must not be reused. Only after that profile passes validation may large-model
method runs begin.

Because the extension was motivated by observed ResNet-18 results, all
large-model comparisons are reported as exploratory. They cannot rescue the
original primary hypotheses or be presented as preregistered evidence.

## Frozen architecture selection

At 23:15 CST on 2026-08-10, before the formal nine-round profile, the selection
rule chose `wide_resnet50_2`. The full-local feasibility run
`pilotA8_wide_resnet50_2_fedavg_full_local_seed1_20260810` completed one update
on all four clients. The split feasibility run
`pilotA8_wide_resnet50_2_split_feasibility_seed1_20260810` completed `stem`,
`layer2`, and `layer4` once on every client, producing 12 client records, 12
server records, and zero failure records. Its validation report is valid and
has SHA-256
`ca44fadbca05048b91991ca15da685f3905e19574ef505043ba876ee0ea57ba3`.

The 8 GiB `orin140` client used swap during split feasibility, which is retained
as a resource-pressure limitation. It did not OOM and the graph contract was
supported, so neither timing nor accuracy was consulted and the frozen rule
does not permit falling back to a smaller candidate.

## Frozen large-model profile selection

The fresh nine-round profile
`profileA8_wide_resnet50_2_candidates_seed1_20260810` completed with 36 client
records, 36 suffix-server records, and zero failures. The validation report is
valid and has SHA-256
`05f0b57b925bac61a26a866e920c31431803491606eb61e24464ad7cbaa24032`.
Following the pre-specified rule, the first occurrence of each cut was excluded
from selection. The steady round-time means were 71.129613 s for `stem`,
71.853825 s for `layer2`, and 68.273908 s for `layer4`. Therefore
`best_global_fixed` is frozen to `layer4`. Per-client steady fit time freezes
`static_heterogeneous` to `stem` on `win136` and `layer4` on `orin140`,
`orin118`, and `orin238`. These selections were frozen before any formal
large-model comparison run.
