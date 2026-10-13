# FedCompass replacement protocol

Implementation updated 10 October 2026. FedCompass replaces FedProx in the default physical FL comparison. The current study uses three Orins and the A6000, four tasks and five original paired seeds in two independent protocols: 100 fixed-ten-round jobs with five synchronous methods, and 160 common-budget jobs with eight methods including FedAsync, FedBuff and FedCompass. FedCompass is excluded from the fixed-round protocol because global asynchronous update counts do not match complete synchronous local-epoch rounds. See [ASYNC_FL_BASELINES.md](ASYNC_FL_BASELINES.md) for the admitted source, common settings and expanded asynchronous controls.

The preceding six-method, 120-job plan stopped after 94 successful attempts and two failures, leaving 90 complete paired endpoints and four unpaired successes. Its original BERT graph failure, isolated Transformers 5.8.1 dependency amendment, DeepLab connection failure and reboot-interrupted diagnostic pilot remain under `results/fedcompass_physical_20261010/`. These outcomes do not enter either new population. New raw results are under `results/fl_dual_protocol_20261010/`; both populations must pass complete raw audits before their numerical results enter the paper. Historical FedProx measurements retain their identities.

The common-budget method set is FedAvg, FedAsync, FedBuff, FedCompass, Fixed-SFL-25/50/75 and SplitFleet. The CoSplit-UCB/Independent-UCB/Greedy/L2S placement comparison has its own paused plan. FedCompass is a full-model, semi-asynchronous FL baseline; it does not offload a suffix to the split server.

## Source and mechanism

Use Li et al., *FedCompass*, ICLR 2024, [primary paper](https://arxiv.org/html/2309.14675v2), and the [authors' repository](https://github.com/APPFL/FedCompass) at commit `af96db2204795b0b0addfdec219b2a541f4b1ab3`. Source inspection is recorded in `paper/evidence/FEDCOMPASS_SOURCE_AUDIT_20261009.json`.

Retain computing-power estimates from observed response times, dynamic local step assignment, arrival-group formation, group deadlines, model-version tracking, late-update buffering and staleness-aware aggregation. Start clients from the same initialization through one initial readiness barrier. Subsequent training must progress by arrival groups, without the existing all-client barrier on every round.

`experiments/baselines/fedcompass.py` ports the scheduler and delta aggregator into a single-owner event loop. `experiments/common/fedcompass_server.py` runs concurrent, one-at-a-time-per-client Flower fit RPCs with the existing `FullClient` and frozen workload bundles. Both `physical_multitask` and `orchestrate_physical_multitask` accept `fedcompass`; `www2027_study` forwards its configuration and audits the new protocol. No APPFL dependency is required. The implementation uses the existing task loss, preprocessing, private data and full client model, including scalar and integer state tensors.

## Implemented source choices

| Choice | Primary-paper setting | Pinned example/source behavior | Required treatment |
|:--|:--|:--|:--|
| Latest arrival factor | Appendix E uses `lambda=1.2` | Example YAML sets `latest_time_factor=1.5` | Port default is 1.2, exposed as `--fedcompass-latest-time-factor`. |
| Polynomial staleness | Paper uses decay `(1 + staleness)^(-a)` | Aggregator returns `(1 + staleness)^a`; YAML supplies `a=0.5` | Port uses `(1 + delay)^(-a)` with `--fedcompass-staleness-exponent 0.5`, verified numerically. |
| Client weights | Formulation permits importance weights `p_i` | Pinned aggregator uses `1 / num_clients` | Port uses frozen partition size / total size to match FedAvg's sample-weighted objective. |
| Step limits | Classic-dataset configurations use `Q_min / Q_max = 0.2`; FLamby uses 20/100 | Example uses 20/100 | Port defaults to `--fedcompass-min-steps 20 --fedcompass-max-steps 100`; freeze workload-specific limits before primary seeds. |

The default mixing coefficient is `--fedcompass-alpha 0.9`, speed momentum is 0.9, and fit/property RPC timeout is `--fedcompass-rpc-timeout-sec 1800`. For each accepted client, the parameter update is `global -= alpha * p_i * (1 + delay)^(-a) * (dispatch_base - trained_state)`. Each dispatch carries the version of its actual base, even when several groups expire together. Parameter deltas are not divided by local steps. Nonparameter buffers use the sample-weighted average of the included local states; integer buffers retain their dtype and scalar shape. This buffer adapter and the sample-weighted objective are documented departures from the pinned code.

Optimizers restart per dispatch, matching the pinned trainer and the existing synchronous clients. Local batches retain partial tails and repeat the shuffled loader only until the assigned number of steps has completed. The current dispatch for a client replaces its older late-buffer entry when both would otherwise enter the same aggregation, as in upstream dictionary merging. Complete groups and expired groups release their arrived clients; missing clients remain in flight and later enter the general buffer.

The pinned scheduler's global-epoch counter is not a count of global model updates: single/buffered arrivals increment it, and group aggregation adds the number of aggregated updates. The port separately records dispatches, completed fits, optimizer steps, processed samples and actual model-version increments. Without `--time-budget-sec`, `--rounds` is a global-update cap for FedCompass and a synchronous-round cap for other methods, so equal values do not establish matched work.

Use plain FedCompass for the primary comparison. Momentum/normalization variants are separate methods if later evaluated. Record every source/configuration change and its validation; neither a locally modified adapter nor a different workload is a full reproduction of the original experiments.

## Matched physical comparison

Start with ResNet-50/CIFAR-10 on the three existing Orin workers and the A6000 coordinator/suffix server. Run all primary methods as a fresh matched cohort before extending to BERT, RF-DETR and DeepLab. Within a paired seed, use identical checkpoint, train/evaluation partitions, preprocessing, batch size, client devices, precision policy, loss and common local-optimizer settings. Record whether the optimizer restarts per dispatch for every method. Each method gets the same hardware availability and independent run; preserve actual server use rather than introducing suffix offloading into FedCompass.

Freeze source/dependency hashes, learning rates, step limits, evaluation intervals, quality targets, time budgets, seeds, network-change times and method order after separate validation/pilot runs and before primary execution. Give methods equal tuning resources and never use the test set for selection. Use at least five planned paired seeds; retain every failure and avoid replacement seeds. Historical and new cohorts must not be pooled because the source, aggregation protocol and training budgets differ.

Use wall-clock-triggered network changes for every method. A degradation at synchronous round 6 cannot be matched to an arbitrary FedCompass arrival or model-version count.

## Endpoints and timing

The primary endpoints are task quality versus coordinator wall-clock time, quality at a common time budget, and time to prespecified quality targets. A target that is never reached remains unreached; do not substitute a failed or incomplete run's last timestamp as its target time. For four-task extension, use each task's native quality metric.

Record an initial evaluation and use a common wall-clock evaluation cadence, with model-version/snapshot times and evaluation completion times. Count evaluation, calibration, switching, communication, aggregation and scheduler preparation consistently in the measured job scope. Data download and external deployment can remain outside that scope if excluded for every method. Do not trigger a full evaluation after every asynchronous arrival, which would impose a different evaluation cost.

Pass `--time-budget-sec` and `--evaluation-interval-sec` to the orchestrator for all methods. A time budget supersedes their round caps. The clock begins before server scheduler/calibration preparation, includes readiness and evaluations, and accepts only models committed before the deadline. The final evaluation reads the frozen endpoint after outstanding RPCs drain; its completion time is logged and excluded from budget-limited TTQ if it exceeds the deadline. Synchronous in-progress rounds restore both global prefix/full-model parameters and suffix state when they finish too late. Application communication includes all completed dispatched work, including drained work, and excludes transport overhead. A budget too short for a single valid update fails validation.

Results contain `fedcompass_config`, `dispatch_records`, `aggregation_records`, `group_events`, `global_updates`, `completed_dispatches`, `total_optimizer_steps`, `total_examples_processed`, speed estimates, model-version/snapshot and completion times, and training-stop/drain durations. Dispatches include assigned steps, group expected/deadline times and coordinator-observed client idle time. Fit receipts distinguish aggregated, superseded, unaggregated and discarded updates. Communication accounting covers every completed dispatch. The validator checks one initial barrier, population, hashes, exact steps/tail sample counts, staleness weights, update uniqueness and budget endpoints. Summaries use asynchronous dispatch/checkpoint accounting and omit fixed-work speed ratios; study analysis disables total-duration speed claims where work is not comparable.

Figure 5 is scheduled to show the new synchronous fixed-round and expanded common-budget populations in separate panels after both complete. A FedCompass result must come from actual execution and its own source receipts. The historical six-method evidence in Figure 7 retains its FedProx label and original population.

## Validation before primary execution

Confirm that assigned steps vary with measured client response times, group deadlines release arrived clients, late clients do not force an all-client barrier, and staleness weights follow the declared formula. Check grouped and single-client aggregation against their declared weighting objective and parameter/buffer treatment. Verify exact local optimizer-step accounting, client data isolation, comparable initialization, inclusive coordinator timing, failure retention and complete source receipts. A local correctness fixture provides implementation evidence only, not physical performance evidence.

Run the focused correctness checks with:

```bash
.venv/bin/python -m pytest -q --disable-warnings \
  tests/test_fedcompass.py tests/test_fedcompass_rpc.py \
  tests/test_time_budget_server.py tests/test_physical_multitask_summary.py \
  tests/test_communication_analysis.py tests/test_www2027_study.py
```

The RPC tests use three local CPU clients and the real Flower bidirectional gRPC transport, with tensor fixtures confined to tests. They cover a global-update cap, a timed FedCompass run, timed FedAvg, one-barrier execution, exact repeated local steps, scalar/integer serialization, drained traffic and explicit failure propagation. These checks do not supply manuscript performance values. Upstream attribution is preserved in `dependency_patches/FEDCOMPASS_NOTICE.md`.
