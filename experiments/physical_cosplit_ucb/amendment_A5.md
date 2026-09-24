# Protocol amendment A5: active-session isolation after reconnect

## Trigger

The retained, invalid run `confirmR2_best_global_fixed_seed1_20260810`
completed 100 rounds but recorded one `GrpcBridgeClosed` failure at round 49.
After the affected physical client reconnected with a new Flower session ID,
the server emitted five suffix records per round despite there being only four
physical devices. Strict validation rejected the run.

## Diagnosis

Persistent suffix models introduced by amendment A3 were intentionally cached
across rounds, but result collection iterated over every cached model rather
than only the models configured for the current round. The profile-guided
placement policy also retained the obsolete Flower session ID after learning
the replacement session for the same logical device.

## Prospective repair

Before any further confirmatory run:

1. Track the suffix model IDs configured for the current round and collect
   results only from that active set. Cached inactive models remain available
   for reuse but cannot enter aggregation or measurement.
2. When fit metrics associate a new Flower session ID with an already known
   logical device, evict the prior session mapping from the placement policy.
3. Add regression tests for both conditions and synchronize the resulting
   source manifest to all five physical hosts.

No dataset, partition, model, optimizer, pacing window, candidate split,
hypothesis, estimand, or acceptance threshold is changed. Failed rounds are
not retried and any run containing a recorded failure remains invalid.
