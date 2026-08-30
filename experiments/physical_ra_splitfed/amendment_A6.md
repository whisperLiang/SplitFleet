# Protocol amendment A6: recovery-window timing

## Trigger

The retained, invalid run
`confirm_static_heterogeneous_seed1_20260810` recorded a
`GrpcBridgeClosed` failure in round 49. The missing physical client was
`orin238`. Its client log showed that the first reconnect attempt raised
`RuntimeError: Unable to connect ... after 1 attempts` approximately 906
seconds after the process started, despite the configured 60-attempt,
900-second recovery bounds.

## Diagnosis

The A4 reconnect implementation measured `max_wait_time` from initial client
startup, including all healthy training time. Consequently, once a run had
lasted at least 900 seconds, its first transient disconnect immediately
exhausted the recovery window. Attempt counting likewise described lifetime
connection attempts rather than consecutive recovery failures.

## Prospective repair

Before any further final-matrix run:

1. Start the recovery timer at the first consecutive stream/connection
   failure, not at client startup.
2. Reset both the failure counter and recovery timer after any successfully
   received server message.
3. Preserve the configured 60-consecutive-failure and 900-second limits.
4. Add regression tests covering a first disconnect after more than 900
   seconds of healthy runtime and reset after successful recovery.
5. Synchronize the resulting source manifest to all five physical hosts.

No dataset, partition, model, optimizer, pacing window, candidate split,
hypothesis, estimand, or acceptance threshold is changed. Failed rounds are
not retried and any run containing a recorded failure remains invalid.

Because this amendment changes the source manifest, the previously valid A5
run `confirmR3_best_global_fixed_seed1_20260810` remains valid as a standalone
infrastructure run but is not pooled into the A6 final paired method matrix.
