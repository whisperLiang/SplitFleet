# Protocol amendment A4: long-lived Wi-Fi transport reliability

- Date: 2026-08-10
- Timing: after the second incomplete seed-1 baseline attempt, before any
  complete confirmatory run existed
- Trigger: `orin140` disconnected during round 90; the Python process remained
  alive but did not restore its Flower stream, leaving only three clients

The attempt in `confirmR1_best_global_fixed_seed1_20260810` is retained as an
invalid failed run. It is not used in confirmatory estimands. The first failure
stream is also retained even though its string representation was empty.

All three Jetson clients use the 2.4-GHz `wlP1p1s0` interface with Wi-Fi power
saving enabled. At diagnosis, `orin140` reported 35,651 cumulative received
drops, compared with 1,345 and 1,644 on `orin118` and `orin238`. Short ping tests
had zero packet loss. An attempt to disable power saving was rejected because
sudo requires a password; no host network setting was changed.

The transport is amended to use a 30-second gRPC keepalive, a 10-second
keepalive timeout, and bounded reconnect backoff. Unexpected stream exhaustion
now enters the existing reconnect path instead of being treated as normal
completion. Failure records now store exception type and `repr` as well as
`str`.

This repair does not retry a failed round, relax validation, change client
participation, or alter training. Any recorded failure still invalidates the
run. No workload, data, model, optimizer, pacing, placement, hypothesis,
threshold, or statistical analysis rule changed.
