# Protocol amendment A1 — 2026-08-10

Timing: after candidate profiling and before any confirmatory training run.

The profile exposed a large first-use runtime-construction cost that differs by
device and cut.  To prevent a method's first choice from determining its cache
state, every confirmatory method now executes the same stem, layer2, and layer4
sequence in rounds 1--3.  These rounds use the same optimizer and therefore
leave every sibling method at the same post-round-3 model state.  System
estimands use rounds 4--100; accuracy curves retain all rounds.

The resource-adaptive claim is tested with a predeclared dynamic condition:
natural LAN in rounds 4--25, recorded application-level pacing of win136 to
10 Mbps uplink/50 Mbps downlink in rounds 26--60, and recovery in rounds
61--100.  Model execution and RPC remain physical.  The constrained phase is
described as controlled application pacing, not a changed physical NIC.

This amendment was made before any confirmatory outcome.  The 10/50 Mbps values
come from the repository's pre-existing `poor` network profile rather than
from optimization against the completed physical candidate profile.
