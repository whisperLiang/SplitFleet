# FedCompass port attribution

The experimental FedCompass controller is a port of the scheduling and delta
aggregation algorithms in Li et al., *FedCompass*, ICLR 2024, checked against
https://github.com/APPFL/FedCompass at commit
`af96db2204795b0b0addfdec219b2a541f4b1ab3`, especially
`src/appfl/scheduler/compass_scheduler.py` and
`src/appfl/aggregator/fedcompass_aggregator.py`.

The local port replaces timer threads with a Flower event loop, uses the
paper's decaying polynomial staleness, sample-size client weights, typed
sample-weighted buffers and explicit model-version/time-budget receipts.
It does not bundle the upstream APPFL runtime. The upstream license follows.

MIT License

Copyright (c) 2024 Argonne National Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
