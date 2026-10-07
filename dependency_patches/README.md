# Local TorchLens 2.34.1 patch

`torchlens_tinygrad_parameter_views.patch` binds a captured named tinygrad
parameter view to the current live Tensor after state loading. A transposed or
sliced parameter cannot safely replace the flattened storage of its ancestor
buffer. The binding requires exact parameter object identity and preserves the
training autograd path.

The bundled wheel was rebuilt from a clean copy of the sibling TorchLens source
at `2a073a86300b1bec736107c565b003f7d86920c2`. Of its 603 Python files, only
`torchlens/split/adapters/tinygrad.py` differs from the previous local wheel.
This is a local patched artifact; it retains the upstream version number.

- Previous wheel SHA-256: `a0b154a3dbb43638f3b66457b95fd633676b01d5baaad2ffded81536bd05451a`
- Patched wheel SHA-256: `325c28ea4015b2a7bab96cde8b2876e7cef8c1590636d4061408593f13bf2a7c`

The SplitFleet regression checks reshape, transpose and slice values before and
after state loading, including nonuniform values and native SGD update parity.
The execution receipts and dependency content audits live under
`results/www2027_final/physical_execution_20261006_025318/`.
