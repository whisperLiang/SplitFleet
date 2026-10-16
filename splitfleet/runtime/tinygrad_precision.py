"""Configure native FP32 compiler rounding before constructing model tensors."""

from importlib.metadata import version


def configure_device(device):
    """Keep multiply and add separately rounded in native and split execution.

    Native replay materializes multiplication nodes before their reductions.
    Contracting those operations only in the full model changes deep BN
    inputs. Both methods use the same compiler options and a distinct binary
    cache namespace. No compiler body, SDK class or registry is modified.
    """
    if version('tinygrad') != '0.13.0':
        raise RuntimeError('Tinygrad FP32 compiler policy requires tinygrad 0.13.0')
    from tinygrad import Device
    from tinygrad.runtime.support.compiler_cpu import ClangJITCompiler
    from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

    compiler = Device[device].compiler
    if isinstance(compiler, ClangJITCompiler):
        arguments, options = compiler.args, ('-fno-associative-math', '-ffp-contract=off')
        marker = '_splitfleet_ordered_cpu_fp32_v3'
    elif isinstance(compiler, NVRTCCompiler):
        arguments, options = compiler.compile_options, ('--fmad=false',)
        marker = '_splitfleet_separate_fp32_operations'
    else:
        raise RuntimeError(f'Unverified Tinygrad FP32 compiler: {type(compiler).__name__}')
    if getattr(compiler, marker, False):
        if any(option not in arguments for option in options):
            raise RuntimeError('Tinygrad FP32 compiler policy changed after initialization')
        return
    arguments.extend(options)
    if compiler.cachekey is not None:
        compiler.cachekey += marker
    setattr(compiler, marker, True)
