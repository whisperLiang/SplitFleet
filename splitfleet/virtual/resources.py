"""Actual worker limits are scoped and restored when Flower reuses the actor."""

import os
import platform
import resource
from contextlib import contextmanager

import torch


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if platform.system() == "Darwin" else 1024)


@contextmanager
def worker_resources(profile, *, mode):
    threads = torch.get_num_threads()
    affinity = None
    limited = profile.cpu_limit is not None or profile.affinity is not None
    if mode == "emulation" and limited:
        if platform.system() != "Linux" or not hasattr(os, "sched_getaffinity"):
            raise RuntimeError("CPU affinity emulation is unsupported on this platform")
        affinity = os.sched_getaffinity(0)
        chosen = (
            set(profile.affinity)
            if profile.affinity is not None
            else set(sorted(affinity)[: profile.cpu_limit])
        )
        if (
            not chosen
            or not chosen.issubset(affinity)
            or (profile.cpu_limit and len(chosen) < profile.cpu_limit)
        ):
            raise ValueError("Requested CPUs are outside the worker's allowed affinity")
    try:
        if affinity is not None:
            # Linux affinity is per thread. Apply to existing Torch/BLAS threads
            # too; newly created threads inherit the calling thread's mask.
            previous_masks = {}
            for entry in os.listdir("/proc/self/task"):
                tid = int(entry)
                try:
                    previous_masks[tid] = os.sched_getaffinity(tid)
                    os.sched_setaffinity(tid, chosen)
                except ProcessLookupError:
                    pass
        torch.set_num_threads(profile.torch_threads)
        if profile.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA profile cannot execute: no worker-visible GPU")
            torch.cuda.reset_peak_memory_stats(profile.device)
        yield {
            "resource_source": "emulated" if affinity is not None else "measured",
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else [],
            "torch_threads": torch.get_num_threads(),
        }
    finally:
        torch.set_num_threads(threads)
        if affinity is not None:
            for entry in os.listdir("/proc/self/task"):
                tid = int(entry)
                try:
                    os.sched_setaffinity(tid, previous_masks.get(tid, affinity))
                except ProcessLookupError:
                    pass
