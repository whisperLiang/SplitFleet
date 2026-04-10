"""Memory efficiency utilities for SplitFleet.

Provides memory management helpers including:
- Context managers for memory cleanup
- Tensor pooling for reduced allocations
- Memory monitoring utilities
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Tuple

import torch


class TensorPool:
    """Pool for reusing tensor memory across operations.
    
    Reduces memory allocation overhead by maintaining a pool
    of tensors that can be reused instead of reallocated.
    """
    
    def __init__(self, max_size: int = 100):
        """Initialize the tensor pool.
        
        Args:
            max_size: Maximum number of tensors to keep in the pool.
        """
        self._pool: Dict[Tuple[Tuple[int, ...], torch.dtype, torch.device], torch.Tensor] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def get_or_create(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Get a tensor from the pool or create a new one.
        
        Args:
            shape: Tensor shape.
            dtype: Tensor dtype.
            device: Tensor device.
        
        Returns:
            A tensor (possibly reused from pool).
        """
        key = (shape, dtype, device)
        if key in self._pool:
            self._hits += 1
            return self._pool.pop(key)
        
        self._misses += 1
        return torch.empty(shape, dtype=dtype, device=device)
    
    def release(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool for reuse.
        
        Args:
            tensor: Tensor to release back to the pool.
        """
        if len(self._pool) >= self._max_size:
            return  # Pool full, let tensor be garbage collected
        
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        self._pool[key] = tensor
    
    def clear(self) -> int:
        """Clear all tensors from the pool.
        
        Returns:
            Number of tensors cleared.
        """
        count = len(self._pool)
        self._pool.clear()
        return count
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "pool_size": len(self._pool),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
        }


@contextmanager
def memory_cleanup(
    clear_cuda: bool = True,
    run_gc: bool = True,
) -> Generator[None, None, None]:
    """Context manager for automatic memory cleanup.
    
    Performs garbage collection and optional CUDA cache clearing
    on exit from the context.
    
    Args:
        clear_cuda: Whether to clear CUDA cache (if available).
        run_gc: Whether to run garbage collection.
    
    Example:
        with memory_cleanup():
            # Perform memory-intensive operations
            result = process_large_tensors()
        # Memory is cleaned up automatically
    """
    try:
        yield
    finally:
        if run_gc:
            gc.collect()
        if clear_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()


class MemoryMonitor:
    """Monitor memory usage during operations.
    
    Tracks peak memory usage and provides utilities for
    memory-aware execution.
    """
    
    def __init__(self) -> None:
        self._peak_memory: int = 0
        self._current_memory: int = 0
    
    def update(self) -> int:
        """Update current memory tracking.
        
        Returns:
            Current memory usage in bytes.
        """
        if torch.cuda.is_available():
            self._current_memory = torch.cuda.memory_allocated()
            self._peak_memory = max(self._peak_memory, self._current_memory)
        return self._current_memory
    
    @property
    def peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        return self._peak_memory
    
    @property
    def peak_memory_mb(self) -> float:
        """Get peak memory usage in megabytes."""
        return self._peak_memory / (1024 * 1024)
    
    def reset(self) -> None:
        """Reset memory tracking."""
        self._peak_memory = 0
        self._current_memory = 0


def get_memory_info() -> Dict[str, Any]:
    """Get current memory information.
    
    Returns:
        Dictionary with memory statistics.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_allocated": torch.cuda.memory_allocated(),
            "cuda_reserved": torch.cuda.memory_reserved(),
            "cuda_max_allocated": torch.cuda.max_memory_allocated(),
            "cuda_max_reserved": torch.cuda.max_memory_reserved(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
        })
    
    return info


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Optimize a model for inference.
    
    Applies memory optimizations suitable for inference mode:
    - Sets model to eval mode
    - Disables gradient computation
    
    Args:
        model: Model to optimize.
    
    Returns:
        The optimized model.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def clear_model_gradients(model: torch.nn.Module) -> None:
    """Efficiently clear all model gradients.
    
    Uses set_to_none=True for memory efficiency.
    
    Args:
        model: Model to clear gradients for.
    """
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    model.zero_grad(set_to_none=True)
