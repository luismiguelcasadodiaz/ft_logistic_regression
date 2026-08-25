"""
Matrix Multiplication Benchmark: CPU vs GPU (Quadro P620)
"""

import time
import numpy as np
import torch

def benchmark_cpu(size, dtype=np.float32, n_runs=3):
    """Matrix multiply on CPU using NumPy."""
    a = np.random.rand(size, size).astype(dtype)
    b = np.random.rand(size, size).astype(dtype)

    # warm-up (helps normalize CPU caches/frequency scaling)
    _ = a @ b

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = a @ b
        times.append(time.perf_counter() - start)

    return min(times)  # best-case, reduces noise


def benchmark_gpu(size, device, dtype=torch.float32, n_runs=3):
    """Matrix multiply on GPU using PyTorch + CUDA."""
    a = torch.rand(size, size, dtype=dtype, device=device)
    b = torch.rand(size, size, dtype=dtype, device=device)

    # warm-up: first CUDA call includes context/kernel init overhead
    torch.cuda.synchronize()
    _ = a @ b
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = a @ b
        torch.cuda.synchronize()  # wait for GPU to actually finish
        times.append(time.perf_counter() - start)

    return min(times)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Check driver/PyTorch CUDA build.")
        return
    print(torch.cuda.get_arch_list())
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU detected: {gpu_name}")
    print(f"PyTorch version: {torch.__version__}, CUDA: {torch.version.cuda}\n")

    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    print(f"{'Size':>6} | {'CPU (s)':>10} | {'GPU (s)':>10} | {'Speedup':>8}")
    print("-" * 45)

    for size in sizes:
        cpu_time = benchmark_cpu(size)
        try:
            gpu_time = benchmark_gpu(size, device)
        except RuntimeError as e:
            print(f"{size:>6} | GPU error (likely out of memory): {e}")
            continue

        speedup = cpu_time / gpu_time
        print(f"{size:>6} | {cpu_time:>10.5f} | {gpu_time:>10.5f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    main()