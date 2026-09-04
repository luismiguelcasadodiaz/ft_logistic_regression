import subprocess
import sys


def query_gpu_nvidia_smi():
    """Query GPU info via nvidia-smi (works for NVIDIA GPUs)."""
    try:
        query = (
            "index,name,driver_version,memory.total,memory.used,memory.free,"
            "temperature.gpu,utilization.gpu,utilization.memory,power.draw,"
            "power.limit,clocks.sm,clocks.mem"
        )
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        headers = query.split(",")
        gpus = []
        for line in result.stdout.strip().split("\n"):
            values = [v.strip() for v in line.split(",")]
            gpus.append(dict(zip(headers, values)))
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def query_gpu_pynvml():
    """Query GPU info via pynvml (more detailed, NVIDIA only)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "index": i,
                "name": pynvml.nvmlDeviceGetName(handle),
                "memory_total_MB": mem.total // (1024 ** 2),
                "memory_used_MB": mem.used // (1024 ** 2),
                "memory_free_MB": mem.free // (1024 ** 2),
                "temperature_C": pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU),
                "utilization_%": pynvml.nvmlDeviceGetUtilizationRates(
                    handle).gpu,
                "power_W": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            })
        pynvml.nvmlShutdown()
        return gpus
    except ImportError:
        print("pynvml not installed (pip install nvidia-ml-py)",
              file=sys.stderr)
        return None
    except Exception as e:
        print(f"pynvml error: {e}", file=sys.stderr)
        return None


def query_gpu_torch():
    """Query GPU info via PyTorch (if installed) — cross-platform (CUDA/MPS)"""
    try:
        import torch
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_MB": props.total_memory // (1024 ** 2),
                    "multi_processor_count": props.multi_processor_count,
                    "compute_capability": f"{props.major}.{props.minor}",
                })
        elif torch.backends.mps.is_available():
            gpus.append({"name": "Apple MPS (Metal) device", "backend": "mps"})
        return gpus if gpus else None
    except ImportError:
        return None


if __name__ == "__main__":
    print("=== nvidia-smi ===")
    smi_data = query_gpu_nvidia_smi()
    if smi_data:
        for gpu in smi_data:
            print(gpu)
    else:
        print("No NVIDIA GPU found via nvidia-smi (or not installed).")

    print("\n=== pynvml ===")
    nvml_data = query_gpu_pynvml()
    if nvml_data:
        for gpu in nvml_data:
            print(gpu)

    print("\n=== PyTorch ===")
    torch_data = query_gpu_torch()
    if torch_data:
        for gpu in torch_data:
            print(gpu)
    else:
        print("PyTorch not installed or no GPU detected.")
