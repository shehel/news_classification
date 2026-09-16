"""Cluster GPU and CUDA diagnostic entrypoint for news_google_challenge."""
import os
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tracking import add_clearml_args, early_init, finalize_tracker

def main():
    _clearml_task = early_init()
    import argparse
    parser = argparse.ArgumentParser(description="Cluster GPU Diagnostic")
    add_clearml_args(parser)
    args = parser.parse_args()
    tracker = finalize_tracker(_clearml_task, args)

    print("=" * 60)
    print("CLUSTER GPU & ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"User: {os.environ.get('USER')}, UID: {os.getuid()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")

    print("\n--- 1. NVIDIA-SMI (with and without CUDA_VISIBLE_DEVICES) ---")
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print("Default env nvidia-smi:", res.stdout if res.stdout else res.stderr)
    except Exception as e:
        print("nvidia-smi failed:", e)

    try:
        clean_env = os.environ.copy()
        clean_env.pop("CUDA_VISIBLE_DEVICES", None)
        res = subprocess.run(["nvidia-smi"], env=clean_env, capture_output=True, text=True)
        print("Unset CUDA_VISIBLE_DEVICES nvidia-smi:", res.stdout if res.stdout else res.stderr)
    except Exception as e:
        print("nvidia-smi unset failed:", e)

    print("\n--- 2. Device Node Permissions & Access Test ---")
    import stat
    for dev in sorted(os.listdir("/dev")):
        if "nvidia" in dev:
            dev_path = os.path.join("/dev", dev)
            try:
                st = os.stat(dev_path)
                mode = oct(stat.S_IMODE(st.st_mode))
                print(f"  {dev_path}: mode={mode} uid={st.st_uid} gid={st.st_gid}")
                with open(dev_path, "rb") as f:
                    pass
                print(f"    -> Can read {dev_path}: YES")
            except Exception as e:
                print(f"    -> Access {dev_path} FAILED: {type(e).__name__} {e}")

    print("\n--- 2b. Singularity libraries (/.singularity.d/libs) ---")
    if os.path.exists("/.singularity.d/libs"):
        libs = [f for f in os.listdir("/.singularity.d/libs") if "cuda" in f or "nvidia" in f]
        print("Singularity cuda/nvidia libs:", libs[:15])
        for lib in libs[:5]:
            lib_path = os.path.join("/.singularity.d/libs", lib)
            print(f"  {lib} -> {os.path.realpath(lib_path)}")
    else:
        print("No /.singularity.d/libs directory")

    print("\n--- 2c. strace / debug nvidia-smi ---")
    try:
        res = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True)
        print("nvidia-smi -q output:\n", res.stdout[:500] if res.stdout else res.stderr[:500])
    except Exception as e:
        print("nvidia-smi -q failed:", e)

    print("\n--- 3. Installed PyTorch & NVIDIA Wheels ---")
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if any(k in line.lower() for k in ["torch", "cuda", "nvidia", "triton"]):
                print("  ", line)
    except Exception as e:
        print("pip list failed:", e)

    print("\n--- 4. Initial PyTorch CUDA Check ---")
    import torch
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        try:
            torch.cuda.init()
        except Exception as e:
            print("torch.cuda.init() failed with:", type(e), e)
    else:
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        x = torch.randn(10, 10, device="cuda")
        print("CUDA tensor allocation success:", x.device, x.shape)

    print("\n--- 5. Clean conflicting cu13 packages if CUDA failed ---")
    if not torch.cuda.is_available():
        print("Attempting to uninstall cuda-pathfinder, cuda-toolkit, and cu13 wheels...")
        uninstall_cmd = [
            sys.executable, "-m", "pip", "uninstall", "-y",
            "cuda-pathfinder", "cuda-toolkit", "cuda-bindings",
            "nvidia-cudnn-cu13", "nvidia-cublas-cu13", "nvidia-cuda-runtime-cu13",
            "nvidia-cuda-nvrtc-cu13", "nvidia-cuda-cupti-cu13", "nvidia-nccl-cu13",
            "nvidia-cusparselt-cu13", "nvidia-nvshmem-cu13",
        ]
        res = subprocess.run(uninstall_cmd, capture_output=True, text=True)
        print(res.stdout[-500:] if res.stdout else res.stderr)

        print("Testing PyTorch in fresh subprocess after removal:")
        test_script = "import torch; print('Subprocess torch.cuda.is_available():', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
        res = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True)
        print("Subprocess stdout:", res.stdout)
        print("Subprocess stderr:", res.stderr)

    print("=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    tracker.close()

if __name__ == "__main__":
    main()
