import sys
print(f"1. Python version: {sys.version.split()[0]}")
try:
    import torch
    print(f"2. torch version: {torch.__version__}")
    print(f"3. torch.version.cuda: {torch.version.cuda}")
except ImportError as e:
    print(f"2/3. torch import error: {e}")

try:
    import mamba_ssm
    print(f"4. mamba-ssm version/status: installed (version {getattr(mamba_ssm, '__version__', 'unknown')})")
    print("5. import mamba_ssm works: YES")
except Exception as e:
    print("4. mamba-ssm version/status: Error or Not Installed")
    print("5. import mamba_ssm works: NO")
    print(f"8. exact Mamba-SSM import/installation error: {type(e).__name__}: {e}")
