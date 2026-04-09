#!/usr/bin/env python3
"""
Quick test to verify Python environment and imports
"""

import sys

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("")

try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        print(f"  MPS available: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import torchvision
    print(f"✓ TorchVision version: {torchvision.__version__}")
except ImportError as e:
    print(f"✗ TorchVision import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"✓ Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"✓ Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")
    sys.exit(1)

try:
    import seaborn as sns
    print(f"✓ Seaborn version: {sns.__version__}")
except ImportError as e:
    print(f"✗ Seaborn import failed: {e}")
    sys.exit(1)

try:
    import scipy
    print(f"✓ SciPy version: {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy import failed: {e}")
    sys.exit(1)

try:
    import sklearn
    print(f"✓ Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")
    sys.exit(1)

try:
    import tqdm
    print(f"✓ tqdm version: {tqdm.__version__}")
except ImportError as e:
    print(f"✗ tqdm import failed: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print(f"✓ Pillow available")
except ImportError as e:
    print(f"✗ Pillow import failed: {e}")
    sys.exit(1)

print("")
print("All imports successful!")
print("")

try:
    from vgg_wongwang_lim_data import LIMDataset
    print("✓ vgg_wongwang_lim_data import successful")
except ImportError as e:
    print(f"✗ vgg_wongwang_lim_data import failed: {e}")
    sys.exit(1)

try:
    from vgg_wongwang_lim import VGGWongWangLIM
    print("✓ vgg_wongwang_lim import successful")
except ImportError as e:
    print(f"✗ vgg_wongwang_lim import failed: {e}")
    sys.exit(1)

try:
    from train_stage1_classification import train_epoch
    print("✓ train_stage1_classification import successful")
except ImportError as e:
    print(f"✗ train_stage1_classification import failed: {e}")
    sys.exit(1)

try:
    from train_stage2_rt_fitting import train_epoch as train_epoch2
    print("✓ train_stage2_rt_fitting import successful")
except ImportError as e:
    print(f"✗ train_stage2_rt_fitting import failed: {e}")
    sys.exit(1)

try:
    from evaluate_vgg_wongwang_lim import evaluate_model
    print("✓ evaluate_vgg_wongwang_lim import successful")
except ImportError as e:
    print(f"✗ evaluate_vgg_wongwang_lim import failed: {e}")
    sys.exit(1)

print("")
print("==============================================")
print("All imports successful! Ready to train.")
print("==============================================")
