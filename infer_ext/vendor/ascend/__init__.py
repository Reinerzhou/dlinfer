from .pytorch_patch import *
from .torch_npu_ops import *

def load_ops():
    import torch
    from pathlib import Path
    extension_file_path = str(Path(__file__).parent / "ascend_extension.so")
    if Path(extension_file_path).exists():
        torch.ops.load_library(extension_file_path)
