import sys, pathlib, types
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# Stub heavy optional dependencies before importing the module
sys.modules.setdefault('transformers', types.ModuleType('transformers'))
sys.modules.setdefault('safetensors', types.ModuleType('safetensors'))
sys.modules.setdefault("model", types.ModuleType("model"))
sys.modules["model"].Transformer = object
sys.modules["model"].ModelArgs = object
safetensors_torch = types.ModuleType("safetensors.torch")
safetensors_torch.load_model = lambda *args, **kwargs: None
sys.modules["safetensors.torch"] = safetensors_torch
sys.modules["transformers"].AutoTokenizer = object
sys.modules['safetensors'].torch = types.ModuleType('torch')
sys.modules['safetensors'].torch.load_model = lambda *args, **kwargs: None
import torch
from inference.generate import sample

def test_sample_top_k_restricts_indices():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    torch.manual_seed(0)
    token = sample(logits, temperature=1.0, top_k=1).item()
    assert token == 2
