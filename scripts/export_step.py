"""Export an inference checkpoint from a numbered training snapshot (step-XXXXXXXX.pt)."""
import sys
import torch
from tiny_voca.config import Config
from tiny_voca.model import TinyVoca
from tiny_voca.train import export_model

source, target = sys.argv[1], sys.argv[2]
state = torch.load(source, map_location="cpu", weights_only=True)
model = TinyVoca(Config.from_dict(state["config"]) if hasattr(Config, "from_dict") else Config(**state["config"]))
model.load_state_dict({k[len("model."):]: v for k, v in state["system"].items() if k.startswith("model.")})
export_model(model, target, state["step"], state["annotation"])
print("exported step", state["step"], "->", target)
