

import torch

from alphago_nn import AlphaZeroNet
from model_manager import ModelCheckpointManager
    # device = torch.device("cpu")
if torch.backends.mps.is_available():
    print("Init for mac")
    device = torch.device("mps")
else:
    print("User cpu")
    device = torch.device("cpu")


size = 11;
input_dim = (17, size, size)
action_count = size * size + 1

input1 = torch.rand((1, 17, 11, 11)).to(device)
input2 = torch.zeros((1, 17, 11, 11)).to(device)
model_manager = ModelCheckpointManager(type(AlphaZeroNet), "/Users/shikaijin/Desktop/projects/Models/after-fix")

infer_model = AlphaZeroNet(input_dim, action_count).to(device)
weights = model_manager.load_latest(device)
if weights is not None:
    print("loading weights")
    infer_model.load_state_dict(weights)

with torch.no_grad():
    p1, v1 = infer_model(input1)
    p2, v2 = infer_model(input2)

print("π1 = π2?", torch.allclose(p1, p2, atol=1e-5))
print("v1 =", v1.item(), "v2 =", v2.item())
