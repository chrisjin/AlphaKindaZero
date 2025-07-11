import os
import torch
from datetime import datetime
from typing import Any, Type
import glob


class ModelCheckpointManager:
    def __init__(self, model_class: Type[torch.nn.Module], checkpoint_dir: str):
        self.model_class = model_class
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model: torch.nn.Module):
        """Save model with timestamp filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.checkpoint_dir, f"{timestamp}.pt")
        torch.save(model.state_dict(), path)
        print(f"✅ Model saved to {path}")

    def load_latest(self, device: torch.device) -> Any:
        """Load model with the latest timestamp."""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "*.pt"))
        if not checkpoint_files:
            return None

        latest_file = max(checkpoint_files, key=os.path.getmtime)
        print(f"📦 Loading model from {latest_file}")

        return torch.load(latest_file, map_location=device)
