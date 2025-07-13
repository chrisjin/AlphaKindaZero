import os
import torch
import shutil
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

    def get_checkpoint_files(self) -> list:
        """Get all checkpoint files sorted by modification time (newest first)."""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "*.pt"))
        if not checkpoint_files:
            return []
        
        # Sort by modification time, newest first
        sorted_files = sorted(checkpoint_files, key=os.path.getmtime, reverse=True)
        return sorted_files

    def load_by_index(self, index: int, device: torch.device) -> Any:
        """
        Load model by index, where index=0 is the latest, index=1 is second to last, etc.
        
        Args:
            index: Index of the model to load (0 = latest, 1 = second to last, etc.)
            device: Device to load the model on
            
        Returns:
            Model state dict or None if index is out of range
        """
        checkpoint_files = self.get_checkpoint_files()
        
        if index >= len(checkpoint_files):
            print(f"⚠️  No model found at index {index}. Available models: {len(checkpoint_files)}")
            return None
        
        model_file = checkpoint_files[index]
        print(f"📦 Loading model from {model_file} (index {index})")
        
        return torch.load(model_file, map_location=device)

    def get_model_info(self) -> list:
        """
        Get information about all available models.
        
        Returns:
            List of tuples (index, filename, modification_time)
        """
        checkpoint_files = self.get_checkpoint_files()
        model_info = []
        
        for i, file_path in enumerate(checkpoint_files):
            filename = os.path.basename(file_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            model_info.append((i, filename, mod_time))
        
        return model_info

    def print_available_models(self):
        """Print all available models with their indices."""
        model_info = self.get_model_info()
        
        if not model_info:
            print("📁 No models found in checkpoint directory")
            return
        
        print(f"📁 Available models in {self.checkpoint_dir}:")
        for index, filename, mod_time in model_info:
            print(f"  [{index}] {filename} ({mod_time.strftime('%Y-%m-%d %H:%M:%S')})")

    def move_model_by_index(self, index: int, dest_dir: str, move_file: bool = False) -> bool:
        """
        Move or copy a model to a new directory by index, preserving the original filename.
        
        Args:
            index: Index of the model to move (0 = latest, 1 = second to last, etc.)
            dest_dir: Destination directory where the model should be moved/copied to
            move_file: If True, move the file (delete original). If False, copy the file.
            
        Returns:
            True if successful, False otherwise
        """
        checkpoint_files = self.get_checkpoint_files()
        
        if index >= len(checkpoint_files):
            print(f"⚠️  No model found at index {index}. Available models: {len(checkpoint_files)}")
            return False
        
        source_file = checkpoint_files[index]
        filename = os.path.basename(source_file)
        dest_path = os.path.join(dest_dir, filename)
        
        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)
        
        try:
            if move_file:
                shutil.move(source_file, dest_path)
                action = "moved"
            else:
                shutil.copy2(source_file, dest_path)
                action = "copied"
            
            print(f"✅ Model at index {index} ({filename}) {action} to {dest_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to {action} model at index {index}: {e}")
            return False

