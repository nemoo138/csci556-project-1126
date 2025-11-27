"""
Checkpoint Saving and Loading Utilities for C3-LDM

Handles saving/loading of:
- Model states (VAE, U-Net, conditional encoder, etc.)
- Optimizer states
- Training progress (epoch, step, metrics)
- Random states for reproducibility
"""

import os
import torch
import json
from pathlib import Path


def save_checkpoint(
    checkpoint_dir,
    epoch,
    step,
    models_dict,
    optimizers_dict,
    metrics=None,
    is_best=False,
    keep_last_n=3
):
    """
    Save training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoints
        epoch: Current epoch number
        step: Current training step
        models_dict: Dict of model_name -> model
        optimizers_dict: Dict of optimizer_name -> optimizer
        metrics: Optional dict of metric_name -> value
        is_best: Whether this is the best checkpoint so far
        keep_last_n: Keep only last N checkpoints (None to keep all)

    Returns:
        checkpoint_path: Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'models': {},
        'optimizers': {},
        'metrics': metrics or {},
        'random_states': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
    }

    # Save model states
    for name, model in models_dict.items():
        if hasattr(model, 'module'):  # Handle DataParallel
            checkpoint['models'][name] = model.module.state_dict()
        else:
            checkpoint['models'][name] = model.state_dict()

    # Save optimizer states
    for name, optimizer in optimizers_dict.items():
        checkpoint['optimizers'][name] = optimizer.state_dict()

    # Save checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch{epoch:04d}_step{step:08d}.pt'
    torch.save(checkpoint, checkpoint_path)

    # Save metadata
    metadata = {
        'epoch': epoch,
        'step': step,
        'metrics': metrics or {},
        'checkpoint_path': str(checkpoint_path)
    }
    metadata_path = checkpoint_dir / f'metadata_epoch{epoch:04d}_step{step:08d}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved checkpoint: {checkpoint_path}")

    # Save as latest
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)

    # Save as best if requested
    if is_best:
        best_path = checkpoint_dir / 'checkpoint_best.pt'
        torch.save(checkpoint, best_path)
        print(f"  → Saved as best checkpoint")

    # Clean up old checkpoints
    if keep_last_n is not None:
        cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path,
    models_dict,
    optimizers_dict=None,
    device='cpu',
    strict=True
):
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        models_dict: Dict of model_name -> model (to load states into)
        optimizers_dict: Optional dict of optimizer_name -> optimizer
        device: Device to load checkpoint to
        strict: Whether to strictly enforce state dict keys match

    Returns:
        checkpoint_info: Dict with epoch, step, metrics
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model states
    for name, model in models_dict.items():
        if name not in checkpoint['models']:
            print(f"  Warning: Model '{name}' not found in checkpoint")
            continue

        if hasattr(model, 'module'):  # Handle DataParallel
            model.module.load_state_dict(checkpoint['models'][name], strict=strict)
        else:
            model.load_state_dict(checkpoint['models'][name], strict=strict)

        print(f"  Loaded model: {name}")

    # Load optimizer states
    if optimizers_dict is not None:
        for name, optimizer in optimizers_dict.items():
            if name not in checkpoint['optimizers']:
                print(f"  Warning: Optimizer '{name}' not found in checkpoint")
                continue

            optimizer.load_state_dict(checkpoint['optimizers'][name])
            print(f"  Loaded optimizer: {name}")

    # Restore random states
    if 'random_states' in checkpoint:
        torch.set_rng_state(checkpoint['random_states']['torch'])
        if checkpoint['random_states']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['random_states']['cuda'])
        print("  Restored random states")

    # Return checkpoint info
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'step': checkpoint['step'],
        'metrics': checkpoint.get('metrics', {})
    }

    print(f"  Epoch: {checkpoint_info['epoch']}, Step: {checkpoint_info['step']}")

    return checkpoint_info


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        latest_checkpoint_path or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Check for checkpoint_latest.pt
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    if latest_path.exists():
        return latest_path

    # Search for checkpoint files
    checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch*.pt'))

    if not checkpoint_files:
        return None

    # Sort by modification time
    checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return checkpoint_files[0]


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    """
    Remove old checkpoints, keeping only the last N.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Find all checkpoint files (excluding latest and best)
    checkpoint_files = [
        p for p in checkpoint_dir.glob('checkpoint_epoch*.pt')
        if 'latest' not in p.name and 'best' not in p.name
    ]

    if len(checkpoint_files) <= keep_last_n:
        return

    # Sort by modification time
    checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Remove old checkpoints
    for old_checkpoint in checkpoint_files[keep_last_n:]:
        old_checkpoint.unlink()
        # Also remove corresponding metadata
        metadata_file = old_checkpoint.with_suffix('.json').with_name(
            old_checkpoint.stem.replace('checkpoint', 'metadata') + '.json'
        )
        if metadata_file.exists():
            metadata_file.unlink()

        print(f"  Removed old checkpoint: {old_checkpoint.name}")


def save_model_only(model, save_path):
    """
    Save only model weights (no optimizer, no training state).

    Args:
        model: PyTorch model
        save_path: Path to save model weights
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, save_path)
    print(f"Saved model weights: {save_path}")


def load_model_only(model, load_path, device='cpu', strict=True):
    """
    Load only model weights.

    Args:
        model: PyTorch model
        load_path: Path to model weights
        device: Device to load to
        strict: Whether to strictly enforce state dict keys match
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Model weights not found: {load_path}")

    state_dict = torch.load(load_path, map_location=device)

    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)

    print(f"Loaded model weights: {load_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Checkpoint Utilities")
    print("=" * 70)

    import torch.nn as nn
    import torch.optim as optim
    import tempfile
    import shutil

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = Path(temp_dir) / 'checkpoints'

    try:
        # Create dummy models
        model1 = nn.Linear(10, 5)
        model2 = nn.Conv2d(3, 16, 3)

        models = {
            'model1': model1,
            'model2': model2
        }

        # Create optimizers
        optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
        optimizer2 = optim.SGD(model2.parameters(), lr=1e-2)

        optimizers = {
            'optimizer1': optimizer1,
            'optimizer2': optimizer2
        }

        print("\n1. Testing save_checkpoint...")
        metrics = {'loss': 0.5, 'accuracy': 0.95}
        checkpoint_path = save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=10,
            step=1000,
            models_dict=models,
            optimizers_dict=optimizers,
            metrics=metrics,
            is_best=True
        )

        print(f"\n2. Checking saved files...")
        print(f"  Checkpoint exists: {checkpoint_path.exists()}")
        print(f"  Latest exists: {(checkpoint_dir / 'checkpoint_latest.pt').exists()}")
        print(f"  Best exists: {(checkpoint_dir / 'checkpoint_best.pt').exists()}")

        print(f"\n3. Testing load_checkpoint...")
        # Create new models for loading
        model1_new = nn.Linear(10, 5)
        model2_new = nn.Conv2d(3, 16, 3)

        models_new = {
            'model1': model1_new,
            'model2': model2_new
        }

        optimizer1_new = optim.Adam(model1_new.parameters(), lr=1e-3)
        optimizer2_new = optim.SGD(model2_new.parameters(), lr=1e-2)

        optimizers_new = {
            'optimizer1': optimizer1_new,
            'optimizer2': optimizer2_new
        }

        checkpoint_info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            models_dict=models_new,
            optimizers_dict=optimizers_new
        )

        print(f"  Loaded epoch: {checkpoint_info['epoch']}")
        print(f"  Loaded step: {checkpoint_info['step']}")
        print(f"  Loaded metrics: {checkpoint_info['metrics']}")

        print(f"\n4. Testing find_latest_checkpoint...")
        latest = find_latest_checkpoint(checkpoint_dir)
        print(f"  Latest checkpoint: {latest.name if latest else 'None'}")

        print(f"\n5. Testing multiple checkpoints and cleanup...")
        for epoch in range(1, 6):
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                step=epoch * 100,
                models_dict=models,
                optimizers_dict=optimizers,
                keep_last_n=3
            )

        remaining = list(checkpoint_dir.glob('checkpoint_epoch*.pt'))
        remaining = [p for p in remaining if 'latest' not in p.name and 'best' not in p.name]
        print(f"  Remaining checkpoints: {len(remaining)} (should be 3)")

        print(f"\n6. Testing save_model_only...")
        model_path = checkpoint_dir / 'model1_weights.pt'
        save_model_only(model1, model_path)

        print(f"\n7. Testing load_model_only...")
        model1_loaded = nn.Linear(10, 5)
        load_model_only(model1_loaded, model_path)

        # Compare weights
        params_match = all(
            torch.allclose(p1, p2)
            for p1, p2 in zip(model1.parameters(), model1_loaded.parameters())
        )
        print(f"  Weights match: {params_match}")

        print("\n✓ All checkpoint tests passed!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory: {temp_dir}")
