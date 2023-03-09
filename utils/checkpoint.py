import os

import torch


def save_checkpoint(checkpoints_dir, epoch_index, model, optimizer, metrics):
    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch_index}.pth.tar")
    print(f'Saving checkpoint to "{checkpoint_path}"')

    checkpoint = {
        "epoch": epoch_index,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }
    torch.save(checkpoint, checkpoint_path)

    print(f'Saved checkpoint to "{checkpoint_path}"')
    return checkpoint


def load_checkpoint(checkpoint_path, model, optimizer):
    print(f'Loading checkpoint from "{checkpoint_path}"')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f'Loaded checkpoint from "{checkpoint_path}"')
    return checkpoint, model, optimizer
