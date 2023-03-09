import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import wandb
from albumentations.pytorch import ToTensorV2

import dataset as D
import networks as N
import utils as U


def setup_loaders(args):
    training_transforms = A.Compose(
        [
            A.Rotate(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]
    )
    training_images_transforms = A.Compose(
        [
            A.ColorJitter(),
            A.GaussianBlur(),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ]
    )
    training_masks_transforms = ToTensorV2()
    training_loader = D.setup_loader(args.training_dataset_dir, training_transforms, training_images_transforms, training_masks_transforms, args.batch_size, args.workers, True, True)

    evaluation_images_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ]
    )
    evaluation_masks_transforms = ToTensorV2()
    evaluation_loader = D.setup_loader(args.evaluation_dataset_dir, None, evaluation_images_transforms, evaluation_masks_transforms, args.batch_size, args.workers, False, True)

    return training_loader, evaluation_loader


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--training-dataset-dir", type=str, required=True)
    parser.add_argument("--evaluation-dataset-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--load-checkpoint-path", type=str)
    parser.add_argument("--logs-dir", type=str, required=True)
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-key", type=str)
    args = parser.parse_args()

    wandb_run = None
    if args.wandb_project is not None:
        wandb.login(key=args.wandb_key)
        wandb_run = wandb.init(
            project=args.wandb_project,
            job_type="training",
            config={
                "wandb-project": args.wandb_project,
                "training-dataset-dir": args.training_dataset_dir,
                "evaluation-dataset-dir": args.evaluation_dataset_dir,
                "batch-size": args.batch_size,
                "workers": args.workers,
                "epochs": args.epochs,
                "learning-rate": args.learning_rate,
                "checkpoints-dir": args.checkpoints_dir,
                "load-checkpoint-path": args.load_checkpoint_path,
                "logs-dir": args.logs_dir,
            }
        )

    training_loader, evaluation_loader = setup_loaders(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = N.UNet(3, training_loader.dataset.num_classes).to(device=device)
    criterion = nn.CrossEntropyLoss(weight=D.compute_median_frequency_class_balancing_weights(training_loader, device))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    checkpoint = {}
    if args.load_checkpoint_path:
        checkpoint, model, optimizer = U.load_checkpoint(args.load_checkpoint_path, model, optimizer, wandb_run)

    for epoch_index in range(checkpoint["epoch"]+1 if checkpoint else 0, args.epochs):
        U.train(epoch_index, training_loader, model, criterion, optimizer, scaler, device, wandb_run)
        metrics = U.evaluate(epoch_index, evaluation_loader, model, criterion, device, args.logs_dir, wandb_run)
        checkpoint = U.save_checkpoint(args.checkpoints_dir, epoch_index, model, optimizer, metrics, wandb_run)


if __name__ == "__main__":
    main()
