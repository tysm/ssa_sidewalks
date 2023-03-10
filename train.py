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
    shared_image_transforms = [
        A.CLAHE(always_apply=True),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
            always_apply=True
        )
    ]
    if args.pretrained:
        shared_image_transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=1.0,
                always_apply=True
            )
        )

    training_transforms = A.Resize(256, 256)
    training_images_transforms = A.Compose(
        [
            *shared_image_transforms,
            ToTensorV2()
        ]
    )
    training_masks_transforms = ToTensorV2()

    if args.data_augmentation:
        training_transforms = A.Compose(
            [
                A.Resize(256, 256),
                A.Rotate(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ]
        )
        training_images_transforms = A.Compose(
            [
                A.ColorJitter(),
                A.GaussianBlur(),
                *shared_image_transforms,
                ToTensorV2()
            ]
        )
    training_loader = D.setup_loader(args.training_dataset_dir, training_transforms, training_images_transforms, training_masks_transforms, args.batch_size, args.workers, True, True)

    evaluation_transforms = A.Resize(256, 256)
    evaluation_images_transforms = A.Compose(
        [
            *shared_image_transforms,
            ToTensorV2()
        ]
    )
    evaluation_masks_transforms = ToTensorV2()
    evaluation_loader = D.setup_loader(args.evaluation_dataset_dir, evaluation_transforms, evaluation_images_transforms, evaluation_masks_transforms, args.batch_size, args.workers, False, True)

    return training_loader, evaluation_loader


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--training-dataset-dir", type=str, required=True)
    parser.add_argument("--evaluation-dataset-dir", type=str, required=True)
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--data-augmentation", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--class_balancing_weights", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--checkpoints-dir", type=str)
    parser.add_argument("--load-checkpoint-path", type=str)
    parser.add_argument("--logs-dir", type=str)
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
                "architecture": args.architecture,
                "pretrained": args.pretrained,
                "data-augmentation": args.data_augmentation,
                "batch-size": args.batch_size,
                "workers": args.workers,
                "class_balancing_weights": args.class_balancing_weights,
                "epochs": args.epochs,
                "learning-rate": args.learning_rate,
                "checkpoints-dir": args.checkpoints_dir,
                "load-checkpoint-path": args.load_checkpoint_path,
                "logs-dir": args.logs_dir,
            }
        )

    training_loader, evaluation_loader = setup_loaders(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = N.get_model(args.architecture, 3, training_loader.dataset.num_classes, args.pretrained).to(device=device)
    criterion = nn.CrossEntropyLoss(weight=D.compute_median_frequency_class_balancing_weights(training_loader, device) if args.class_balancing_weights else None)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    checkpoint = {}
    if args.load_checkpoint_path is not None:
        checkpoint, model, optimizer = U.load_checkpoint(args.load_checkpoint_path, model, optimizer, wandb_run=wandb_run)

    for epoch_index in range(checkpoint["epoch"]+1 if checkpoint else 0, args.epochs):
        U.train(epoch_index, training_loader, model, criterion, optimizer, scaler, device, wandb_run=wandb_run)
        metrics = U.evaluate(epoch_index, evaluation_loader, model, criterion, device, logs_dir=args.logs_dir, wandb_run=wandb_run)

        checkpoint = U.build_checkpoint(epoch_index, model, optimizer, metrics)
        if args.checkpoints_dir is not None:
            U.save_checkpoint(epoch_index, checkpoint, args.checkpoints_dir, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
