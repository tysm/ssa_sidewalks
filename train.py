import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

import dataset as D
import networks as N
from utils import save_checkpoint, load_checkpoint, train, evaluate


def setup_loaders(args):
    training_transforms = A.Compose(
        [
            A.Resize(height=64, width=64),
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

    evaluation_transforms = A.Resize(height=64, width=64)
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
    evaluation_loader = D.setup_loader(args.evaluation_dataset_dir, evaluation_transforms, evaluation_images_transforms, evaluation_masks_transforms, args.batch_size, args.workers, False, True)

    return training_loader, evaluation_loader


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--training-dataset-dir", type=str)
    parser.add_argument("--evaluation-dataset-dir", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoints-dir", type=str)
    parser.add_argument("--load-checkpoint-path", type=str)
    parser.add_argument("--logs-dir", type=str)
    args = parser.parse_args()

    training_loader, evaluation_loader = setup_loaders(args)
    model = N.UNet(3, training_loader.dataset.num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss(weight=D.compute_median_frequency_class_balancing_weights(training_loader, device))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    checkpoint = {}
    if args.load_checkpoint_path:
        checkpoint, model, optimizer = load_checkpoint(args.load_checkpoint_path, model, optimizer)

    for epoch_index in range(checkpoint["epoch"]+1 if checkpoint else 0, args.epochs):
        train(epoch_index, training_loader, model, criterion, optimizer, scaler, device)
        metrics = evaluate(args.logs_dir, epoch_index, evaluation_loader, model, criterion, device)
        checkpoint = save_checkpoint(args.checkpoints_dir, epoch_index, model, optimizer, metrics)


if __name__ == "__main__":
    main()
