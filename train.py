import argparse
import os
import random
import math

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils import data

import metrics as M
import networks as NN
from dataset import Dataset


def setup_loaders(dataset_dir, train_percentage, batch_size, workers):
    items = Dataset.find_items(os.path.join(dataset_dir, "images"), os.path.join(dataset_dir, "masks"), "jpg", "png")
    random.shuffle(items)
    label_colors = Dataset.read_label_colors(dataset_dir)

    train_items = items[:math.ceil(len(items)*train_percentage)]
    train_transforms = A.Compose(
        [
            A.Resize(height=64, width=64),
            A.Rotate(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]
    )
    train_images_transforms = A.Compose(
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
    train_masks_transforms = ToTensorV2()
    train_dataset = Dataset(train_items, label_colors, train_transforms, train_images_transforms, train_masks_transforms)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True)

    val_items = items[math.ceil(len(items)*train_percentage):]
    val_transforms = A.Resize(height=64, width=64)
    val_image_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ]
    )
    val_masks_transforms = ToTensorV2()
    val_dataset = Dataset(val_items, label_colors, val_transforms, val_image_transforms, val_masks_transforms)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False, drop_last=True)

    return train_loader, val_loader


def load_checkpoint(checkpoint_path, model, optimizer):
    print(f'Loading checkpoint from "{checkpoint_path}"')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f'Loaded checkpoint from "{checkpoint_path}"')
    return checkpoint, model, optimizer


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


def train(epoch_index, loader, model, criterion, optimizer, scaler, device):
    model.train()
    loss_accumulator = torch.zeros(1).float().to(device=device)
    with tqdm(loader, desc=f"Training epoch {epoch_index}") as progress_container:
        for batch_index, (images, masks, _, _) in enumerate(progress_container):
            images = images.to(device=device)
            masks = masks.squeeze(dim=1).long().to(device=device)

            # Forward
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss = criterion(predictions, masks)

                loss_accumulator += loss

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update tqdm
            _loss = (loss_accumulator/(batch_index+1)).item()
            progress_container.set_postfix(batch=batch_index, loss=_loss)
    model.eval()


def evaluate(logs_dir, epoch_index, loader, model, criterion, device):
    model.eval()

    color_tensor = torch.tensor(loader.dataset.label_colors)
    images_path = os.path.join(logs_dir, "images.png")
    masks_path = os.path.join(logs_dir, "masks.png")
    predictions_path = os.path.join(logs_dir, "predictions.png")

    iou_accumulator = M.IoU()
    accuracy_accumulator = M.Accuracy()
    loss_accumulator = torch.zeros(1).float().to(device=device)

    with torch.no_grad():
        with tqdm(loader, desc=f"Evaluating epoch {epoch_index}") as progress_container:
            for batch_index, (images, masks, _, _) in enumerate(progress_container):
                images = images.to(device=device)
                masks = masks.squeeze(dim=1).long().to(device=device)

                # Forward
                with torch.cuda.amp.autocast():
                    predictions = model(images)

                    iou_accumulator.evaluate(predictions, masks)
                    accuracy_accumulator.evaluate(predictions, masks)
                    loss_accumulator += criterion(predictions, masks)

                    predicted_masks = torch.argmax(predictions, dim=1)

                # Save batch images
                colored_masks = color_tensor[masks].permute(0, 3, 1, 2).float()/255.0
                colored_predicted_masks = color_tensor[predicted_masks].permute(0, 3, 1, 2).float()/255.0
                torchvision.utils.save_image(images, images_path)
                torchvision.utils.save_image(colored_masks, masks_path)
                torchvision.utils.save_image(colored_predicted_masks, predictions_path)

                # Update tqdm
                _iou = iou_accumulator.iou()
                _mean_iou = iou_accumulator.mean_iou().item()
                _accuracy = accuracy_accumulator.accuracy()
                _mean_accuracy = accuracy_accumulator.mean_accuracy().item()
                _loss = (loss_accumulator/(batch_index+1)).item()
                progress_container.set_postfix(batch=batch_index, miou=_mean_iou, macc=_mean_accuracy, iou=_iou, acc=_accuracy, loss=_loss)
    return {
        "iou": iou_accumulator.iou().tolist(),
        "miou": iou_accumulator.mean_iou().item(),
        "acc": accuracy_accumulator.accuracy().tolist(),
        "macc": accuracy_accumulator.mean_accuracy().item(),
        "loss": (loss_accumulator/loader.size()).item(),
    }


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--dataset-dir", type=str)
    parser.add_argument("--train-percentage", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoints-dir", type=str)
    parser.add_argument("--load-checkpoint-path", type=str)
    parser.add_argument("--logs-dir", type=str)
    args = parser.parse_args()

    train_loader, val_loader = setup_loaders(args.dataset_dir, args.train_percentage, args.batch_size, args.workers)
    model = NN.UNet(3, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = {}
    if args.load_checkpoint_path:
        checkpoint, model, optimizer = load_checkpoint(args.load_checkpoint_path, model, optimizer)

    for epoch_index in range(checkpoint["epoch"]+1 if checkpoint else 0, args.epochs):
        train(epoch_index, train_loader, model, criterion, optimizer, scaler, device)
        metrics = evaluate(args.logs_dir, epoch_index, val_loader, model, criterion, device)
        checkpoint = save_checkpoint(args.checkpoints_dir, epoch_index, model, optimizer, metrics)


if __name__ == "__main__":
    main()
