import os

import torch
import torchvision
from tqdm import tqdm
from torch.utils import data

import metrics as M


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


def train(epoch_index, loader, model, criterion, optimizer, scaler, device):
    model.train()
    loss_accumulator = 0
    with tqdm(loader, desc=f"Training epoch {epoch_index}") as progress_container:
        for batch_index, (images, masks, _, _) in enumerate(progress_container):
            images = images.to(device=device)
            masks = masks.squeeze(dim=1).long().to(device=device)

            # Forward
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss = criterion(predictions, masks)

                loss_accumulator += loss.item()

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update tqdm
            _loss = loss_accumulator/(batch_index+1)
            progress_container.set_postfix(batch=batch_index, loss=_loss)
    model.eval()


def evaluate(logs_dir, epoch_index, loader, model, criterion, device):
    with torch.no_grad():
        model.eval()

        color_tensor = torch.tensor(loader.dataset.class_colors).to(device=device)
        images_path = os.path.join(logs_dir, "images.png")
        masks_path = os.path.join(logs_dir, "masks.png")
        predictions_path = os.path.join(logs_dir, "predictions.png")

        iou_accumulator = M.IoU()
        accuracy_accumulator = M.Accuracy()
        loss_accumulator = torch.zeros(1).float().to(device=device)

        with tqdm(loader, desc=f"Evaluating epoch {epoch_index}") as progress_container:
            for batch_index, (images, masks, _, _) in enumerate(progress_container):
                images = images.to(device=device)
                masks = masks.squeeze(dim=1).long().to(device=device)

                # Forward
                with torch.cuda.amp.autocast():
                    predictions = model(images)

                    iou_accumulator.evaluate(predictions, masks, device)
                    accuracy_accumulator.evaluate(predictions, masks, device)
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
            "loss": (loss_accumulator/len(loader)).item(),
        }
