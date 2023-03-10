import os

import torch
import torchvision
from tqdm import tqdm

import metrics as M


def evaluate(epoch_index, loader, model, criterion, device, logs_dir=None, wandb_run=None):
    if len(loader) == 0:
        return

    with torch.no_grad():
        model.eval()

        color_tensor = torch.tensor(loader.dataset.class_colors).to(device=device)
        if logs_dir is not None:
            images_path = os.path.join(logs_dir, "images.png")
            masks_path = os.path.join(logs_dir, "masks.png")
            predictions_path = os.path.join(logs_dir, "predictions.png")

        iou_accumulator = M.IoU()
        accuracy_accumulator = M.Accuracy()
        loss_accumulator = 0

        with tqdm(loader, desc=f"Evaluating epoch {epoch_index}") as progress_container:
            for batch_index, (images, masks, _, _) in enumerate(progress_container):
                images = images.float().to(device=device)
                masks = masks.squeeze(dim=1).long().to(device=device)

                # Forward
                with torch.cuda.amp.autocast():
                    predictions = model(images)

                    iou_accumulator.evaluate(predictions, masks, device)
                    accuracy_accumulator.evaluate(predictions, masks, device)
                    loss_accumulator += criterion(predictions, masks).item()

                    predicted_masks = torch.argmax(predictions, dim=1)

                # Save batch images
                colored_masks = color_tensor[masks].permute(0, 3, 1, 2).float()/255.0
                colored_predicted_masks = color_tensor[predicted_masks].permute(0, 3, 1, 2).float()/255.0
                if logs_dir is not None:
                    torchvision.utils.save_image(images, images_path)
                    torchvision.utils.save_image(colored_masks, masks_path)
                    torchvision.utils.save_image(colored_predicted_masks, predictions_path)

                # Update tqdm
                _iou = iou_accumulator.iou()
                _mean_iou = iou_accumulator.mean_iou().item()
                _accuracy = accuracy_accumulator.accuracy()
                _mean_accuracy = accuracy_accumulator.mean_accuracy().item()
                _loss = loss_accumulator/(batch_index+1)
                progress_container.set_postfix(batch=batch_index, miou=_mean_iou, macc=_mean_accuracy, iou=_iou, acc=_accuracy, loss=_loss)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch_index,
                    "evaluation_iou": iou_accumulator.iou().tolist(),
                    "evaluation_miou": iou_accumulator.mean_iou().item(),
                    "evaluation_acc": accuracy_accumulator.accuracy().tolist(),
                    "evaluation_macc": accuracy_accumulator.mean_accuracy().item(),
                    "evaluation_loss": loss_accumulator/len(loader)
                },
                step=epoch_index
            )
        return {
            "iou": iou_accumulator.iou().tolist(),
            "miou": iou_accumulator.mean_iou().item(),
            "acc": accuracy_accumulator.accuracy().tolist(),
            "macc": accuracy_accumulator.mean_accuracy().item(),
            "loss": loss_accumulator/len(loader)
        }
