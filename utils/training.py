import torch
from tqdm import tqdm

def train(epoch_index, loader, model, criterion, optimizer, scaler, device, wandb_run=None):
    if len(loader) == 0:
        return

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
    if wandb_run is not None:
        wandb_run.log(
            {
                "training_loss": loss_accumulator/len(loader)
            },
            step=epoch_index
        )
