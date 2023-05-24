from tqdm.auto import tqdm
import wandb
import torch

def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0  # number of examples seen
    batch_ct = 0
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30], gamma=0.1)
    for epoch in tqdm(range(config.epochs)):
        for _, (images, text, labels) in enumerate(loader):

            loss = train_batch(images, text, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
        scheduler.step()

def train_batch(images, text, labels, model, optimizer, criterion, device="cuda"):
    images, text, labels = images.to(device), text.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images, text)
    loss = criterion(outputs, labels)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")