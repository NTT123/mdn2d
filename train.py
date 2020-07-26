"""Training functions."""

import tqdm


def train_one_epoch(epoch, model, dataloader, optimizer, lr_scheduler, device,
                    hx, logger):
    model.train()

    for batch in tqdm.tqdm(dataloader, desc=f"epoch {epoch}"):
        inp = batch.to(device)
        output = model(hx)
        loss = -model.log_likelihood(output, inp).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        logger.add_scalar('loss', loss.item())
        logger.add_scalar('lr', lr_scheduler.get_last_lr()[0])
