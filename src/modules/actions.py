import os

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

from utils.count_ced_for_points import val_ced_calculate
from data.augs import vis_keypoints
from PIL import Image


def train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        scheduler,
        train_cfg,
        save_dir,
        logger=None,
        task=None
) -> None:
    log_frequency = max(1, len(train_dataloader) // 10)
    device = train_cfg.get("device")


    model.to(device)
    model.train()
    optimizer.zero_grad()

    pbar = tqdm.tqdm(range(1, train_cfg.get("max_epoch")+1), ncols=150)
    total_it = 0
    best_val_loss = 100000
    for epoch in pbar:
        pbar.set_description(f"Train {epoch}/{train_cfg.get('max_epoch')}: ")
        avg_loss = torch.tensor(0.)
        pbar_batch = tqdm.tqdm(train_dataloader, ncols=150)
        for idx, batch in enumerate(pbar_batch, 1):
            total_it += 1
            pbar_batch.set_description(f"Batch {idx}/{len(train_dataloader)}: ")
            images, targets, positions, _, mask = batch
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            mask = mask.to(device)
            positions = positions.to(device)
            # out_permuted = torch.gather(output, 1, positions)
            loss = criterion(output, targets.view(targets.shape[0], -1), mask)
            # loss = criterion(output, targets.view(targets.shape[0], -1), mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = (loss.detach().cpu() + avg_loss * max(1, idx - 1)) / (idx)

            if idx % log_frequency == 0 or idx == len(train_dataloader):
                line = f"batch {idx}/{len(train_dataloader)}: loss {loss}"
                pbar.write(line)
                logger.report_scalar("Loss", "Train", iteration=total_it, value=avg_loss)

        if scheduler:
            scheduler.step()
            line = f"lr: {scheduler.get_last_lr()}"
            pbar.write(line)
            for ilr, lr in enumerate(scheduler.get_last_lr()):
                logger.report_scalar(f"LR", f"LR_group_{ilr}", iteration=total_it, value=lr)

        line = f"Train epoch {epoch}/{train_cfg.get('max_epoch')}: loss {avg_loss}"
        pbar.write(line)

        if epoch % train_cfg.get("test_frequency") == 0 or epoch == train_cfg.get("max_epoch"):
            path2save = os.path.join(save_dir, f"epoch_{epoch}")
            os.makedirs(os.path.join(path2save, "ced"), exist_ok=True)
            val_loss = val(model, val_dataloader, criterion, device, pbar, path2save, task, logger, total_it, epoch)
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.pth"))
                best_val_loss = val_loss

        if epoch % train_cfg.get("save_frequency") == 0 or epoch == train_cfg.get("max_epoch"):
            path2save = os.path.join(save_dir, f"epoch_{epoch}")
            os.makedirs(path2save, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(path2save, "model.pth"))


@torch.no_grad()
def val(model, dataloader, criterion, device, pbar_epoch, output_path, task, logger, total_it, epoch):
    log_frequency = max(1, len(dataloader) // 10)

    model.to(device)
    model.eval()

    pbar = tqdm.tqdm(dataloader, ncols=10)
    avg_loss = torch.tensor(0.)

    gt_points = {}
    predicted_points = {"val": {}}
    for idx, batch in enumerate(pbar, 1):
        pbar.set_description(f"Val batch {idx}/{len(dataloader)}: ")
        images, targets, positions, _, mask = batch
        images = images.to(device)
        targets = targets.to(device)
        output = model(images) #.view(images.shape[0], -1, 2)
        mask = mask.to(device)
        positions = positions.to(device)
        # out_permuted = torch.gather(output, 1, positions)
        loss = criterion(output, targets.view(targets.shape[0], -1), mask)
        # loss = criterion(output, targets.view(targets.shape[0], -1)) / images.shape[0]
        avg_loss = (loss.detach().cpu() + avg_loss * max(1, idx - 1)) / (idx)


        targets = targets.cpu().numpy()
        output = output.detach().cpu().view(images.shape[0], -1, 2).numpy()
        bs = targets.shape[0]
        for i in range(bs):
            gt_points[idx * bs + i] = (targets[i, :, 0], targets[i, :, 1])
            predicted_points["val"][idx * bs + i] = (output[i, :, 0], output[i, :, 1])
            if idx == len(dataloader):
                image = vis_keypoints(
                    image=(images[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8),
                    keypoints=output[i],
                    diameter=4,
                    path=os.path.join(output_path, f"val_{i}.png"),
                    return_image=True
                )

        if idx == len(dataloader):
            cols = 4
            raws = bs // cols
            num_ax = cols * raws
            fig, ax = plt.subplots(raws, cols, figsize=(30, 20))
            plt.title(f"val examples")
            for ii in range(num_ax):
                gt_points[idx * bs + i] = (targets[i, :, 0], targets[i, :, 1])
                predicted_points["val"][idx * bs + i] = (output[i, :, 0], output[i, :, 1])
                image = vis_keypoints(
                    image=(images[ii].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8),
                    keypoints=output[ii],
                    diameter=4,
                    path=None,
                    return_image=True
                    )
                ax[ii // cols, ii % cols].imshow(image)

            logger.report_matplotlib_figure(
                title=f"Epoch {epoch}",
                series=f"Example prediction",
                iteration=total_it,
                figure=fig,
                report_interactive=False,
            )


        if idx % log_frequency == 0 or idx == len(dataloader):
            line = f"Val batch {idx}/{len(dataloader)}: loss {loss}"
            pbar.write(line)


    fig = val_ced_calculate(
        predicted_points=predicted_points,
        gt_points=gt_points,
        output_path=output_path,
        normalization_type="bbox",
        # left_eye_idx="36,39",
        # right_eye_idx="42,45",
        epoch=epoch
    )
    logger.report_matplotlib_figure(
        title=f"Epoch {epoch}",
        series=f"CED@0.08",
        iteration=total_it,
        figure=fig,
        report_interactive=False,
    )

    line = f"Val loss: {avg_loss}"
    pbar_epoch.write(line)
    logger.report_scalar("Loss", "Val", iteration=total_it, value=avg_loss)
    return avg_loss.item()


@torch.no_grad()
def test(model, dataloader, criterion, device, output_path, task, logger):
    log_frequency = max(1, len(dataloader) // 10)

    model.to(device)
    model.eval()

    pbar = tqdm.tqdm(dataloader, ncols=150)
    avg_loss = torch.tensor(0.)

    gt_points = {}
    predicted_points = {"test": {}}
    for idx, batch in enumerate(pbar, 1):
        pbar.set_description(f"Val batch {idx}/{len(dataloader)}: ")
        images, targets, positions, _, mask = batch
        images = images.to(device)
        targets = targets.to(device)
        output = model(images) #.view(images.shape[0], -1, 2)
        mask = mask.to(device)
        positions = positions.to(device)
        # out_permuted = torch.gather(output, 1, positions)
        loss = criterion(output, targets.view(targets.shape[0], -1), mask)
        # loss = criterion(output, targets.view(targets.shape[0], -1)) / images.shape[0]
        avg_loss = (loss.detach().cpu() + avg_loss * max(1, idx - 1)) / (idx)


        targets = targets.cpu().numpy()
        output = output.detach().cpu().view(images.shape[0], -1, 2).numpy()
        bs = targets.shape[0]
        for i in range(bs):
            gt_points[idx * bs + i] = (targets[i, :, 0], targets[i, :, 1])
            predicted_points["test"][idx * bs + i] = (output[i, :, 0], output[i, :, 1])

        if idx % log_frequency == 0 or idx == len(dataloader):
            line = f"Test batch {idx}/{len(dataloader)}: loss {loss}"
            pbar.write(line)


    fig = val_ced_calculate(
        predicted_points=predicted_points,
        gt_points=gt_points,
        output_path=output_path,
        normalization_type="bbox",
        # left_eye_idx="36,39",
        # right_eye_idx="42,45"
    )
    logger.report_matplotlib_figure(
        title=f"Test",
        series=f"CED@0.08",
        iteration=0,
        figure=fig,
        report_interactive=False,
    )

    line = f"Test loss: {avg_loss}"
    print(line)
    logger.report_single_value(name="Test loss", value=avg_loss)