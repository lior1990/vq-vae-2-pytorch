import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, utils

from tqdm import tqdm

from dataset import CustomDataset
from discriminator import NLayerDiscriminator, calc_gradient_penalty
from vqvae import VQVAE
from scheduler import CycleScheduler


def train(epoch, loader, model, discriminator, optimizer, optimizer_d, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    for i, img in enumerate(loader):
        img = img.to(device)

        # train D with real
        logits_real = discriminator(img)

        # train D with fake
        fake, latent_loss = model(img)
        logits_fake = discriminator(fake.detach())

        loss_real = -torch.mean(logits_real)
        loss_fake = torch.mean(logits_fake)

        gradient_penalty = calc_gradient_penalty(discriminator, img, fake.detach(), device)

        d_loss = loss_real + loss_fake + gradient_penalty

        discriminator.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # train G
        recon_loss = criterion(fake, img)
        latent_loss = latent_loss.mean()

        logits_fake = discriminator(fake)
        g_loss = -torch.mean(logits_fake)
        loss = recon_loss + latent_loss_weight * latent_loss + g_loss
        model.zero_grad()
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum = recon_loss.item() * img.shape[0]
        mse_n = img.shape[0]

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                f"g_loss: {g_loss.item():.3f}; d_loss: {d_loss.item():.3f}; "
                f"lr: {lr:.5f}"
            )
        )

        if i % 100 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f"sample/{args.name}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = CustomDataset(args.path, transform)
    loader = DataLoader(
        dataset, batch_size=128 // args.n_gpu, shuffle=True, num_workers=0, drop_last=False,
    )

    model = VQVAE(n_embed=args.n_embeddings).to(device)
    discriminator = NLayerDiscriminator().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        os.makedirs(f"checkpoint/{args.name}", exist_ok=True)
        os.makedirs(f"sample/{args.name}", exist_ok=True)

        train(i, loader, model, discriminator, optimizer, optimizer_d, scheduler, device)
        torch.save(model.state_dict(), f"checkpoint/{args.name}/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("path", type=str)
    parser.add_argument("--n_embeddings", type=int, default=512)

    args = parser.parse_args()

    print(args)

    main(args)
