import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ImageFileDataset, CodeRow, CustomDataset
from vqvae import VQVAE
from torchvision import transforms, utils


def generate_samples(model, loader, device, args):
    pbar = tqdm(loader)

    for i, img in enumerate(pbar):
        img = img.to(device)

        with torch.no_grad():
            out, _ = model(img)

        utils.save_image(
            torch.cat([img, out], 0),
            f"generate_samples/{args.name}/{str(i).zfill(5)}.png",
            nrow=img.shape[0],
            normalize=True,
            range=(-1, 1),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--name', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = CustomDataset(args.path, transform, data_rep=1)
    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=False)

    model = VQVAE(n_embed=args.n_embeddings)
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()

    generate_samples(model, loader, device, args)


if __name__ == '__main__':
    main()

