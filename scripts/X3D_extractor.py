import torch
from pytorchvideo.models import x3d
from torch.utils.data import DataLoader, Dataset
from torch import nn
from argparse import ArgumentParser
import pathlib
from tqdm.auto import tqdm
from npy_append_array import NpyAppendArray
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    UniformTemporalSubsample, Normalize, ShortSideScale)
from torchvision.transforms._transforms_video import ToTensorVideo
import os

# Dataset for X3D


class X3DDatasetRGB(Dataset):
    def __init__(self, path: pathlib.Path, block=64, transform=None):
        from utils import VideoStreamer
        self.block = block
        self.data = VideoStreamer(str(path))
        self.transform = transform

    def __len__(self):
        return len(self.data) // self.block

    def __getitem__(self, index: int) -> torch.Tensor:
        if index >= len(self):
            raise IndexError(f"index: {index} is out of bounds!")
        i = index * self.block
        j = i + self.block
        rgb_frames = self.data[i:j]
        if rgb_frames.shape[0] < self.block:
            from utils import pad_to_shape
            rgb_frames = pad_to_shape(rgb_frames, (self.block, 224, 224, 3))
        rgb_frames = torch.tensor(rgb_frames).permute(
            3, 0, 1, 2)  # Convert to C, T, H, W

        if self.transform:
            rgb_frames = self.transform(rgb_frames)

        return rgb_frames


# Initialize model and dataset
def init(args):
    transform = Compose([
        UniformTemporalSubsample(16),  # Subsample 16 frames
        ToTensorVideo(),  # Convert to tensor
        Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
        Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        ShortSideScale(256),  # Resize shorter side to 256
    ])

    dataset = X3DDatasetRGB(
        args.source_file, block=args.window_size, transform=transform)

    model = model = torch.hub.load(
        'facebookresearch/pytorchvideo', 'x3d_l', pretrained=True)
    model.head = nn.Identity()  # Remove classification head for feature extraction

    return model, dataset


# Save embeddings in chunks
def write_embedding_to_file_in_chunks(embedding, filename):
    try:
        with NpyAppendArray(filename) as npaa:
            npaa.append(embedding)
        print(f"Embedding successfully written to {filename.stem}.")
    except Exception as e:
        print(f"An error occurred while writing the embedding to file: {e}")


# Extract features and save to file
def extract_and_save(model, loader, device, filename, batch_size):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device, non_blocking=True)
            embeddings = model(data)
            write_embedding_to_file_in_chunks(
                embeddings.cpu().numpy(), filename)


# Main function
def main():
    parser = ArgumentParser()
    parser.add_argument("source_file", type=pathlib.Path,
                        help="Path to source video files")
    parser.add_argument("dest_file", type=pathlib.Path,
                        help="Destination file for embeddings")
    parser.add_argument("-w", "--window_size", type=int,
                        default=64, help="Number of frames per video block")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=16, help="Batch size for DataLoader")
    parser.add_argument("-nw", "--num_workers", type=int,
                        default=4, help="Number of workers for DataLoader")
    args = parser.parse_args()

    result_file = args.dest_file.parent / \
        f"{args.dest_file.stem}_x3dl{args.dest_file.suffix}"

    if result_file.exists():
        print(f"Result file {
              result_file} already exists. Skipping computation.")
        return

    model, dataset = init(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    extract_and_save(
        model,
        loader,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        result_file,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
