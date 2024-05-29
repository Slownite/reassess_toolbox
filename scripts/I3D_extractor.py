#!/usr/bin/env python3
import torch
from modules import I3D
from torch.utils.data import DataLoader, Dataset
from torch import nn
from utils import VideoStreamer, pad_to_shape
from argparse import ArgumentParser
import pathlib


class I3DDatasetRGB(Dataset):
    def __init__(self, path: pathlib.Path, block=66) -> None:
        self.block = block
        self.data = VideoStreamer(*map(str, list(path.glob("**/rgb_*.mp4"))))

    def __len__(self):
        return len(self.data) // self.block

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        i = index * self.block
        j = i + self.block
        rgb_frames, files = self.data[i:j]
        if rgb_frames.shape[0] < 66:
            rgb_frame = pad_to_shape(rgb_frames, (66, 224, 224, 3))
        assert rgb_frames.shape == (
            self.block,
            224,
            224,
            3,
        ), f"rgb_frames shape is {rgb_frames.shape}, should be ({self.block}, 224, 224, 3) "
        rgb_frames = torch.Tensor(rgb_frames).permute(3, 0, 1, 2)
        return (rgb_frames, files)


class I3DDatasetOF(Dataset):
    def __init__(self, path: pathlib.Path, block=66) -> None:
        self.block = block
        self.data = VideoStreamer(*map(str, (path.glob("**/flow_*.mp4"))))

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        i = index * self.block
        j = i + self.block
        compressed_flows, files = self.data[i:j]
        if compressed_flows.shape[0] < 66:
            compressed_flows = pad_to_shape(compressed_flows, (66, 224, 224, 3))
        assert compressed_flows.shape == (
            self.block,
            224,
            224,
            2,
        ), f"compressed_flows shape is {compressed_flow.shape}, should be ({self.block}, 224, 224, 2)"
        uncompressed_flow = torch.tensor(videos_frame_to_flow(flows)).permute(
            3, 0, 1, 2
        )
        uf_vectors, uf_depth, uf_width, uf_height = uncompressed_flow.shape
        stack_flows = uncompressed_flow.reshape(
            uf_depth * uf_vectors, uf_width, uf_width
        )
        return stack_flows, files


def init(args) -> [I3D, Dataset]:
    if args.model == "rgb":
        model = I3D(in_channels=3)
        dataset = I3DDatasetRGB(args.source_dir, block=args.window_size)
    elif args.model == "of":
        model = I3D(in_channels=2)
        dataset = I3DDatasetRGB(args.source_dir, block=args.window_size)
    else:
        raise ValueError("model type should be rgb or of")
    return model, dataset


def write_embedding_to_file_in_chunks(embedding, filename):
    """
    Writes an embedding vector to a file in chunks.

    Args:
    embedding_generator torch tensor: A torch tensor that contains chunks of the embedding vector.
    filename (pathlib.Path): The name of the file to write the embedding to.
    """
    try:
        # Open the file in write mode
        with open(filename.parent / f"{filename.stem}.pt", "ab") as file:
            torch.save(embedding, file)
        print(f"Embedding successfully written to {filename.stem}.pt")
    except Exception as e:
        print(f"An error occurred while writing the embedding to file: {e}")


def extract_and_save(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    directory: pathlib.Path,
    batch_size: int,
):
    """
    Extracts embeddings from the model using data from the loader and saves them to a file.

    Args:
    model (nn.Module): The model used to extract embeddings.
    loader (DataLoader): The data loader providing input data.
    device (torch.device): The device to run the model on.
    filename (Path): The name of the file to write the embeddings to.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data, files in loader:
            data = data.to(device, non_blocking=True)
            embeddings = model.extract(data)
            write_embedding_to_file_in_chunks(embeddings.cpu(), directory / files[0])


def main():
    parser = ArgumentParser()
    parser.add_argument("source_dir", type=pathlib.Path)
    parser.add_argument("dest_dir", type=pathlib.Path)
    parser.add_argument("-w", "--window_size", type=int, default=66)
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-m", "--model", type=str, default="rgb")
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    args = parser.parse_args()
    model, dataset = init(args)
    print("Initialization done")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("Loader created")
    print("start extracting")
    extract_and_save(
        model,
        loader,
        torch.device("cuda"),
        args.dest_dir,
        args.batch_size,
    )
    print("Done")


if __name__ == "__main__":
    main()
