#!/usr/bin/env python3
import torch
from modules import I3D
from torch.utils.data import DataLoader, Dataset
from torch import nn
from utils import VideoStreamer, pad_to_shape, videos_frame_to_flow
from argparse import ArgumentParser
import pathlib
from tqdm.auto import tqdm
from npy_append_array import NpyAppendArray


class I3DDatasetRGB(Dataset):
    def __init__(self, path: pathlib.Path, block=64) -> None:
        self.block = block
        self.data = VideoStreamer(str(path))

    def __len__(self):
        return len(self.data) // self.block

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        i = index * self.block
        j = i + self.block
        rgb_frames = self.data[i:j]
        if rgb_frames.shape[0] < 64:
            rgb_frames = pad_to_shape(rgb_frames, (64, 224, 224, 3))
        assert rgb_frames.shape == (
            self.block,
            224,
            224,
            3,
        ), f"rgb_frames shape is {rgb_frames.shape}, should be ({self.block}, 224, 224, 3) "
        rgb_frames = torch.Tensor(rgb_frames).permute(3, 0, 1, 2)
        return rgb_frames


class I3DDatasetOF(Dataset):
    def __init__(self, path: pathlib.Path, block=64) -> None:
        self.block = block
        self.data = VideoStreamer(str(path))

    def __len__(self):
        return len(self.data) // self.block

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise IndexError(f"index: {index} is out bound!")
        i = index * self.block
        j = i + self.block
        compressed_flows = self.data[i:j]
        if compressed_flows.shape[0] < 64:
            compressed_flows = pad_to_shape(
                compressed_flows, (64, 224, 224, 3))
        assert compressed_flows.shape == (
            self.block,
            224,
            224,
            3,
        ), f"compressed_flows shape is {compressed_flows.shape}, should be ({self.block}, 224, 224, 3)"
        uncompressed_flow = torch.tensor(videos_frame_to_flow(compressed_flows)).permute(
            3, 0, 1, 2
        )
        stack_flows = uncompressed_flow
        return stack_flows


def init(args) -> [I3D, Dataset]:
    if args.model == "rgb":
        model = I3D(in_channels=3, pretrained_weights=args.weights,
                    final_endpoint=args.layer)
        dataset = I3DDatasetRGB(args.source_file, block=args.window_size)
    elif args.model == "of":
        model = I3D(in_channels=2, pretrained_weights=args.weights,
                    final_endpoint=args.layer)
        dataset = I3DDatasetOF(args.source_file, block=args.window_size)
    else:
        raise ValueError("model type should be rgb or of")
    return model, dataset


def write_embedding_to_file_in_chunks(embedding, filename):
    """
    Writes an embedding vector to a file in chunks.

    Args:
    embedding_generator numpy array: A torch tensor that contains chunks of the embedding vector.
    filename (pathlib.Path): The name of the file to write the embedding to.
    """
    try:
        # Open the file in write mode
        with NpyAppendArray(filename) as npaa:
            npaa.append(embedding)
        print(f"Embedding successfully written to {filename.stem}.")
    except Exception as e:
        print(f"An error occurred while writing the embedding to file: {e}")


def extract_and_save(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    filename: pathlib.Path,
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
        for data in tqdm(loader):
            print("load data on gpu")
            data = data.to(device, non_blocking=True)
            print("end load")
            print("start extract")
            embeddings = model.extract(data)
            print("finish extract")
            print("write to file")
            write_embedding_to_file_in_chunks(
                embeddings.cpu().numpy(), filename)
            print("finish writing to file")


def main():
    parser = ArgumentParser()
    parser.add_argument("source_file", type=pathlib.Path)
    parser.add_argument("dest_file", type=pathlib.Path)
    parser.add_argument("-w", "--window_size", type=int, default=64)
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-m", "--model", type=str, default="rgb")
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("-l", '--layer', type=str, default="Mixed_5c")
    parser.add_argument("--weights", default=None, type=pathlib.Path)
    args = parser.parse_args()
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
        args.dest_file.parent /
        f"{args.dest_file.stem}_{args.layer}{args.dest_file.suffix}",
        args.batch_size,
    )


if __name__ == "__main__":
    main()
