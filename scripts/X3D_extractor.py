import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from argparse import ArgumentParser
import pathlib
from tqdm.auto import tqdm
from npy_append_array import NpyAppendArray
from torchvision.transforms import Compose
import glob


class X3DDatasetRGB(Dataset):
    def __init__(self, video_path, block=64, transform=None):
        from utils import VideoStreamer
        self.block = block
        self.data = VideoStreamer(str(video_path))
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

        return rgb_frames.float()


# Initialize model and dataset
def init(video_path, window_size):
    transform = Compose([
        # Add any necessary transforms here
    ])

    dataset = X3DDatasetRGB(video_path, block=window_size, transform=transform)

    model = torch.hub.load(
        'facebookresearch/pytorchvideo', 'x3d_m', pretrained=True
    )
    model.head = nn.Identity()  # Remove classification head for feature extraction

    return model, dataset


# Save embeddings in chunks
def write_embedding_to_file_in_chunks(embedding, filename):
    try:
        with NpyAppendArray(filename) as npaa:
            npaa.append(embedding)
        print(f"Embedding successfully written to {filename}.")
    except Exception as e:
        print(f"An error occurred while writing the embedding to file: {e}")


# Extract features and save to file
def extract_and_save(model, loader, device, filename):
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
    parser.add_argument("source_dir", type=pathlib.Path,
                        help="Path to source directory containing videos")
    parser.add_argument("-w", "--window_size", type=int,
                        default=64, help="Number of frames per video block")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=16, help="Batch size for DataLoader")
    parser.add_argument("-nw", "--num_workers", type=int,
                        default=0, help="Number of workers for DataLoader")
    args = parser.parse_args()

    video_files = glob.glob(str(args.source_dir / "**/*.mp4"), recursive=True)
    if not video_files:
        print("No video files found in the specified directory.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for video_file in video_files:
        video_path = pathlib.Path(video_file)
        output_file = video_path.parent / f"{video_path.stem}_x3dm.npy"

        if output_file.exists():
            print(
                f"Output file {output_file} already exists. Skipping {video_file}.")
            continue

        print(f"Processing {video_file}...")

        model, dataset = init(video_path, args.window_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        extract_and_save(model, loader, device, output_file)


if __name__ == "__main__":
    main()
