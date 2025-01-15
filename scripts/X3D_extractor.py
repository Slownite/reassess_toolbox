import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from argparse import ArgumentParser
import pathlib
from tqdm.auto import tqdm
from npy_append_array import NpyAppendArray
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.functional import normalize, resize


class UniformTemporalSubsample:
    def __init__(self, num_samples):
        """
        Uniformly subsamples frames from a video.
        Args:
            num_samples (int): Number of frames to sample.
        """
        self.num_samples = num_samples

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video tensor of shape (T, C, H, W).
        Returns:
            torch.Tensor: Subsampled video of shape (num_samples, C, H, W).
        """
        t = video.shape[0]
        indices = torch.linspace(0, t - 1, self.num_samples).long()
        return video[indices]


class Normalize:
    def __init__(self, mean, std):
        """
        Normalizes each frame of a video.
        Args:
            mean (list): Mean values for each channel.
            std (list): Standard deviation values for each channel.
        """
        self.mean = mean
        self.std = std

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video tensor of shape (T, C, H, W).
        Returns:
            torch.Tensor: Normalized video of shape (T, C, H, W).
        """
        return torch.stack([normalize(frame, self.mean, self.std) for frame in video])


class ShortSideScale:
    def __init__(self, target_size):
        """
        Resizes the video so the shorter side matches the target size.
        Args:
            target_size (int): Target size for the shorter side.
        """
        self.target_size = target_size

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video tensor of shape (T, C, H, W).
        Returns:
            torch.Tensor: Resized video of shape (T, C, new_H, new_W).
        """
        _, _, h, w = video.shape
        if h < w:
            new_h, new_w = self.target_size, int(w * self.target_size / h)
        else:
            new_h, new_w = int(h * self.target_size / w), self.target_size

        return torch.stack([resize(frame, (new_h, new_w)) for frame in video])


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
            print("load on gpu")
            data = data.to(device, non_blocking=True)
            print("load done")
            print("inference start")
            embeddings = model(data)
            print("inference done")
            print("saving time")
            write_embedding_to_file_in_chunks(
                embeddings.cpu().numpy(), filename)
            print("saving done")


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
