from typing import Any
from typing import Optional
import cv2 as cv
import numpy as np


def get_position(value: int, acc_arr: list[int]) -> tuple[int, int]:
    """
    Get the position of a value in an accumulated array.

    Args:
        value (int): The value to search for.
        acc_arr (list[int]): The accumulated array.

    Returns:
        tuple[int, int]: The position of the value in the accumulated array.
    """
    for i, acc_value in enumerate(acc_arr):
        if value < acc_value:
            return i, value - acc_arr[i - 1]
    return len(acc_arr), value - acc_arr[-2]


class VideoCaptureWrapper:
    def __init__(self, filename, shape=None):
        """
        Initialize a VideoCaptureWrapper object.

        Args:
            filename: The filename of the video.
            shape: The desired shape of the video frames (optional).
        """
        self.cap = cv.VideoCapture(filename)
        self.total_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.shape = shape

    def __getitem__(self, index):
        """
        Get a frame or frames from the video.

        Args:
            index: The index or slice of indices to retrieve.

        Returns:
            np.ndarray: The frame or frames from the video.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(self.total_frames)
            return np.array([self.get_frame(i) for i in range(start, stop, step)])
        elif isinstance(index, int):
            if index < 0:
                index += self.total_frames
            if index < 0 or index >= self.total_frames:
                raise IndexError("Frame number is out of range")
            return self.get_frame(index)
        else:
            raise TypeError("Invalid argument type")

    def get_frame(self, frame_number):
        """
        Get a specific frame from the video.

        Args:
            frame_number: The frame number to retrieve.

        Returns:
            np.ndarray: The frame from the video.
        """
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Could not read frame from video")
        if self.shape is None:
            return frame
        frame = cv.resize(frame, (self.shape[0], self.shape[1]))
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def __len__(self):
        """
        Get the total number of frames in the video.

        Returns:
            int: The total number of frames in the video.
        """
        return self.total_frames

    def __del__(self):
        """
        Release the video capture object.
        """
        self.cap.release()


def add_video_capture(
    paths: tuple[str, ...], videos: list[Any], cumul_length: list[int], shape=None
) -> tuple[list[Any], list[int], list[str]]:
    """
    Add video capture objects to the list of videos.

    Args:
        paths (tuple[str, ...]): The paths of the videos.
        videos (list[Any]): The list of videos.
        cumul_length (list[int]): The cumulative length of the videos.
        shape: The desired shape of the video frames (optional).

    Returns:
        tuple[list[Any], list[int]]: The updated list of videos and cumulative length.
    """
    files = []
    for _, file in enumerate(paths):
        vid = VideoCaptureWrapper(file, shape=shape)
        num_frames = int(len(vid))
        new_cumul_length = cumul_length[-1] + num_frames
        videos.append(vid)
        cumul_length.append(new_cumul_length)
        files.append(file)
    return videos, cumul_length, files


class VideoStreamer:
    def __init__(
        self,
        *paths: tuple[str, ...],
        batch: int = 1,
        shape: Optional[tuple[int, int]] = None
    ) -> None:
        """
        Initialize a VideoStreamer object.

        Args:
            paths (tuple[str, ...]): The paths of the videos.
            batch (int): The batch size (optional).
            shape (tuple[int, int]): The desired shape of the video frames (optional).
        """
        self.batch = batch
        self.shape = shape
        self.videos, self.cumul_length, self.files = add_video_capture(
            paths, [], [0], self.shape
        )

    def append(self, *paths: tuple[str, ...]):
        """
        Append video capture objects to the list of videos.

        Args:
            paths (tuple[str, ...]): The paths of the videos.
        """
        videos, cumul_length = add_video_capture(
            paths, self.videos, self.cumul_length, self.shape
        )
        self.videos = videos
        self.cumul_length = cumul_length

    def __len__(self):
        """
        Get the total number of frames in the video stream.

        Returns:
            int: The total number of frames in the video stream.
        """
        return self.cumul_length[-1]

    def __getitem__(self, value):
        """
        Get a frame or frames from the video stream.

        Args:
            value: The index or slice of indices to retrieve.

        Returns:
            np.ndarray: The frame or frames from the video stream.
        """
        if isinstance(value, slice):
            start, stop, step = value.indices(len(self))
            array = []
            files = []
            for i in range(start, stop, step):
                item, file = self.get_item(i)
                array.append(item)
                files.append(file)
            return np.array(array), files

        elif isinstance(value, int):
            if value >= len(self) or value < 0:
                raise IndexError("Index out of bounds")
            return self.get_item(value)
        else:
            raise TypeError("Invalid argument type")

    def get_item(self, value):
        """
        Get a specific frame from the video stream.

        Args:
            value: The index of the frame to retrieve.

        Returns:
            np.ndarray: The frame from the video stream.
        """
        video_index, index = get_position(value, self.cumul_length)
        return self.videos[video_index - 1][index], self.files[video_index - 1]


if __name__ == "__main__":
    video_path = "C:/Users/hp/OneDrive - Institut National de Statistique et d'Economie Appliquee/Bureau/REASSEASS/data/video1.ASF"

    video = VideoCaptureWrapper(video_path)
    print(video.__len__())
