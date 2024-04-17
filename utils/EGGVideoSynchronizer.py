import numpy as np
import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.EGGReader import EEGReader , EEGStream
from utils.stream import VideoCaptureWrapper


# class ClassSynchronization:
#     def __init__(self, video1_path, video2_path, eeg_path, block_size=10):
#         """
#         Initialize a ClassSynchronization object.

#         Args:
#             video1_path (str): Path to the first video file.
#             video2_path (str): Path to the second video file.
#             eeg_path (str): Path to the EEG file.
#             block_size (int): Number of frames per block.
#         """
#         self.video1 = VideoCaptureWrapper(video1_path)
#         self.video2 = VideoCaptureWrapper(video2_path)
#         self.eeg = EEGReader(eeg_path)
#         self.block_size = block_size
#         self.eeg_freq = self.eeg.get_sampling_freq()

#     def __getitem__(self, idx):
#         """
#         Get a block of synchronized video frames, sound, EEG data, and annotations.

#         Args:
#             idx (int): Block index.

#         Returns:
#             Tuple containing arrays of video frames, sound, annotations, and EEG sampling frequency.
#         """
#         # Calculate the range of frames for this block
#         start_frame = idx * self.block_size
#         end_frame = start_frame + self.block_size

#         # Extract video frames
#         frames_video1 = self.video1[start_frame:end_frame]
#         frames_video2 = self.video2[start_frame:end_frame]

#         # For simplicity, assuming array_image combines frames from both videos
#         array_image = np.concatenate((frames_video1, frames_video2), axis=0)

#         # Placeholder for array_sound extraction logic
#         array_sound = np.zeros((self.block_size,))  # Dummy sound data

#         # Synchronize and extract EEG data corresponding to the video frames
#         # Assuming the video frame rate and EEG sampling frequency are known
#         video_frame_rate = 30  # Example frame rate
#         start_sample = int(start_frame * (self.eeg_freq / video_frame_rate))
#         end_sample = int(end_frame * (self.eeg_freq / video_frame_rate))
#         array_eeg = self.eeg.get_data(start_sample, end_sample)

#         # Placeholder for annotation or label extraction logic
#         array_label = np.zeros((self.block_size,))  # Dummy labels

#         return array_image, array_sound, array_label, self.eeg_freq


# if __name__ == '__main__':

#     video1= 'C:/Users/hp/OneDrive - Institut National de Statistique et d\'Economie Appliquee/Bureau/REASSEASS/data/video1.ASF'
#     video2= 'C:/Users/hp/OneDrive - Institut National de Statistique et d\'Economie Appliquee/Bureau/REASSEASS/data/video2.ASF'
#     eeg= 'C:/Users/hp/OneDrive - Institut National de Statistique et d\'Economie Appliquee/Bureau/REASSEASS/data/231115A-B.edf'
#     block=5
#     data = ClassSynchronization(video1, video2, eeg, block_size=block)

#     array_image, array_sound, array_label, frequence_EEG = data[0]
#     print(array_image.shape[0]) # donnera le nombre d'image par block


#     # Test fetching a single frame directly from the VideoCaptureWrapper
#     test_video = VideoCaptureWrapper(video1)
#     test_frame = test_video.get_frame(0)  # Assuming get_frame method exists
#     print(test_frame.shape)  # Should print the dimensions of the frame

class EEGVideoSynchronizer:
    def __init__(self, video_paths, eeg_path, video_frame_rate, eeg_sampling_rate, block_size_frames):
        """
        Initializes the EEGVideoSynchronizer object.
        
        Args:
            video_paths (list[str]): List of paths to video files.
            eeg_path (str): Path to the EEG file.
            video_frame_rate (float): Frame rate of the video in frames per second.
            eeg_sampling_rate (float): Sampling rate of the EEG data in samples per second.
            block_size_frames (int): Number of video frames per block.
        """
        # Initialize video streamers for each video path
        self.videos = [VideoCaptureWrapper(path) for path in video_paths]
        self.eeg = EEGReader(eeg_path)
        self.video_frame_rate = video_frame_rate
        self.eeg_sampling_rate = eeg_sampling_rate
        self.block_size_frames = block_size_frames
        self.samples_per_frame = self.eeg_sampling_rate / self.video_frame_rate
        
    def get_block(self, block_idx):
        """
        Gets synchronized video frames, EEG data, and annotations for the specified block index.
        
        Args:
            block_idx (int): Index of the block to retrieve.
            
        Returns:
            dict: A dictionary containing synchronized video frames, EEG data, and annotations.
        """
        start_frame = block_idx * self.block_size_frames
        end_frame = start_frame + self.block_size_frames
        
        # Fetch video frames for each video streamer
        video_frames = [video[start_frame:end_frame] for video in self.videos]
        
        # Calculate corresponding EEG sample range
        start_sample = int(start_frame * self.samples_per_frame)
        end_sample = int(end_frame * self.samples_per_frame)
        
        # Fetch EEG data for the corresponding sample range
        eeg_data = self.eeg.get_data(start_sample, end_sample)
        
        # Fetch annotations for the corresponding time range
        start_time = start_frame / self.video_frame_rate
        end_time = end_frame / self.video_frame_rate
        annotations = self.eeg.get_annotations(start_time, end_time)
        
        return {
            'video_frames': video_frames,
            'eeg_data': eeg_data,
            'annotations': annotations,
            'eeg_sampling_rate': self.eeg_sampling_rate
        }

if __name__ == '__main__':
    video1= 'C:/Users/hp/OneDrive - Institut National de Statistique et d\'Economie Appliquee/Bureau/REASSEASS/data/video1.ASF'
    video2= 'C:/Users/hp/OneDrive - Institut National de Statistique et d\'Economie Appliquee/Bureau/REASSEASS/data/video2.ASF'
    eeg= 'C:/Users/hp/OneDrive - Institut National de Statistique et d\'Economie Appliquee/Bureau/REASSEASS/data/231115A-B.edf'
    synchronizer = EEGVideoSynchronizer(video_path=video1, eeg_path=eeg, block_size_frames=10)

    # Get the first synchronized block of video frames and EEG data
    video_block, eeg_data = synchronizer[0]

    # Process or analyze the synchronized data here...
