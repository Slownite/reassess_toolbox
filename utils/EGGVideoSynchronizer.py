import shutil
import subprocess
import numpy as np
import sys 
import os 
from pyedflib import highlevel

import re
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.EGGReader import EEGReader , EEGStream
from utils.stream import VideoStreamer

from scripts.optical_flow_converter import process_all_videos, video_to_optical_flow

# def concatenate_files(file_paths, output_filename, directory):
#     """ Concatenate multiple files into a single file. """
#     output_path = Path(directory) / output_filename
#     if not output_path.exists():
#         if str(file_paths[0]).endswith('.edf'):
#             # Concatenate EDF files
#             signals, signal_headers, header = highlevel.read_edf(str(file_paths[0]))
#             for file in file_paths[1:]:
#                 new_signals, _, _ = highlevel.read_edf(str(file))
#                 signals = np.concatenate((signals, new_signals), axis=1)
#             highlevel.write_edf(str(output_path), signals, signal_headers, header)
#         else:
#             # Concatenate video files
#             inputs = ' '.join([f"file '{str(v)}'" for v in sorted(file_paths)])
#             command = f"ffmpeg -y -f concat -safe 0 -i \"{inputs}\" -c copy {output_path}"
#             os.system(command)
#     return output_path

class EEGVideoSynchronizer:
    
    def __init__(self, video_directory, eeg_directory, video_frame_rate, eeg_sampling_rate, block_size_frames, compute_method='tvl1'):
        """
        Initializes the EEGVideoSynchronizer object.
        
        Args:
            video_paths (list[str]): List of paths to video files.
            eeg_path (str): Path to the EEG file.
            video_frame_rate (float): Frame rate of the video in frames per second.
            eeg_sampling_rate (float): Sampling rate of the EEG data in samples per second.
            block_size_frames (int): Number of video frames per block.
            compute_method (str): Method for computing optical flow.
        """        
        self.video_frame_rate = video_frame_rate
        self.eeg_sampling_rate = eeg_sampling_rate
        self.block_size_frames = block_size_frames
        self.compute_method = compute_method

        # Initialize the video streamer with all video files found in the directory
        video_paths = [str(p) for p in Path(video_directory).glob("*.mp4")]
        self.video_streamer = VideoStreamer(*video_paths)
        self.eeg = EEGReader(self.find_files(eeg_directory, "*.edf")) 

        # # Setup paths for concatenated video and EEG files
        # self.video_path = self.find_files(video_directory, "*.mp4")
        # self.eeg_path = self.find_files(eeg_directory, "*.edf")
        
        # # Initialize readers and capture objects
        # self.video_capture = VideoStreamer(self.video_path)
        # self.eeg_reader = EEGReader(self.eeg_path)
    
    def find_files(self, directory, pattern):
        """ Find all files in a directory that match a certain pattern. """
        return list(Path(directory).glob(pattern))
    
        
    def get_block(self,block_idx,output_dir):
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
        #video_frames = self.video_capture.get_frames(start_frame, end_frame)

        video_frames = self.video_streamer[start_frame:end_frame]


        # Calculate corresponding EEG sample range
        # start_sample = int(start_frame * self.samples_per_frame)
        # end_sample = int(end_frame * self.samples_per_frame)
        start_sample = int(start_frame * self.eeg_samplioung_rate / self.video_frame_rate)
        end_sample = int(end_frame * self.eeg_sampling_rate / self.video_frame_rate)
        

        # Fetch EEG data for the corresponding sample range
        eeg_data = self.eeg.get_data(start_sample, end_sample)
         
        # Fetch annotations for the corresponding time range
        start_time = start_frame / self.video_frame_rate
        end_time = (start_frame + self.block_size_frames) / self.video_frame_rate
        annotations = self.eeg.get_annotations(start_time, end_time)


        # Compute optical flows and save them as a video
        optical_flow_video_path = video_to_optical_flow(
            video_frames, 
            output_dir, 
            compute_method=self.compute_method, 
            output_format='mp4'
        )

        return {
            'video_frames': video_frames,
            'optical_flows': optical_flow_video_path,
            'eeg_data': eeg_data,
            'annotations': annotations }

if __name__ == '__main__':
    video_dir= 'C:/Users/hp/OneDrive - Institut National de Statistique et d\'Economie Appliquee/Bureau/REASSEASS/data/'
    eeg= 'C:/Users/hp/OneDrive - Institut National de Statistique et d\'Economie Appliquee/Bureau/REASSEASS/data/231115A-B.edf'
    #video_block, eeg_data = synchronizer[0]
    
    #synchronizer = EEGVideoSynchronizer(video_path=video1, eeg_path=eeg, block_size_frames=10)
    synchronizer = EEGVideoSynchronizer(video_dir, eeg, video_frame_rate=20, eeg_sampling_rate=256, block_size_frames=10, compute_method='tvl1')

    # Get the first synchronized block of video frames and EEG data
    
    block_data = synchronizer.get_block(0, output_dir)
    print(block_data)

    # Process or analyze the synchronized data here...
