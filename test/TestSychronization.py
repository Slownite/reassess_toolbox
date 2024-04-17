
import cv2
import mne
import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.EGGVideoSynchronizer import EEGVideoSynchronizer

def main():
    # Replace these with the actual paths to your video and EEG files
    video1_path = "C:/Users/hp/OneDrive - Institut National de Statistique et d'Economie Appliquee/Bureau/REASSEASS/data/video1.ASF"
    eeg_path = "C:/Users/hp/OneDrive - Institut National de Statistique et d'Economie Appliquee/Bureau/REASSEASS/data/231115A-B.edf"
    
    # Assuming a video frame rate of 30 frames per second and EEG sampling rate of 256 samples per second
    video_frame_rate = 30
    eeg_sampling_rate = 256
    
    # Set the block size to 10 video frames
    block_size_frames = 1000
    
    # Initialize the synchronizer
    synchronizer = EEGVideoSynchronizer(
        video_paths=[video1_path],
        eeg_path=eeg_path,
        video_frame_rate=video_frame_rate,
        eeg_sampling_rate=eeg_sampling_rate,
        block_size_frames=block_size_frames
    )
    
    # Get the first block of synchronized data
    block_data = synchronizer.get_block(0)
    
    # Access the synchronized video frames, EEG data, and annotations from the first block
    video_frames = block_data['video_frames'][0]  # Accessing the first video's frames
    eeg_data = block_data['eeg_data']
    annotations = block_data['annotations']
    
    # Print some information about the fetched data
    print(f"Retrieved {len(video_frames)} video frames.")
  

    print(f"Retrieved EEG data shape: {eeg_data.shape}")
    print(eeg_data)

    print(f"Retrieved {len(annotations)} annotations.")
    print(annotations)
    
    # Display the first video frame of the first block
    cv2.imshow("First frame of the first block", video_frames[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
