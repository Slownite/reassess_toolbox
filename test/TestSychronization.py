import cv2
import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.EGGVideoSynchronizer import EEGVideoSynchronizer

def main():
    # Replace these with the actual paths to your video and EEG directory
    directory_path = 'C:/Users/hp/OneDrive - Institut National de Statistique et d\'Economie Appliquee/Bureau/REASSEASS/data/'
    
    synchronizer = EEGVideoSynchronizer(directory_path, video_frame_rate=20, eeg_sampling_rate=256, block_size_frames=10)

    
    # Get the first block of synchronized data
    block_data = synchronizer.get_block(0)
    
    # Access the synchronized video frames, EEG data, and annotations from the first block
    video_frames, video_flows, eeg_data, annotations = block_data
    
    print(f"Retrieved {len(video_frames)} video frames.")
    print(f"Retrieved {len(video_flows)} optical flow frames.")

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
