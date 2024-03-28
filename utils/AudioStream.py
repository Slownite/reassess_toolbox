import librosa
import numpy as np
import soundfile as sf 

class AudioWrapper:
    def __init__(self, filepath, sr=None):
        """
        Initialize an AudioWrapper object using librosa.

        Args:
            filepath (str): Path to the audio file.
            sr (int, optional): Sampling rate to use for loading the audio.
                                Librosa's default sampling rate (22050) is used.
        """
        self.filepath = filepath
        self.sr = sr
        self.audio, self.sr = librosa.load(filepath, sr=sr)
        self.total_samples = len(self.audio)

    def __len__(self):
        """
        Get the total number of audio samples.

        Returns:
            int: The total number of audio samples.
        """
    
        return self.total_samples
    
    def get_samples(self, start_sample=0, end_sample=None):
        """
        Get a specific range of samples from the audio file.

        Args:
            start_sample (int): Starting index for samples to retrieve.
            end_sample (int, optional): Ending index for samples to retrieve.
                                        If None, all samples until the end of the file are retrieved.

        Returns:
            np.ndarray: The requested range of audio samples.
        """
        if end_sample is None or end_sample > self.total_samples:
            end_sample = self.total_samples
        return self.audio[start_sample:end_sample]
    
    def get_sample_rate(self):
        """
            Get the sampling rate of the audio file.

            Returns:
                int: The sampling rate of the audio file.
            """
        #librosa.get_samplerate(path)
        return self.sr
    
    def get_duration(self):
        """
        Get the duration of the audio file in seconds.

        Returns:
            float: The duration of the audio file in seconds.
                    We can also use the get_duration function of librosa.

        """
        return self.total_samples / self.sr

class AudioStream:
    def __init__(self, filepath, batch_size=1024, sr=None, frame_length=2048, hop_length=1024):
        """
        Initialize an AudioStream object for streaming audio data in blocks.

        Args:
            filepath (str): Path to the audio file.
            batch_size (int): Number of frames to load per batch.
            sr (int, optional): Sampling rate to use for loading the audio. If None, librosa's default is used.
            frame_length (int): The length of the frame/window in samples for each block.
            hop_length (int): The number of samples to advance between frames.
        """
        self.filepath = filepath
        self.batch_size = batch_size
        self.frame_length = frame_length
        self.hop_length = hop_length

        # Access the metadata of the audio file to get sample rate and total samples
        with sf.SoundFile(filepath) as f:
            self.sr = sr if sr is not None else f.samplerate
            self.total_samples = len(f)

    def stream(self):
        """
        Stream audio file in blocks.

        Yields:
            np.ndarray: The next block of audio samples.
        """
        # Use librosa.stream with the specified frame_length and hop_length
        return librosa.stream(self.filepath, 
                              block_length=self.batch_size, 
                              frame_length=self.frame_length, 
                              hop_length=self.hop_length)

    def get_samples(self, start_sample=0, end_sample=None):
        """
        Get a specific range of samples from the audio file in a streaming context.

        Args:
            start_sample (int): Starting index for samples to retrieve.
            end_sample (int, optional): Ending index for samples to retrieve. If None, all samples until the end of the file are retrieved.

        Returns:
            np.ndarray: The requested range of audio samples.
        """
        collected_samples = []
        current_sample = 0

        for block in self.stream():
            block_start_sample = current_sample
            block_end_sample = current_sample + len(block)
            current_sample += len(block)

            # Skip blocks before the start_sample
            if block_end_sample <= start_sample:
                continue

            # If the current block contains the start_sample
            if start_sample < block_end_sample:
                start_index = max(0, start_sample - block_start_sample)
                end_index = None if end_sample is None else end_sample - block_start_sample

                # If the current block reaches or exceeds the end_sample
                if end_index is not None and end_index <= len(block):
                    collected_samples.append(block[start_index:end_index])
                    break
                else:
                    collected_samples.append(block[start_index:])

            # Stop if we have reached the end_sample
            if end_sample is not None and current_sample >= end_sample:
                break

        # Concatenate collected samples into a single array to return
        return np.concatenate(collected_samples, axis=0) if collected_samples else np.array([])
    

    def get_samples_with_soundfile(self, start_sample=0, end_sample=None):

            with sf.SoundFile(self.filepath, 'r') as f:
                # Seek to the start sample
                f.seek(start_sample)
                # Calculate the number of samples to read
                num_samples = end_sample - start_sample if end_sample else self.total_samples - start_sample
                # Read and return the specified range of samples
                samples = f.read(num_samples)
                return samples
    
    def __len__(self):
        """
        Get the total number of audio samples.

        Returns:
            int: The total number of audio samples.
        """
        return self.total_samples
    
    def get_duration(self):
        """
        Get the duration of the audio file in seconds.

        Returns:
            float: The duration of the audio file in seconds.
        """
        return self.total_samples / self.sr