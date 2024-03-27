import librosa
import numpy as np

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
    def __init__(self, filepath, batch_size=1024, sr=None):
        """
        Initialize an AudioStream object for streaming audio data in blocks.

        Args:
            filepath (str): Path to the audio file.
            batch_size (int): Length of each block to stream, in frames.
            sr (int, optional): Sampling rate to use for loading the audio.
                                Librosa's default sampling rate (22050) is used IF NOT.
        """
        self.filepath = filepath
        self.batch_size = batch_size
        self.sr = sr

    def stream(self):
        """
        Stream audio file in blocks.

        Yields:
            np.ndarray: The next block of audio samples.
        """
        for block in librosa.stream(self.filepath, 
                                    batch_size=self.batch_size, 
                                    frame_length=2048, 
                                    hop_length=1024, 
                                    sr=self.sr):
            yield block
