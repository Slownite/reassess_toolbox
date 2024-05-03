import librosa
import numpy as np
import soundfile as sf 
import moviepy.editor as mpy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class AudioEager:
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
    def __init__(self, *paths):
        """ Initialize AudioStreamer with multiple audio file paths. """
        self.files = []
        self.cumul_samples = [0]  # Cumulative samples for positioning across files
        for path in paths:
            with sf.SoundFile(path) as f:
                self.files.append({
                    'path': path,
                    'total_samples': f.frames,
                    'samplerate': f.samplerate
                })
                self.cumul_samples.append(self.cumul_samples[-1] + f.frames)

    def __len__(self):
        """ Returns the total number of samples across all files. """
        return self.cumul_samples[-1]

    def __getitem__(self, index):
        """ Get audio samples using integer index or slice for chunks. """
        if isinstance(index, slice):
            start, stop, _ = index.indices(self.cumul_samples[-1])
            return self.get_samples(start, stop)
        elif isinstance(index, int):
            if index < 0 or index >= self.cumul_samples[-1]:
                raise IndexError("Sample index out of bounds")
            return self.get_samples(index, index + 1)
        else:
            raise TypeError("Invalid index type")

    def get_samples(self, start_sample, end_sample):
        """ Retrieve samples from files handling transitions between files. """
        samples = []
        file_index, local_start = self.get_position(start_sample)

        while start_sample < end_sample and file_index < len(self.files):
            file_info = self.files[file_index]
            local_end = min(file_info['total_samples'], local_start + (end_sample - start_sample))
            with sf.SoundFile(file_info['path'], 'r') as f:
                f.seek(local_start)
                samples.append(f.read(local_end - local_start))

            start_sample += local_end - local_start
            file_index += 1
            local_start = 0

        return np.concatenate(samples) if samples else np.array([])
    
    
    def get_position(self, sample):
        """ Find the file index and local sample index for a global sample index. """
        for i in range(1, len(self.cumul_samples)):
            if sample < self.cumul_samples[i]:
                return i - 1, sample - self.cumul_samples[i - 1]
        return -1, -1

    def get_duration(self):
        """
        Get the total duration of all audio files in seconds.

        Returns:
            float: The total duration of all audio files in seconds.
        """
        total_duration = 0
        for file in self.files:
            total_duration += file['total_samples'] / file['samplerate']
        return total_duration
