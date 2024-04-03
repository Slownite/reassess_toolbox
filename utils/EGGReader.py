import mne

class EEGReader:
    def __init__(self, edf_path):
        """
        Initialize an EEGReaderMNE object.

        Args:
            edf_path (str): Path to the EDF file.
        """
        self.edf_path = edf_path
        self.raw = mne.io.read_raw_edf(edf_path, preload=False)
        self.sampling_freq = self.raw.info['sfreq']
        self.num_channels = len(self.raw.ch_names)
        self.total_samples = self.raw.n_times
        self.annotations = self.raw.annotations

    def get_data(self, start_sample=0, end_sample=None):
        """
        Get EEG data within a specific range of samples.

        Args:
            start_sample (int): Starting index for samples to retrieve.
            end_sample (int, optional): Ending index for samples to retrieve.
                                 
        Returns:
            np.ndarray: The EEG data within the specified range of samples.
        """
        return self.raw.get_data(start=start_sample, stop=end_sample)

    def get_annotations(self, start_time=0, end_time=None):
        """
        Get annotations within a specific time range.

        Args:
            start_time (float): Start time of the range in seconds.
            end_time (float, optional): End time of the range in seconds.
                                         If None, all annotations after start_time will be returned.

        Returns:
            list: List of annotations within the specified time range.
        """
        if end_time is None:
            end_time = self.raw.times[-1]  # Use the end of recording if end_time is not specified
        start_index, _ = self.get_annotation_position(start_time)
        end_index, _ = self.get_annotation_position(end_time)
        return self.annotations.description[start_index:end_index]

    def get_annotation_position(self, time_point):
        """
        Get the position of a time point within the annotations.

        Args:
            time_point (float): The time point to search for in seconds.

        Returns:
            tuple: Position of the time point in the annotations (index, offset).
        """
        onset_samples = self.annotations.onset * self.sampling_freq
        for i, onset_sample in enumerate(onset_samples):
            if time_point < onset_sample / self.sampling_freq:
                if i > 0:
                    return i, time_point - onset_samples[i - 1] / self.sampling_freq
                else:
                    return 0, time_point
        return len(self.annotations), time_point - onset_samples[-1] / self.sampling_freq

    def get_sampling_freq(self):
        """
        Get the sampling frequency of the EEG data.

        Returns:
            float: Sampling frequency in Hz.
        """
        return self.sampling_freq

    def __len__(self):
        """
        Get the total number of samples in the EEG recording.

        Returns:
            int: Total number of samples.
        """
        return self.total_samples


class EEGStream:
    def __init__(self, edf_path, chunk_duration=1.0):
        """
        Initialize an EEGStreamMNE object for streaming EEG data in chunks.

        Args:
            edf_path (str): Path to the EDF file.
            chunk_duration (float): Duration of each chunk in seconds.
        """
        self.edf_path = edf_path
        self.chunk_duration = chunk_duration
        self.raw = mne.io.read_raw_edf(edf_path, preload=False)
        self.sampling_freq = self.raw.info['sfreq']
        self.num_channels = len(self.raw.ch_names)
        self.total_samples = self.raw.n_times
        self.annotations = self.raw.annotations
        self.duration = self.total_samples / self.sampling_freq

    def stream(self):
        """
        Stream EEG data in chunks.

        Yields:
            np.ndarray: The next chunk of EEG data.
        """
        start_time = 0
        while start_time < self.duration:
            end_time = start_time + self.chunk_duration
            if end_time > self.duration:
                end_time = self.duration

            # Fetching the chunk of data
            start_sample = int(start_time * self.sampling_freq)
            end_sample = int(end_time * self.sampling_freq)
            data_chunk = self.raw.get_data(start=start_sample, stop=end_sample)

            yield data_chunk

            start_time += self.chunk_duration

    def get_annotations_in_chunk(self, chunk_start, chunk_end):
        """
        Get annotations that fall within a specific chunk of data.

        Args:
            chunk_start (float): Start time of the chunk in seconds.
            chunk_end (float): End time of the chunk in seconds.

        Returns:
            list: List of annotations within the specified chunk.
        """
        annotations_in_chunk = [annot for annot in self.annotations if chunk_start <= annot['onset'] < chunk_end]
        return annotations_in_chunk
