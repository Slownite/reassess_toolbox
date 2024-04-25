import mne
import numpy as np

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


# class EEGStream:
#     def __init__(self, edf_path, chunk_duration=1.0):
#         """
#         Initialize an EEGStreamMNE object for streaming EEG data in chunks.

#         Args:
#             edf_path (str): Path to the EDF file.
#             chunk_duration (float): Duration of each chunk in seconds.
#         """
#         self.edf_path = edf_path
#         self.chunk_duration = chunk_duration
#         self.raw = mne.io.read_raw_edf(edf_path, preload=False)
#         self.sampling_freq = self.raw.info['sfreq']
#         self.num_channels = len(self.raw.ch_names)
#         self.total_samples = self.raw.n_times
#         self.annotations = self.raw.annotations
#         self.duration = self.total_samples / self.sampling_freq

#     def __getitem__(self):
#         """
#         Stream EEG data in chunks.

#         Yields:
#             np.ndarray: The next chunk of EEG data.
#         """
#         start_time = 0
#         while start_time < self.duration:
#             end_time = start_time + self.chunk_duration
#             if end_time > self.duration:
#                 end_time = self.duration

#             # Fetching the chunk of data
#             start_sample = int(start_time * self.sampling_freq)
#             end_sample = int(end_time * self.sampling_freq)
#             data_chunk = self.raw.get_data(start=start_sample, stop=end_sample)

#             yield data_chunk

#             start_time += self.chunk_duration

#     def get_annotations_in_chunk(self, chunk_start, chunk_end):
#         """
#         Get annotations that fall within a specific chunk of data.

#         Args:
#             chunk_start (float): Start time of the chunk in seconds.
#             chunk_end (float): End time of the chunk in seconds.

#         Returns:
#             list: List of annotations within the specified chunk.
#         """
#         annotations_in_chunk = [annot for annot in self.annotations if chunk_start <= annot['onset'] < chunk_end]
#         return annotations_in_chunk



class EEGStream:
    def __init__(self, *edf_paths, chunk_duration=1.0):
        """
        Initialize an EEGStream object to handle streaming EEG data in chunks from multiple EDF files.

        Args:
            edf_paths (tuple[str, ...]): Paths to the EDF files.
            chunk_duration (float): Duration of each chunk in seconds.
        """
        self.edf_paths = edf_paths
        self.chunk_duration = chunk_duration
        self.files = []
        self.cumul_length = [0]  # Cumulative samples across files

        for path in edf_paths:
            raw = mne.io.read_raw_edf(path, preload=True)
            self.files.append({
                'raw': raw,
                'samples': raw.n_times,
                'sfreq': raw.info['sfreq'],
                'annotations': raw.annotations
            })
            self.cumul_length.append(self.cumul_length[-1] + raw.n_times)

    def __getitem__(self, chunk_index):
        chunk_size = int(self.chunk_duration * self.files[0]['sfreq'])
        global_start_sample = chunk_index * chunk_size
        file_index, local_start = self.get_position(global_start_sample)
        if file_index == -1:
            raise IndexError("Chunk index out of range")

        raw = self.files[file_index]['raw']
        global_end_sample = global_start_sample + chunk_size
        data_chunks = []

        while global_start_sample < global_end_sample and file_index < len(self.files):
            local_end = local_start + (global_end_sample - global_start_sample)
            if local_end > self.files[file_index]['samples']:
                local_end = self.files[file_index]['samples']

            data_chunk = raw.get_data(start=local_start, stop=local_end)
            data_chunks.append(data_chunk)

            global_start_sample += local_end - local_start
            file_index += 1
            if file_index < len(self.files):
                raw = self.files[file_index]['raw']
                local_start = 0

        return np.concatenate(data_chunks, axis=1) if data_chunks else np.array([])

    def get_position(self, global_sample):
        for i in range(1, len(self.cumul_length)):
            if global_sample < self.cumul_length[i]:
                return i - 1, global_sample - self.cumul_length[i - 1]
        return -1, -1  # if global_sample is out of bounds

    def get_annotations_in_chunk(self, chunk_index):
        """
        Get annotations that fall within a specific chunk of EEG data.

        Args:
            chunk_index (int): Index of the chunk for which to retrieve annotations.

        Returns:
            list: List of annotations within the specified chunk.
        """
        chunk_start_time = chunk_index * self.chunk_duration
        chunk_end_time = (chunk_index + 1) * self.chunk_duration
        file_index, _ = self.get_position(chunk_index * int(self.chunk_duration * self.files[0]['sfreq']))

        annotations = self.files[file_index]['annotations']
        annotations_in_chunk = [annot for annot in annotations if chunk_start_time <= annot['onset'] < chunk_end_time]
        return annotations_in_chunk