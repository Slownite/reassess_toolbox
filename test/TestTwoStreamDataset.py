#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch
import pathlib
import numpy as np

# Assuming the dataset class is in dataset_module
from datasets import TwoStreamDataset


class TestTwoStreamDataset(unittest.TestCase):
    def setUp(self):
        # Mock the pathlib.Path object
        self.mock_path = MagicMock(spec=pathlib.Path)

        # Create example file paths
        self.rgb_files = ["rgb_1.mp4", "rgb_2.mp4"]
        self.of_files = ["flow_1.mp4", "flow_2.mp4"]
        self.edf_files = ["file_1.edf", "file_2.edf"]

        # Mock the glob method to return different paths based on input
        def mock_glob(pattern):
            if "rgb" in pattern:
                return self.rgb_files
            elif "flow" in pattern:
                return self.of_files
            elif ".edf" in pattern:
                return self.edf_files

        self.mock_path.glob = mock_glob

        # Mock EEGVideoSynchronizer and VideoStreamer
        self.mock_synchronizer = MagicMock()
        self.mock_synchronizer.__getitem__.return_value = (
            np.zeros((5, 224, 224, 3)),  # frames
            np.zeros((5, 224, 224, 2)),  # compressed_flows
            None,
            None,
            "annotations",
        )
        self.mock_streamer = MagicMock()
        self.mock_streamer.__len__.return_value = 10

    @patch("utils.EEGVideoSynchronizer", return_value=None)
    @patch("utils.VideoStreamer", return_value=None)
    def test_init(self, mock_streamer, mock_synchronizer):
        dataset = TwoStreamDataset(self.mock_path, block=5)
        self.assertEqual(dataset.size, 10)
        self.assertEqual(dataset.block, 5)

    def test_len(self):
        dataset = TwoStreamDataset(self.mock_path, block=5)
        self.assertEqual(len(dataset), 10)

    @patch("utils.videos_frame_to_flow", return_value=np.zeros((5, 224, 224, 2)))
    def test_getitem_valid_index(self, mock_flow_func):
        dataset = TwoStreamDataset(self.mock_path, block=5)
        result = dataset.__getitem__(0)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_getitem_out_of_bounds(self):
        dataset = TwoStreamDataset(self.mock_path, block=5)
        with self.assertRaises(IndexError):
            dataset.__getitem__(5)


if __name__ == "__main__":
    unittest.main()
