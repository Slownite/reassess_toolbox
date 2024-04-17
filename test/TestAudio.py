import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.AudioStream import AudioEager, AudioStream 

class TestAudioClasses(unittest.TestCase):
    def setUp(self):
        # This is a path to an example audio file you want to use for testing
        self.filepath = './test/output_audio.wav'
        self.sr = 22050  # Example sampling rate, adjust as necessary
        self.audio_eager = AudioEager(self.filepath, self.sr)
        self.audio_stream = AudioStream(self.filepath, 1024, self.sr, 2048, 1024)

    def test_audio_eager_length(self):
        self.assertEqual(len(self.audio_eager), self.audio_eager.total_samples)

    def test_audio_eager_get_samples(self):
        start_sample = 100
        end_sample = 200
        samples = self.audio_eager.get_samples(start_sample, end_sample)
        self.assertEqual(len(samples), end_sample - start_sample)

    def test_audio_eager_get_duration(self):
        duration = self.audio_eager.get_duration()
        self.assertAlmostEqual(duration, self.audio_eager.total_samples / self.sr)

    def test_audio_stream_get_samples(self):
        start_sample = 100
        end_sample = 200
        samples = self.audio_stream.get_samples(start_sample, end_sample)
        self.assertTrue(len(samples) > 0)  # Adjust based on expected behavior

    def test_audio_stream_get_samples_with_soundfile(self):
        start_sample = 100
        end_sample = 200
        samples = self.audio_stream.get_samples_with_soundfile(start_sample, end_sample)
        self.assertTrue(len(samples) > 0)  # Adjust based on expected behavior

    def test_audio_stream_length(self):
        self.assertEqual(len(self.audio_stream), self.audio_stream.total_samples)

    def test_audio_stream_get_duration(self):
        duration = self.audio_stream.get_duration()
        self.assertAlmostEqual(duration, self.audio_stream.total_samples / self.sr)

if __name__ == '__main__':
    unittest.main()
