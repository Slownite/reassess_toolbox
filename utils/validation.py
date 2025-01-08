import cv2 as cv
import pathlib
from encoding_labels import read_annotations


def validate_annotations_videos(annotations_files, video_files):
    videos_framecount = sum([int(cv.VideoCapture(video).get(
        cv.CAP_PROP_FRAME_COUNT)) for video in video_files])
    annotations_count = sum([len(list(read_annotations(file)))
                            for file in annotations_files])
    assert videos_framecount == annotations_count
