# import the necessary packages
import os

import cv2
import imutils
import numpy as np
import natsort

from centroid_tracker.centroidtracker import CentroidTracker
from yolo.model import YOLO
from tqdm import tqdm


class Focus_Base_Keyframe_Extractor():

    def __init__(self):
        # initialize our centroid tracker and frame dimensions
        self.ct = CentroidTracker()

    def generate_frames_with_class_objects(self, folder_path, class_label):
        # checking whether the given path is a directory
        if os.path.isdir(folder_path):
            fnames = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, f))]

        else:
            fnames = [folder_path]

        fnames = natsort.natsorted(fnames, reverse=False)

        frames_with_focused_object = []

        for f in tqdm(fnames, desc='Processing Batch'):

            (H, W) = (None, None)

            frame_object = cv2.imread(f)

            # tracking
            frame = imutils.resize(frame_object, width=400)

            # if the frame dimensions are None, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # predict the objects in the frame
            predictions = YOLO().predict(frame)

            labels = predictions[1]

            print(str(f)[len(folder_path):] + "->")
            print(labels)
            print("\n")

            if class_label in labels:
                frames_with_focused_object.append(str(f)[len(folder_path):])

        return frames_with_focused_object


def run():
    objs = Focus_Base_Keyframe_Extractor()
    frame_output_path = "./data/generated_frames/"
    print(objs.generate_frames_with_class_objects(frame_output_path, "person"))


if __name__ == "__main__":
    run()
