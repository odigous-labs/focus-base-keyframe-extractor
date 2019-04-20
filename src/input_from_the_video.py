# import the necessary packages
import os

import time
import cv2
import imutils
import natsort
import numpy as np
from centroid_tracker.centroidtracker import CentroidTracker
from yolo.model import YOLO


class Object_Selector():

    def __init__(self):
        # initialize our centroid tracker and frame dimensions
        self.ct = CentroidTracker()

    def select_the_object_from_the_frame(self, folder_path):
        # checking whether the given path is a directory
        if os.path.isdir(folder_path):
            fnames = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, f))]

        else:
            fnames = [folder_path]

        fnames = natsort.natsorted(fnames, reverse=False)

        frame_predictions_dictionary_with_id = {}

        for f in fnames:

            (H, W) = (None, None)

            frame_object = cv2.imread(f)

            # tracking
            frame = imutils.resize(frame_object, width=400)

            # if the frame dimensions are None, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # predict the objects in the frame
            predictions = YOLO().predict(frame)

            rects = []
            centroids = []
            boxes = predictions[0]
            labels = predictions[1]
            frame_predictions_dictionary_with_id[str(f)[len(folder_path):]] = []

            image = frame.copy()
            image_h, image_w, _ = image.shape

            color_mod = 255

            for i in range(len(boxes)):
                xmin = int(boxes[i][0] * image_w)
                ymin = int(boxes[i][1] * image_h)
                xmax = int(boxes[i][2] * image_w)
                ymax = int(boxes[i][3] * image_h)

                temp = np.array([xmin, ymin, xmax, ymax])

                rects.append(temp.astype("int"))

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (color_mod, 255, 0), 2)

                centroids.append([int((xmin + xmax) / 2), int((ymin + ymax) / 2)])

            # update our centroid tracker using the computed set of bounding
            # box rectangles
            objects = self.ct.update(rects)

            object_ids_available = []
            labels_on_frame = []

            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)

                cv2.circle(image, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)
                centroid_co = [centroid[0], centroid[1]]

                if centroid_co in centroids:
                    i = centroids.index(centroid_co)
                    text = "ID {}".format(objectID)
                    object_ids_available.append(objectID)
                    labels_on_frame.append(labels[i])
                    frame_predictions_dictionary_with_id[str(f)[len(folder_path):]].append(text)

                cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # show the output frame
            cv2.imshow("Frame", image)
            cv2.waitKey(3000) & 0xFF

            cv2.destroyAllWindows()
            condition = True

            while condition:
                selected_inp = input(
                    "\nEnter the object ID to be focused or enter 'S' to skip and 'T' to terminate and 'A' to show "
                    "again (\"Ex: ""3\") :")

                if selected_inp == "S":
                    break
                elif selected_inp == "T":
                    condition = False
                    exit()
                elif selected_inp == "A":
                    # show the output frame
                    cv2.imshow("Frame", image)
                    cv2.waitKey(2000) & 0xFF

                    cv2.destroyAllWindows()
                    condition = True
                elif selected_inp.isdigit():
                    if int(selected_inp) in object_ids_available:
                        label_selected = labels_on_frame[object_ids_available.index(int(selected_inp))]
                        print("successfully selected the object ID ", selected_inp, label_selected)
                        return int(selected_inp), label_selected
                    else:
                        print("ID not available. Try again")
                        condition = True
                else:
                    print("invalid input. Try again")
                    condition = True

        # do a bit of cleanup
        cv2.destroyAllWindows()


def run():
    objs = Object_Selector()
    frame_output_path = "./data/generated_frames/"
    print(objs.select_the_object_from_the_frame(frame_output_path))


if __name__ == "__main__":
    run()
