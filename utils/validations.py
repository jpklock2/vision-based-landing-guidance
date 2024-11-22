import os
import numpy as np


class DefaultError(Exception):
    pass


def validate_sequence(sequence_path):

    # groundtruth.txt: the path of the bounding boxes file
    boxes_path = os.path.join(sequence_path, 'groundtruth.txt')
    frames_path = os.path.join(sequence_path, 'frames')

    # load bounding boxes using numpy
    boxes = np.loadtxt(boxes_path, delimiter=',')
    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, axis=0)

    frames_number = len([name for name in os.listdir(frames_path) if os.path.isfile(os.path.join(frames_path, name))])
    frames_difference = frames_number - boxes.shape[0]

    if frames_difference > 0:
        with open(boxes_path, "a") as boxes_file:
            for _ in range(frames_difference):
                boxes_file.write("0,1,2,3\n")
