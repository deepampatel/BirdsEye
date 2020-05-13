import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
import zipfile
import json
import argparse

# from collections import defaultdict
# from io import StringIO
import cv2

tf.gfile = tf.io.gfile
if tf.__version__ < "1.13.0":
    raise ImportError(
        "Please upgrade your tensorflow installation to v1.13.* or later!"
    )

sys.path.insert(0, "utils")
import label_map_util
import people_class_util as class_utils
import visualization_utils as vis_util


MODEL_NAME = "ssd_mobilenet_v1_coco_2018_01_28"  # This is the fastest model but with low accuracy
# MODEL_NAME = 'faster_rcnn_nas_lowproposals_coco_2017_11_08'
# MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + ".tar.gz"
DOWNLOAD_BASE = "http://download.tensorflow.org/models/object_detection/"

PATH_TO_CKPT = MODEL_NAME + "/frozen_inference_graph.pb"

PATH_TO_LABELS = "utils/person_label_map.pbtxt"
M = np.load("../../transormation_matrix.npy")
M = np.array(M, np.float32)

if not os.path.exists(MODEL_FILE):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if "frozen_inference_graph.pb" in file_name:
            tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")
NUM_CLASSES = 50
# loading specified class/category description
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)


# some helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


cap = cv2.VideoCapture("test_video.mp4")
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            # read every frame
            success, image_np = cap.read()
            width, height, _ = image_np.shape

            if not success:
                print(">>>>>  End of the video file...")
                break

            # flaten image using numpy
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            scores = detection_graph.get_tensor_by_name("detection_scores:0")
            classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded},
            )
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.2,
            )

            i = int(num_detections[0])
            temp = []
            scores = np.squeeze(scores)
            for k, j in enumerate(boxes[0][:i]):
                if scores[k] > 0.2:

                    ymin = int(j[0] * width)
                    xmin = int(j[1] * height)
                    ymax = int(j[2] * width)
                    xmax = int(j[3] * height)
                    temp.append([xmin, ymin])

            pts = cv2.perspectiveTransform(
                np.array(temp, dtype=np.float32,).reshape(1, -1, 2), M,
            )

            img = np.zeros(((5000, 2000, 3)), np.uint8)
            k = 0
            for point in pts[0]:
                print(point, temp[k])
                cv2.circle(img, tuple(point), 40, (0, 255, 0), -1)

                k += 1

            # creating annotation array which can be sent to API for further operations

            # writting to json file for now, this block will contain API/DB code to handle data.
            # show annotated image on desktop
            cv2.imshow("object detection", img)
            cv2.imshow("II", image_np)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
