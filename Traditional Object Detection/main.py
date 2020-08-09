import argparse
import time

import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression
from tensorflow.keras.applications import ResNet50V2, imagenet_utils
from tensorflow.keras.applications.resnet import preprocess_input

from utils import image_pyramid,sliding_window

# IMG_PATH = "test.jpg"

# img = cv2.imread(IMG_PATH)
# images = utils.image_pyramid(img)
# for i in images:
#     cv2.imshow("sam",i)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
    help="Path of the image")
ap.add_argument("-s","--size",type=str,default="(200, 150)",
    help="ROI size (in pixels)")
ap.add_argument("-c","--min-conf",type=float,default=0.9,
    help="Minimum confidence required for the ibject detection")
ap.add_argument("-v","--visualize",type=bool,default=False,
    help="Boolean to decide whether to show debugging visualization")
args = vars(ap.parse_args())

WIDTH = 600
PYR_SCALE = 1.5 
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224,224)

resnetModel = ResNet50V2(include_top=True,weights='imagenet')
orig = cv2.imread(args["image"])
orig = imutils.resize(orig,width=WIDTH)
(H,W) = orig.shape[:2]

pyramid = image_pyramid(orig,scale = PYR_SCALE,min_size=ROI_SIZE)

rois = []
locs = []
start = time.time()

for image in pyramid:
    scale = W/float(image.shape[1])

    for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
        pass