import cv2
import time
import argparse

import numpy as np
import numpy.ma as ma
import scipy.stats as st
import matplotlib.pyplot as plt

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
all_images = []
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    # global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cropping = True
        if len(refPt) < 2:
            new_image = cv2.circle(image, refPt[-1], 2, (0, 255, 0), -1)
        else:
            new_image = cv2.line(image, refPt[-2], refPt[-1], (0, 255, 0), 2)
        all_images.append(new_image.copy())
        cv2.imshow("image", new_image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
all_images.append(image.copy())
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        #clean up
        cv2.destroyWindow("Masked Image")
        all_images.pop(-1)
        image = all_images[-1].copy()
        refPt.pop(-1)
        cropping = False
    elif key == ord("k"):
        # Cropping is done, approximate all points to a closed polygon and compute statistics of points
        # inside the polygon.
        black_frame = np.zeros_like(image).astype(np.uint8)
        cv2.fillPoly(black_frame, np.asarray([refPt]), (255, 255, 255))
        mask = black_frame == 255
        mask_array = np.asarray(mask[:, :, 0], dtype=np.uint8)
        targetROI = all_images[0] * mask
        masked_grey_array = ma.masked_array(np.asarray(cv2.cvtColor(all_images[0], cv2.COLOR_BGR2GRAY)),
                                            mask=np.logical_not(mask_array)).flatten()
        mean = np.mean(masked_grey_array)
        std = np.std(masked_grey_array)
        hist = cv2.calcHist([all_images[0]], [0], mask_array, [256], [0, 256])
        print(round(mean, 2), " , ", round(np.std(masked_grey_array), 2))
        mm, ss = cv2.meanStdDev(all_images[0], mask=mask_array)
        # Show the cropped area of image
        cv2.imshow("Masked Image", targetROI)
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        print("##############################################################")
        print("Image statistics have been calculated:")
        print("Mean pixel value: ", round(mean, 2),
              ", standard deviation is: ",
              round(std, 2))
        print("Plotting pixel histogram...")
        print("##############################################################")
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
        print(len(masked_grey_array))
        break
# close all open windows
cv2.destroyAllWindows()
