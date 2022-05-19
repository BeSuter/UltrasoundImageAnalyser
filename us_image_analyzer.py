import os
import cv2
import argparse

import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    all_results = []
    all_max = []
    all_min = []
    all_area = []
    PIXEL_AREA = 0.000313
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f",
                    "--file",
                    required=False,
                    default=False,
                    help="If wanted, please specify the name of the file where to append the values to.")
    ap.add_argument("-t",
                    "--time",
                    required=False,
                    default=999999999,
                    help="Specifiy the time point of the analyzed sample. Will be specified in the csv file.",
                    type=float)
    ap.add_argument("-i",
                    "--image",
                    nargs='+',
                    required=True,
                    help="Please enter the path to each image you want to analyze")
    ap.add_argument("-p",
                    "--NOplot",
                    required=False,
                    action='store_false',
                    help="Flag for plotting image histograms. "
                         "If no plots are desired, set this flag to False by typing --NOplot!")
    args = vars(ap.parse_args())

    # load the image, clone it, and setup the mouse callback function
    for img_path in args["image"]:
        all_images = []
        refPt = []
        cropping = False

        image = cv2.imread(img_path)
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
                area = cv2.contourArea(np.around(np.array([[pt] for pt in refPt])).astype(np.int32))
                mean = np.mean(masked_grey_array)
                std = np.std(masked_grey_array)
                min = np.min(masked_grey_array)
                max = np.max(masked_grey_array)
                hist = cv2.calcHist([all_images[0]], [0], mask_array, [256], [0, 256])
                # print(round(mean, 2), " , ", round(np.std(masked_grey_array), 2))
                mm, ss = cv2.meanStdDev(all_images[0], mask=mask_array)
                # Show the cropped area of image
                cv2.imshow("Masked Image", targetROI)
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                print("##############################################################")
                print(f"Single image statistics have been calculated for {img_path}:")
                print(f"Chosen area was {round(area*PIXEL_AREA, 2)} cm^2")
                print("Mean pixel value: ", round(mean, 2),
                      ", standard deviation is: +/- ",
                      round(std, 2))
                print("Max pixel value: ", round(max, 2),
                      ", min pixel value: ", round(min,2))
                if args["NOplot"]:
                    print("Plotting pixel histogram...")
                    plt.figure()
                    plt.title(f"Pixel Histogram of {img_path}")
                    plt.xlabel("Bins")
                    plt.ylabel("# of Pixels")
                    plt.plot(hist)
                    plt.xlim([0, 256])
                    plt.show()
                break
        # close all open windows
        cv2.destroyAllWindows()
        all_results.append(mean)
        all_min.append(min)
        all_max.append(max)
        all_area.append(area)
    print("##############################################################")
    print()
    print("##############################################################")
    print(f"Full image analyzes of {len(all_results)} images is complete")
    print(f"Mean pixel brightness over all images is: ", round(np.mean(all_results), 2),
          ", standard deviation is: +/- ",
          round(np.std(all_results), 2))
    print(f"Average max pixel value is {round(np.mean(all_max), 2)} +/- {round(np.std(all_max), 2)}")
    print(f"Average min pixel value is {round(np.mean(all_min), 2)} +/- {round(np.std(all_min), 2)}")
    print(f"Average cropped area was {round(np.mean(all_area)*PIXEL_AREA, 2)} "
          f"+/- {round(np.std(all_area)*PIXEL_AREA, 2)} cm^2")
    print()
    print("!!!! Computed area is only approximately correct for images with a depth scale of 7.8 cm !!!!")
    print()

    if args["file"]:
        print(f"Saving data to {args['file']} file")
        if args["time"] == 999999999:
            print("Assuming that no time point was provided")
            time_point = [np.nan]
        else:
            time_point = [args["time"]]
        df_new = pd.DataFrame({'time': time_point,
                               'mean': [round(np.mean(all_results), 2)],
                               'mean_std': [round(np.std(all_results), 2)],
                               'avg_max_pix': [round(np.mean(all_max), 2)],
                               'max_pix_std': [round(np.std(all_max), 2)],
                               'avg_min_pix': [round(np.mean(all_min), 2)],
                               'min_pix_std': [round(np.std(all_min), 2)],
                               'avg_area': [round(np.mean(all_area)*PIXEL_AREA, 2)],
                               'area_std': [round(np.std(all_area)*PIXEL_AREA, 2)],
                               '#samples': [len(all_results)]})
        if os.path.exists(args["file"]):
            df = pd.read_csv(args["file"])
            df = df.append(df_new, ignore_index=True)
        else:
            df = pd.DataFrame(df_new)
        df = df.sort_values(by=['time'])
        df.to_csv(args["file"], index=False)
