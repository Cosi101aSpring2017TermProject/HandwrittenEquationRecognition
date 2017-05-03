#This class draw boxes in each symbols in handwritten equation
#output a list of images in raw resolution

#Possibly helpful reference: http://cs229.stanford.edu/proj2013/JimenezNguyen_MathFormulas_final_paper.pdf
#http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html
#used basic format: https://github.com/zhjch05/cs101-hw4/blob/master/termproject/cvFindContour.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

class SymbolSegmentor:

    def save_image(mun_tmp, img_tmp):
        string_temp = "test_output_" + str(mun_tmp) + ".png"
        cv2.imwrite(string_temp, img_tmp)
        plt.imshow(img_tmp)
        plt.show()

    # same as in the openCV tutorial
    im_gray = cv2.imread("test_image_for_contour.png", 0)
    plt.imshow(im_gray)
    plt.show()

    # convert to grayscale
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    im = 255 - im_gray

    # threshold
    ret, im_th = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)

    # find contour, ctrs is the contour
    im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ctr = ctrs[1]
    # ctr = ctrs[4]
    # im3 = np.zeros((128, 1693, 3), np.uint8)
    img_num_counter = 0
    for contour in ctrs:
        symbol_container = np.zeros((128, 1693, 3), np.uint8)
        cv2.drawContours(symbol_container, contour, -1, (0, 255, 0), 3)
        save_image(img_num_counter, symbol_container)
        img_num_counter = img_num_counter + 1


    # plt.imshow(ctrs)
    # plt.show()
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for rect in rects:
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    im = 255 - im
    plt.imshow(im)
    plt.show()
    cv2.imwrite('test_output.png', im)

    # raw_imgs = []
    # img_boxes = []
    def __init__(self):
        raw_imgs = []
        img_boxes = []
        #TODO: read all equations in annotated folder and drawing boxes around them recode the position of the box in self.img_boxes
        #TODO: put all images of single symbol in raw_imgs, NO processing needed.

