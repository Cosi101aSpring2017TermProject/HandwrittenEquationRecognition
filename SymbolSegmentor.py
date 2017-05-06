#This class draw boxes in each symbols in handwritten equation
#output a list of images in raw resolution

#Possibly helpful reference: http://cs229.stanford.edu/proj2013/JimenezNguyen_MathFormulas_final_paper.pdf
#http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html
#used basic format: https://github.com/zhjch05/cs101-hw4/blob/master/termproject/cvFindContour.py
# why can't I push it
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import os
import skimage.io
import re

class SymbolSegmentor:

    show_imgs = False

    def get_folder(self, full_dir: str):
        print(os.sep)
        dir_component = full_dir.split(os.sep)
        print(dir_component[len(dir_component) - 1])
        return dir_component[len(dir_component) - 1]

    def save_image(self, folder, img_filename, postfix, img_tmp):
        # full_filename = img_filename.replace('.png', postfix + '.png')
        # full_path = os.path.join('\symbols', full_filename)
        # if folder != self.get_folder(os.getcwd()):
        #     if not os.path.exists(os.getcwd() + os.sep + folder):
        #         os.makedirs(os.getcwd() + os.sep + folder)
        #     os.chdir(folder)
        # print("save " + full_filename + '\nat ' + os.getcwd())
        if not os.path.exists(folder):
            os.makedirs(folder)
        # dir_new = folder + '/' + full_filename
        # skimage.io.imsave(os.getcwd()+'/'+folder+'/'+img_filename, img_tmp)
        cv2.imwrite(img_filename, img_tmp)
        # plt.imshow(img_tmp)
        # plt.show()

    def segment(self, img_folder_dir, file_name):
        full_dir = img_folder_dir + file_name
        im_gray = cv2.imread(full_dir, 0)
        if self.show_imgs:
            plt.imshow(im_gray)
            plt.show()


        # convert to grayscale
        im_gray = cv2.blur(im_gray, (5, 5), 2)
        # im_gray = cv2.bilateralFilter(im_gray, 5, 7, 7)
        # im_gray2 = cv2.GaussianBlur(im_gray, (5, 5), 0)
        # im_gray3 = cv2.GaussianBlur(im_gray, (7, 7), 0)

        im = 255 - im_gray
        if self.show_imgs:
            plt.imshow(im_gray)
            plt.show()

        # threshold
        ret, im_th = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY_INV)
        # ret2, im_th2 = cv2.threshold(im_th, 127, 255, cv2.THRESH_BINARY_INV)
        # ret3, im_th3 = cv2.threshold(im_th2, 127, 255, cv2.THRESH_BINARY_INV)

        if self.show_imgs:
            plt.imshow(im_th)
            plt.show()
        # find contour, ctrs is the contour
        im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ctr = ctrs[1]
        # ctr = ctrs[4]
        # im3 = np.zeros((128, 1693, 3), np.uint8)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        # zero = 0
        img_num_counter = 0
        rect_counter = 0
        for contour in ctrs:
            symbol_container = np.zeros((128, 1693, 3), np.uint8)
            cv2.drawContours(symbol_container, contour, -1, (0, 255, 0), 3)
            rect = rects[rect_counter]
            rect_counter = rect_counter + 1
            # cv2.rectangle(symbol_container, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            # print(rect[0])
            # (rect[1])
            # print(rect[2])
            # print(rect[3])
            vertex_x_begin = rect[0] - 10
            vertex_x_end = rect[0] + rect[2] + 10
            vertex_y_begin = rect[1] - 10
            vertex_y_end = rect[1] + rect[3] + 10
            # print(vertex_x_end - vertex_x_begin)
            # print(vertex_y_end - vertex_y_begin)
            crop_img = symbol_container[vertex_y_begin:vertex_y_end, vertex_x_begin:vertex_x_end]
            # print()
            # for i in range(0, vertex_x_end-vertex_x_begin):
            #    for j in range(0, vertex_y_end - vertex_y_begin):
            #        int_tmp = cv2.pointPolygonTest(contour, (i,j), False)
            #        if int_tmp == 0:
            #            crop_img[j][i] = (0, 255, 0)
            if self.show_imgs:
                plt.imshow(crop_img)
                plt.show()
            # print(crop_img[0])
            # print(im_gray[0])
            rec_string = str(vertex_y_begin)+'_'+str(vertex_y_end)+'_'+str(vertex_x_begin)+'_'+str(vertex_x_end)
            # MARK: IO
            if not os.path.exists("symbols"):
                os.makedirs("symbols")
            cv2.imwrite("symbols/" + file_name.replace('.png', '_unclassified_'+rec_string+'.png'), cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
            # MARK: IO
            img_num_counter = img_num_counter + 1
            # rect_counter = zero
            # plt.imshow(ctrs)
            # plt.show()

        for rect in rects:
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        im = 255 - im
        # plt.imshow(im)
        # plt.show()





    def __init__(self):
        print("start segmenting equation pics in annotated folder")
        files = [f for f in listdir('./annotated')]

        for f in files:
            a = f.split("_")
            if len(a) == 3:
                print(f)
                self.segment('./annotated/', f)
