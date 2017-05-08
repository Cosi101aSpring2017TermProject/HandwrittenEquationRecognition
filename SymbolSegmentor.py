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
import skimage.measure
import skimage.io
import re
import numpy

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

    def trimZeros(self, two_d_array_img):
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        found_first_nonzero_pixel = True
        for y in range(0, len(two_d_array_img)):
            for x in range(0, len(two_d_array_img[y])):
                if two_d_array_img[y][x] != 0:
                    if found_first_nonzero_pixel:
                        x1 = x
                        x2 = x
                        y1 = y
                        y2 = y
                        found_first_nonzero_pixel = False
                    if x < x1:
                        x1 = x
                    if x > x2:
                        x2 = x
                    if y < y1:
                        y1 = y
                    if y > y2:
                        y2 = y

        trimmed_image = two_d_array_img[y1:y2, x1:x2]
        if self.show_imgs:
            print(trimmed_image)
            print("x1:%d y1:%d x2:%d y2:%d" % (x1, y1, x2, y2))
            plt.imshow(trimmed_image)
            plt.show()
        return x1, y1, x2, y2, trimmed_image

    def segment(self, img_folder_dir, file_name):
        full_dir = img_folder_dir + file_name
        im_gray = cv2.imread(full_dir, 0)
        raw_image = cv2.imread(full_dir, 0)
        if self.show_imgs:
            plt.imshow(im_gray)
            plt.show()


        # convert to grayscale
        im_gray = cv2.blur(im_gray, (3, 3), 2)
        # blurred_raw = cv2.blur(raw_image, (3, 3), 2)
        # im_gray = cv2.bilateralFilter(im_gray, 5, 7, 7)
        # im_gray2 = cv2.GaussianBlur(im_gray, (5, 5), 0)
        # im_gray3 = cv2.GaussianBlur(im_gray, (7, 7), 0)

        im = 255 - im_gray
        if self.show_imgs:
            plt.imshow(im_gray)
            plt.show()

        # threshold
        ret, im_th = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY_INV)


        # find contour, ctrs is the contour
        # im2, ctrs, hier = cv2.findContours(im_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #openCV
        connected_regions, num_of_labels = skimage.measure.label(im_th, connectivity=5, neighbors=8, background=0, return_num=True)
        print(str(num_of_labels)+" regions found in "+file_name)
        if self.show_imgs:
            plt.imshow(im_th)
            plt.show()
            print(connected_regions)
            plt.imshow(connected_regions)
            plt.show()
        for i in range(1, num_of_labels + 1):
            testimage = (connected_regions == i) * 200
            if self.show_imgs:
                plt.imshow(testimage)
                plt.show()
            x1, y1, x2, y2, symbol_img = self.trimZeros(testimage)
            rec_string = str(y1) + '_' + str(y2) + '_' + str(x1) + '_' + str(x2)
            # MARK: IO
            if not os.path.exists("symbols"):
                os.makedirs("symbols")
            saved_name = "symbols/" + file_name.replace('.png', '_unclassified_' + rec_string + '.png')
            trimmed_img = im_th[y1:y2, x1:x2]
            inners, numer_of_inner = skimage.measure.label(trimmed_img, connectivity=5,
                                                           neighbors=8, background=0, return_num=True)
            if numer_of_inner == 1:
                if self.show_imgs:
                    plt.imshow(raw_image[y1:y2, x1:x2])
                    plt.show()
                if (y2-y1)*(x2-x1) > 10:
                    cv2.imwrite(saved_name, raw_image[y1:y2, x1:x2])
            else:
                temp_x1 = 0
                temp_y1 = 0
                temp_x2 = 0
                temp_y2 = 0
                temp_image = []
                is_first = True
                for j in range(1, numer_of_inner + 1):
                    inner = (connected_regions == j) * 200
                    x1_in, y1_in, x2_in, y2_in, trimmed_inner = self.trimZeros(inner)
                    if is_first:
                        temp_x1 = x1_in
                        temp_x2 = x2_in
                        temp_y1 = y1_in
                        temp_y2 = y2_in
                        temp_image = trimmed_inner
                        is_first = False
                    prev_area = (temp_x2-temp_x1)*(temp_y2-temp_y1)
                    current_area = (x2_in-x1_in)*(y2_in-y1_in)
                    if current_area > prev_area:
                        temp_x1 = x1_in
                        temp_x2 = x2_in
                        temp_y1 = y1_in
                        temp_y2 = y2_in
                        temp_image = trimmed_inner
                if (temp_x2-temp_x1)*(temp_y2-temp_y1) > 10:
                    print(temp_image)
                    cv2.imwrite(saved_name, numpy.array(temp_image))




        # rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        # # zero = 0
        # img_num_counter = 0
        # rect_counter = 0
        # for contour in ctrs:
        #     symbol_container = np.zeros((128, 1693, 3), np.uint8)
        #     cv2.drawContours(symbol_container, contour, -1, (0, 255, 0), 3)
        #     rect = rects[rect_counter]
        #     rect_counter = rect_counter + 1
        #     # cv2.rectangle(symbol_container, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        #     vertex_x_begin = rect[0]
        #     vertex_x_end = rect[0] + rect[2]
        #     vertex_y_begin = rect[1]
        #     vertex_y_end = rect[1] + rect[3]
        #     crop_img = symbol_container[vertex_y_begin:vertex_y_end, vertex_x_begin:vertex_x_end]
        #     # print()
        #     # for i in range(0, vertex_x_end-vertex_x_begin):
        #     #    for j in range(0, vertex_y_end - vertex_y_begin):
        #     #        int_tmp = cv2.pointPolygonTest(contour, (i,j), False)
        #     #        if int_tmp == 0:
        #     #            crop_img[j][i] = (0, 255, 0)
        #     if self.show_imgs:
        #         plt.imshow(crop_img)
        #         plt.show()
        #     # print(crop_img[0])
        #     # print(im_gray[0])
        #     rec_string = str(vertex_y_begin)+'_'+str(vertex_y_end)+'_'+str(vertex_x_begin)+'_'+str(vertex_x_end)
        #     if not os.path.exists("symbols"):
        #         os.makedirs("symbols")
        #     cv2.imwrite("symbols/" + file_name.replace('.png', '_unclassified_'+rec_string+'.png'), cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
        #     img_num_counter = img_num_counter + 1
        #     # rect_counter = zero
        #     # plt.imshow(ctrs)
        #     # plt.show()
        #
        # for rect in rects:
        #     cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # im = 255 - im
        # # plt.imshow(im)
        # # plt.show()


    def __init__(self):
        print("start segmenting equation pics in annotated folder")
        files = [f for f in listdir('./annotated')]

        for f in files:
            a = f.split("_")
            if len(a) == 3:
                print(f)
                self.segment('./annotated/', f)
