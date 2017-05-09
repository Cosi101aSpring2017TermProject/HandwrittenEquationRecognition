#This class draw boxes in each symbols in handwritten equation
#output a list of images in raw resolution

#Possibly helpful reference: http://cs229.stanford.edu/proj2013/JimenezNguyen_MathFormulas_final_paper.pdf
#http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html
#used basic format: https://github.com/zhjch05/cs101-hw4/blob/master/termproject/cvFindContour.py
# why can't I push it
import cv2
import matplotlib.pyplot as plt
from os import listdir
import os
import skimage.measure
import skimage.io
import numpy

class SymbolSegmentor:

    show_imgs = False
    # For debugging, set to true will show all the steps of image processing
    loading_imgs = False
    # Do you want to do the segmentation of each input, it might take some time.
    # If you run this the first time, or want to read from other place, set to True

    ignored_pixel_limit = 60
    # if the area of clipped symbol is lower than this number, we think it's probably a noise, ignore

    def get_folder(self, full_dir: str):
        print(os.sep)
        dir_component = full_dir.split(os.sep)
        print(dir_component[len(dir_component) - 1])
        return dir_component[len(dir_component) - 1]

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
        return x1, y1, x2, y2, trimmed_image

    # where all the segmentation happens
    def segment(self, img_folder_dir, file_name):
        full_dir = img_folder_dir + file_name
        im_gray = cv2.imread(full_dir, 0)
        raw_image = cv2.imread(full_dir, 0)
        img_with_box = cv2.imread(full_dir, 0)
        if self.show_imgs:
            plt.imshow(im_gray)
            plt.title("raw image: "+file_name)
            plt.show()

        #  blur the image a little bit
        im_gray = cv2.blur(im_gray, (3, 3), 2)

        im = 255 - im_gray
        if self.show_imgs:
            print(im_gray)
            plt.imshow(im_gray)
            plt.title("blurred image: " + file_name)
            plt.show()

        # threshold
        ret, im_th = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY_INV)

        # Find the connected region with skimage.measure.label
        # This function will set all each connected region to the same value, each region has a unique value
        connected_regions, num_of_labels = skimage.measure.label(im_th, connectivity=5, neighbors=8, background=0, return_num=True)
        print(str(num_of_labels)+" regions found in "+file_name)

        if self.show_imgs:
            plt.imshow(im_th)
            plt.title("threshold blurred image: " + file_name)
            plt.show()
            print(connected_regions)
            plt.imshow(connected_regions)
            plt.title(str(num_of_labels)+" connected regions")
            plt.show()
        for i in range(1, num_of_labels + 1):
            testimage = (connected_regions == i) * 200
            if self.show_imgs:
                plt.imshow(testimage)
                plt.title("symbol "+str(i))
                plt.show()
            x1, y1, x2, y2, symbol_img = self.trimZeros(testimage)
            cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 1)
            rec_string = str(y1) + '_' + str(y2) + '_' + str(x1) + '_' + str(x2)
            # MARK: IO
            if not os.path.exists("symbols"):
                os.makedirs("symbols")
            saved_name = "symbols/" + file_name.replace('.png', '_unclassified_' + rec_string + '.png')

            # MARK: Double check to see if there are inner image of the given region
            # This is to deal with the square root
            trimmed_img = im_th[y1:y2, x1:x2]
            inners, numer_of_inner = skimage.measure.label(trimmed_img, connectivity=5,
                                                           neighbors=8, background=0, return_num=True)
            if numer_of_inner == 1:
                if (y2-y1)*(x2-x1) > self.ignored_pixel_limit:
                    if self.show_imgs:
                        plt.imshow(raw_image[y1:y2, x1:x2])
                        plt.title("symbol " + str(i) + " clopped from raw image")
                        plt.show()

                    cv2.imwrite(saved_name, raw_image[y1:y2, x1:x2])
            else:
                print("found "+str(numer_of_inner)+" regions in this symbol, which means it has inner symbol(s)")
                temp_x1 = 0
                temp_y1 = 0
                temp_x2 = 0
                temp_y2 = 0
                temp_image = []
                is_first = True
                for j in range(1, numer_of_inner + 1):
                    inner = (inners == j) * 200
                    if self.show_imgs:
                        plt.imshow(inner)
                        plt.title("inner " + str(j))
                        plt.show()
                    x1_in, y1_in, x2_in, y2_in, trimmed_inner = self.trimZeros(inner)
                    if self.show_imgs:
                        plt.imshow(trimmed_inner)
                        plt.title("trimmed inner " + str(j))
                        plt.show()
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
                if (temp_x2-temp_x1)*(temp_y2-temp_y1) > self.ignored_pixel_limit:
                    if self.show_imgs:
                        print("save outer image " + saved_name + ":")
                        print(temp_image)
                        print("\n")
                        plt.imshow(numpy.array(temp_image))
                        plt.title("outer in symbol " + str(i))
                        plt.show()
                    cv2.imwrite(saved_name, numpy.array(temp_image))
        if not os.path.exists("boxed"):
            os.makedirs("boxed")
        boxed_img_path = "boxed/" + file_name
        if self.show_imgs:
            plt.imshow(img_with_box)
            plt.title(boxed_img_path)
            plt.show()
        cv2.imwrite(boxed_img_path, img_with_box)

    @staticmethod
    def clear(folder_name):  # delete all files in given dir, if the dir does not exist, create it
        path_name = "./"+folder_name
        print("clear all files in folder: "+folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            for previously_saved_symbol in [f for f in listdir(path_name)]:
                os.remove(path_name+"/"+previously_saved_symbol)

    def __init__(self):
        if self.loading_imgs:
            print("start segmenting equation pics in annotated folder")
            files = [f for f in listdir('./annotated')]
            self.clear("boxed")
            self.clear("symbols")
            for f in files:
                a = f.split("_")
                if len(a) == 3:
                    print(f)
                    self.segment('./annotated/', f)
        else:
            print("segmenting done, start classification directly")