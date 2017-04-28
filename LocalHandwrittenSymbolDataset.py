import os
import skimage.io
import skimage.transform
import skimage.util
import SymbolClassification
from PIL import Image
from PIL import ImageFilter
import numpy as np
from os import listdir
import random


class LocalSymbolData:
    training_imgs = []
    labels = []
    classificationDic = SymbolClassification.ClassificationDictionary('')
    ind = 0;


    def __init__(self, classification_dictionary):
        training_imgs = []
        labels = []
        self.classificationDic = classification_dictionary
        ind = 0
        print(self.classificationDic.get_classification_array("("))
        #get path
        print("hi")
        path1 = './annotated'
        files = [f for f in listdir(path1)]

        for f in files:
            # print(f)
            a = f.split("_")
            if len(a) == 8:
                # print(a)
                im = Image.open(path1 + '/' + f).convert('L')

                width = float(im.size[0])
                height = float(im.size[1])
                newIm = Image.new("L", (28, 28), (0))

                if width > height:
                    hei = int(round((20.0 / width * height), 0))
                    if hei == 0:
                        hei = 1
                    img = im.resize((20, hei), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                    top = int(round(((28 - hei) / 2), 0))
                    newIm.paste(img, (4, top))
                else:
                    wid = int(round((20.0 / height * width), 0))
                    if wid == 0:
                        wid = 1
                    img = im.resize((wid, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                    lef = int(round(((28 - wid) / 2), 0))
                    newIm.paste(img, (lef, 4))

                img = np.reshape(newIm, (1, 784))
                training_imgs.append(img)
                # print(a[3])
                labels.append(self.classificationDic.get_classification_array(a[3]))
                ind = ind + 1;


        # print(self.classificationDic.get_classification_array("y"))
        # eg. we have a b c, 3 symbols. Calling get_classification_array("b") will return [0, 1, 0]

        # TODO: READ ALL PICTURES OF SYMBOLS FROM FOLDER /annotated

        # TODO: CONVERT ALL IMGS TO 28*28 AND PLATTEN THEM to 1*784 like PA4, PUT ALL LABELS IN self.labels
        # for given index images in self.training_imgs should match with the label in self.label

    def read(raw_images):
        print("hi2")

        #TODO: Read a list of imgs in any resolution, return resized imgs with demension of 1*784
        return []


    def next_batch(self,n):
        training_imgs = self.training_imgs
        labels = self.labels
        ind = self.ind
        # x = np.array()
        # y = np.array()
        # for num in range(1,n):
        # z = np,array([x,y])

        x = []
        y = []
        for num in range(0,n):
            if ind == 0:
                break;
            x.append(training_imgs[num]);
            y.append(labels[num])
            ind = ind - 1;
        z = [x,y]
        print(z)
        #TODO: This should do the same thing as train.next_batch(x) in tensorflow
        #TODO: Return a array of 2 element, the first should be the NEXT n images in self.training_imgs
        # , the second should be corresponding labels
        return z

    # useful helper methods
    def dilate(img, radius):
        from skimage.morphology import square
        from skimage.morphology import dilation
        return dilation(img, selem=square(radius))

    def make_square_img(img_arr):
        (vertical_pixel, horizontal_pixel) = img_arr.shape
        if vertical_pixel > horizontal_pixel:
            vertical_padding = int(round(vertical_pixel * 0.15))
            horizontal_padding = int(round((vertical_pixel * 1.3 - horizontal_pixel) / 2))
            padding = ((vertical_padding, vertical_padding), (horizontal_padding, horizontal_padding))
            return skimage.util.pad(img_arr, padding, 'constant', constant_values=0)
        else:
            horizontal_padding = int(round(horizontal_pixel * 0.15))
            vertical_padding = int(round((horizontal_pixel * 1.3 - vertical_pixel) / 2))
            padding = ((vertical_padding, vertical_padding), (horizontal_padding, horizontal_padding))
            return skimage.util.pad(img_arr, padding, 'constant', constant_values=0)

    def filter_zero_or_one(img_arr):
        filter_threshold = 0.145
        [vert, hori] = img_arr.shape
        for i in range(vert):
            for j in range(hori):
                if img_arr[i, j] > filter_threshold:
                    img_arr[i, j] = 1
                else:
                    img_arr[i, j] = 0
        return img_arr

