import os
import skimage.io
import skimage.transform
import skimage.util
import SymbolClassification


class LocalSymbolData:
    training_imgs = []
    labels = []
    classificationDic: SymbolClassification.ClassificationDictionary = None

    def __init__(self, classification_dictionary: SymbolClassification.ClassificationDictionary):
        training_imgs = []
        labels = []
        self.classificationDic = classification_dictionary
        # print(self.classificationDic.get_classification_array("("))
        # print(self.classificationDic.get_classification_array("y"))
        # eg. we have a b c, 3 symbols. Calling get_classification_array("b") will return [0, 1, 0]

        # TODO: READ ALL PICTURES OF SYMBOLS FROM FOLDER /annotated

        # TODO: CONVERT ALL IMGS TO 28*28 AND PLATTEN THEM to 1*784 like PA4, PUT ALL LABELS IN self.labels
        # for given index images in self.training_imgs should match with the label in self.label

    def read(raw_images):
        #TODO: Read a list of imgs in any resolution, return resized imgs with demension of 1*784
        return []


    def next_batch(n):
        #TODO: This should do the same thing as train.next_batch(x) in tensorflow
        #TODO: Return a array of 2 element, the first should be the NEXT n images in self.training_imgs
        # , the second should be corresponding labels
        return []

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

