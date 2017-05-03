
import LocalHandwrittenSymbolDataset
import SymbolSegmentor
from os import listdir
import skimage
import skimage.util
import skimage.transform
from skimage import exposure
import tensorflow as tf
import argparse
import sys
import numpy
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import SymbolClassification

def main(_):
    classficationDic = SymbolClassification.ClassificationDictionary('./annotated')
    localData = LocalHandwrittenSymbolDataset.LocalSymbolData(classficationDic)
    for i in range(150):
        print("-------------")
        localData.next_batch(50)
        print("-------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)