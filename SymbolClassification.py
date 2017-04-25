import LocalHandwrittenSymbolDataset
import SymbolSegmentor
from os import listdir
import tensorflow as tf
import argparse
import sys
import numpy


class ClassificationDictionary:

    classification_dict = dict()

    def __init__(self):
        files = [f for f in listdir('./annotated')]
        classification = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])

        for f in files:
            a = f.split("_")
            if len(a) == 8:
                print(f)
                print(a[3])
                classification.add(a[3])
        sorted_classification = numpy.array(sorted(classification))
        print("xxx")
        print(sorted_classification)
        print("there are %d kinds of symbols" % len(classification))

        for i in range(0, len(classification)):
            self.classification_dict[sorted_classification[i]] = i

        print(self.classification_dict)

    def get_classification_array(self, symbol):
        if len(self.classification_dict) == 0:
            print("You cannot call get_classification_array(symbol) for now, \n the classification_dict is empty.")
            return numpy.array([])
        zeros_array = numpy.zeros(len(self.classification_dict))
        index = self.classification_dict[symbol]
        if index is None:
            print("cannot identify given symbol " + symbol)
            print("return all %d zeros" % len(self.classification_dict))
            return zeros_array
        else:
            zeros_array[index] = 1.0
            return zeros_array




def main(_):

    classficationDic = ClassificationDictionary()
    localData = LocalHandwrittenSymbolDataset.LocalSymbolData(classficationDic)
    testing_data = SymbolSegmentor.SymbolSegmentor()


    #TODO: DEFINE MODEL

    #TODO: TRAIN

    #TODO: TEST



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)