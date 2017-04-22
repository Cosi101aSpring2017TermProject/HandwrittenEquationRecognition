#This class draw boxes in each symbols in handwritten equation
#output a list of images in raw resolution

#Possibly helpful reference: http://cs229.stanford.edu/proj2013/JimenezNguyen_MathFormulas_final_paper.pdf
#http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html

class SymbolSegmentor:
    raw_imgs = []
    img_boxes = []
    def __init__(self):
        raw_imgs = []
        img_boxes = []
        #TODO: read all equations in annotated folder and drawing boxes around them recode the position of the box in self.img_boxes
        #TODO: put all images of single symbol in raw_imgs, NO processing needed.

