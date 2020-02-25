import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np 
import os
from PIL import Image
import pytesseract
import argparse
import sys
import matplotlib.pyplot as plt

#Import utils
import debug_utils.utils as utils

#pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Emil\\AppData\\Local\\Tesseract-OCR\\tesseract.exe' 

class ocrutils:
    '''Class to do basic image preprocessing and perform OCR using tesseract'''
    def __init__(self):
        self.img = None
        self.gray_img = None

    def load_img(self):
        '''Function to load the image corresponding to the path defined by the user'''
        ap = argparse.ArgumentParser()
        ap.add_argument("-i","--image",required=True,help = "Path to input image")
        args = vars(ap.parse_args())

        #Load Image
        self.img = cv2.imread(args["image"])
        display("Input image",self.img)
        self.extract_table()
        self.preprocess_img()
        self.run_tesseract()

    def extract_table(self):
        '''
        Function for table extraction from a loan document
        '''
        display("Input image",self.img)
        self.gray_img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        display("Grayscale image",self.gray_img)

        #Assume that the paper is white and the ink is black. 
        _,thresh_img = cv2.threshold(self.gray_img,220,255,cv2.THRESH_BINARY_INV)
        print("thresh_img",thresh_img)
        display("Thresholded image",thresh_img)

        #Finding Contours on the image and extracting the largest one
        

    def preprocess_img(self):
        '''Preprocessing image for OCR
        Involves the following steps:
        1) Scaling to 300 DPI(Ideally)
        2) Increase contrast of the image
        3) Binarize the image
        4) Removing noise 
        5) Deskew the image (Correct for rotation)'''

        #Binarization using Otsu
        self.gray_img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        display("Grayscale image",self.gray_img)

        #Histogram for the grayscale image
#        hist,bin_edges = np.histogram(self.gray_img,bins=256,range=(0,1)) 
        #Using the histogram of the grayscale image explain why OTSU's doesn't work
        
        # hist,bin_edges = np.histogram(self.gray_img,bins=256,range=(0,256)) 
        # plt.title("Histogram of grayscale image")
        # plt.plot(hist)
        # plt.show()

        self.gray_img = cv2.blur(self.gray_img,(5,5))
        display("Blurred Image",self.gray_img)
        # ret,thresh_img = cv2.threshold(self.gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # print("Ret",ret)
        # utils.display("Otsu",thresh_img)

        #Adaptive Thresholding
        adaptive_img = cv2.adaptiveThreshold(self.gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        utils.display("Adaptive Thresholding",adaptive_img)
        

    def run_tesseract(self):
        '''Function to run OCR using tesseract on self.img'''
        #Conversion to grayscale
        img_gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        display("Grayscale Image",img_gray)
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(str(filename),img_gray)
        print("Image saved to disk")

        #Load image from disk and apply run_tesseract
        text = pytesseract.image_to_string(Image.open(filename))
        #Save text to a file
        f = open("ocr_cheque.txt","w")
        f.write(text)
        f.close()
        print("Written into file")

def breakpoint():
    inp = input("Waiting for input...")

def display(txt,img):
    '''Utility function to display an image with window name txt'''
    winname = txt
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,img)
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        sys.exit()

if __name__ == '__main__':
    ocrutils_obj = ocrutils()
    ocrutils_obj.load_img()
