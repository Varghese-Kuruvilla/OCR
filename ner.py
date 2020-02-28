#Script to implement NER 
import os
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import pytesseract
from PIL import Image
import spacy
from spacy import displacy

from debug_utils import utils

class nerutils:
    '''
    Class to implement NER on the text input
    '''

    def process_img(self,img):
        '''
        Extract text, call OCR and pass the output to the spacy

        Attributes
        -------------
        img: Numpy array
            Image to be processed
        '''
        utils.display("Image",img)
        
        #TODO: Try to remove hardcoding
        crop_img = img[:,396:img.shape[1]] 
        utils.display("Cropped Image",crop_img)

        crop_img = np.bitwise_not(crop_img)
        utils.display("Cropped Image",crop_img)

        #Run tesseract
        self.run_tesseract(crop_img)


    def run_tesseract(self,ocr_img):
        '''
        Function to run OCR using tesseract on ocr_img

        Parameters

        ---------------

        ocr_img: Numpy array
                Preprocessed image on which OCR is run using tesseract
        '''

        # display("Image before passing to tesseract",ocr_img)

        # Define config parameters.
	    # '-l eng'  for using the English language
	    # '--oem 1' for using LSTM OCR Engine
        config = ('-l eng --oem 3')

        filename = "{}.png".format(os.getpid())
        cv2.imwrite(str(filename),ocr_img)
        print("Image saved to disk")

        #Load image from disk and apply run_tesseract
        text = pytesseract.image_to_string(Image.open(filename),config=config)
        #Save text to a file
        f = open(str(os.getpid()) + ".txt","w")
        f.write(text)
        f.close()
        print("Written into file")

        #Remove the image file
        if os.path.isfile(str(filename)):
            os.remove(str(filename))
        else: 
            print("Error: %s file not found" % myfile)

        #Reading the contents of the text file
        f = open(str(os.getpid()) + ".txt","r")
        # text = f.read()

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        displacy.serve(doc, style="ent")

        #For debug
        # nlp = spacy.load("en_core_web_sm")
        # for line in f:
        #     line = line.strip()
        #     print("line:",line)
        #     doc = nlp(line)
        #     ents = list(doc.ents)
        #     for i in range(0,len(ents)):
        #         print("{},{}".format(ents[i].text,ents[i].label_))
        #         utils.breakpoint()
            # print("doc.ents",doc.ents)


       
        
            



#For debug
if __name__ == '__main__':
    nerutils_obj = nerutils()
    img = cv2.imread("/home/varghese/Nanonets/OCR/images/debug.jpg")
    nerutils_obj.process_img(img)