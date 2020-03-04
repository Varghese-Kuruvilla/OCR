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
import re
import pandas as pd
import csv

#Import utils
#import debug_utils.utils as utils
sys.path.append('/home/varghese/Nanonets/OCR')
from code_testing.ner import nerutils

#For debug
import time
 

class ocrutils:
    '''Class to do basic image preprocessing and perform OCR using tesseract'''
    def __init__(self):
        self.img = None
        self.gray_img = None
        self.rx_dict = {'name':re.compile(r'(?P<name>^nam.*)', flags=re.IGNORECASE),
        'pan_no':re.compile(r'(?P<pano_no>pan.*)',flags=re.IGNORECASE),
        'father_name':re.compile(r'(?P<father_name>father.*)',flags=re.IGNORECASE),
        'relationship':re.compile(r'(?P<relationship>relation.*)',flags=re.IGNORECASE),
        'residential_addr':re.compile(r'(?P<residential_addr>resident.*)',flags=re.IGNORECASE),
        'period_stay':re.compile(r'(?P<period_stay>period.*)',flags=re.IGNORECASE),
        'tel_no':re.compile(r'(?P<tel_no>tel.*)',flags=re.IGNORECASE),
        'mobile_no':re.compile(r'(?P<mobile_no>mob.*)',flags=re.IGNORECASE),
        'email':re.compile(r'(?P<email>e.*ai.*)',flags=re.IGNORECASE)}
        self.parse_dict = {'name':[],'pan_no':[],'father_name':[],'relationship':[],'residential_addr':[],'period_stay':[],'tel_no':[],
                            'mobile_no':[],'email':[]}
        #TODO:See if there is a better way to do this
        self.ret = False
        self.key = None

    def load_img(self):
        '''Function to load the image and call the function for extracting the required ROI'''

        #Load Image
        self.img = cv2.imread('../input_image/Loan_application_form_digital_v2.jpg')
        # self.img = cv2.imread('/home/varghese/Nanonets/OCR/images/Loan_application_scanned.jpg')
        # display("Image",self.img)
        self.extract_table()

    def extract_table(self):
        '''
        Function for extracting the ROI(table) from the document
        '''

        #Image for drawing contours
        #For debug
        cnt_img = np.copy(self.img)

        #Local variables
        box_coord_ls = []
        merged_coord_ls = []
        threshold = 10
        flag = False

        # display("Input image",self.img)

        self.gray_img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        # display("Grayscale image",self.gray_img)

        #Assume that the paper is white and the ink is black. 
        _,thresh_img = cv2.threshold(self.gray_img,240,255,cv2.THRESH_BINARY_INV)
        # display("Thresholded image",thresh_img)

        #Finding Contours on the image and extracting the largest one, this corresponds to our ROI(table)
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        large_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1] #Extract only the table
        # print("len(contours)",len(contours))

        #Creating a mask of the ROI
        table_mask = np.zeros((self.img.shape[0],self.img.shape[1]),dtype = np.uint8)
        table_mask = cv2.drawContours(table_mask,large_contour,0,(255,255,255),-1)
        # display("table_mask",table_mask)
      

        table_img = cv2.bitwise_and(thresh_img,thresh_img,mask=table_mask)
        # display("table_img",table_img)

        #Performing NER at this stage
        # nerutils_obj.process_img(table_img)

        #Extract each individual contour from the table and merge neighbouring contours
        contours, hierarchy = cv2.findContours(table_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cell_contours = sorted(contours,key=cv2.contourArea,reverse=True)[1:35]

        for cnt in cell_contours:
            x,y,w,h = cv2.boundingRect(cnt)

            # if(w >= 300 and w <= 400):
            if(w >= 300):
                #Merge neighbouring contours
                box_coord = np.array([x,y,w,h]).reshape(4,-1)
                print("box_coord:",box_coord)
                

                for i in range(0,len(box_coord_ls)):
                    if(abs((box_coord_ls[i][0,0] + box_coord_ls[i][2,0]) - box_coord[0,0]) < threshold and abs(box_coord_ls[i][1,0] - box_coord[1,0]) < threshold):
                        flag = True
                        xmin = box_coord_ls[i][0,0]
                        ymin = box_coord_ls[i][1,0]
                        xmax = box_coord_ls[i][0,0] + box_coord_ls[i][2,0] + box_coord[2,0]
                        ymax = box_coord_ls[i][1,0] + box_coord_ls[i][3,0]

                        element = np.array([xmin,ymin,xmax-box_coord_ls[i][0,0],ymax-box_coord_ls[i][1,0]]).reshape(4,-1)
                        box_coord_ls[i] = element
                        merged_coord = np.array([xmin,ymin,xmax,ymax]).reshape(4,-1)
                        merged_coord_ls.append(merged_coord)

                if(flag == False): 
                    box_coord_ls.append(box_coord)
                flag = False
        
        #Draw rectangles
        cnt_img = np.copy(self.img)
        # display("Input image",self.img)
        for coord in box_coord_ls:
            ocr_img = self.img[coord[1,0]:coord[1,0] + coord[3,0],coord[0,0]:coord[0,0] + coord[2,0]]
            # display("OCR_Image",ocr_img)

            #Preprocess the image before passing it to Tesseract
            self.preprocess_img(ocr_img)
            # cv2.rectangle(cnt_img,(coord[0,0],coord[1,0]),(coord[0,0] + coord[2,0],coord[1,0] + coord[3,0]),(0,0,255),5)
        # display("Contour Image",cnt_img)


        #Save the final dictionary as a CSV file       
        # print("self.parse_dict",self.parse_dict)
        #Check using NER
        nerutils_obj.check_ocr(self.parse_dict)
        # utils.breakpoint()

        #Writing self.parse_dict into a CSV file
        # with open('../csv_files/'+ str(os.getpid()) + '.csv','w') as csvfile:
        #     fieldnames = list(self.parse_dict.keys())
        #     writer = csv.DictWriter(csvfile,fieldnames=fieldnames)

        #     writer.writeheader()
        #     writer.writerow(self.parse_dict)


    def preprocess_img(self,cnt_img):
        '''
        Function to carry out preprocessing, before passing the image to Tesseract
        Preprocessing image for OCR
        Involves the following steps:
        1) Scaling to 300 DPI(Ideally)
        2) Increase contrast of the image
        3) Binarize the image
        4) Removing noise 
        5) Deskew the image (Correct for rotation)
        
        Parameters

        --------------------

        cnt_img: Numpy array
                Image cropped to an individual contour which is to be preprocessed
        '''
        self.counter = 0 #Counter keeps track of which picture is passed to OCR
        self.ret = False
        self.key = None 
        #Conversion to grayscale
        gray_img = cv2.cvtColor(cnt_img,cv2.COLOR_BGR2GRAY)
        # display("Grayscale image",gray_img)
        
        #Thresholding the image
        _,thresh_img = cv2.threshold(gray_img,230,255,cv2.THRESH_BINARY)
        # display("thresh_img",thresh_img)

        #Split the entire image into individual boxes and pass each one to run_tesseract
        coord_ls = [0,362,693,thresh_img.shape[1]]
        for i in range(0,len(coord_ls)-1):
            box_img = thresh_img[:,coord_ls[i]:coord_ls[i+1]]
            #For debug
            # display("box_img",box_img)
            self.counter = (i + 1)
            self.run_tesseract(box_img)
        
        

    def run_tesseract(self,ocr_img):
        '''
        Function to run OCR using tesseract on ocr_img

        Parameters

        ---------------

        ocr_img: Numpy array
                Preprocessed image which is passed to Tesseract
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
        print("Output written into text file")

        #For debug
        # f = open(str(os.getpid()) + ".txt","r")
        # str_read = f.read()
        # print("str_read:",str_read)

        #Remove the image file
        if os.path.isfile(str(filename)):
            os.remove(str(filename))
        else: 
            print("Error: %s file not found" % myfile)
        
        self.parse_output()


    def parse_line(self,line):
        '''
        Function to parse a line of text against the compiled regular expression

        Parameters

        ---------------------

        line: String
            Single line of text from the text file(containing OCR output)
        '''
        for key,rx in self.rx_dict.items():
            match = rx.search(line)
            if(match):
                return key,True
        
        #No matches
        return False,False


    def parse_output(self):
        '''
        Function to parse the text file using regex
        '''
        f = open(str(os.getpid()) + ".txt","r")


        # for line in f:
        line = f.read()
        line = line.strip()            
        # line = line.lower()
        line_result = re.findall(r"[0-9a-zA-Z ]+|[0-9a-zA-Z]", str(line))
        line_result = ' '.join(map(str, line_result)) 
        print("line_result:",line_result)
        # breakpoint()
        # print("self.counter:",self.counter)
        if(self.counter > 1):
            if(self.ret == True):
                self.parse_dict[str(self.key)].append(str(line_result))

        else:        
            self.key,self.ret = self.parse_line(line_result)

        f.close()
        # print("self.parse_dict",self.parse_dict)


        #Delete txt files after parsing them
        if(os.path.isfile(str(os.getpid()) + ".txt")):
            os.remove(str(os.getpid()) + ".txt")
        else:
            print("Text file not found")
        
        

def breakpoint():
    inp = input("Waiting for input...")

def display(txt,img):
    '''Utility function to display an image with window name txt'''
    winname = txt
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,img)
    key = cv2.waitKey(1)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        sys.exit()

if __name__ == '__main__':
    
    ocrutils_obj = ocrutils()
    nerutils_obj = nerutils()
    #Main Function
    ocrutils_obj.load_img()
