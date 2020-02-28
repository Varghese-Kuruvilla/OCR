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

#Import utils
import debug_utils.utils as utils

#pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Emil\\AppData\\Local\\Tesseract-OCR\\tesseract.exe' 

class ocrutils:
    '''Class to do basic image preprocessing and perform OCR using tesseract'''
    def __init__(self):
        self.img = None
        self.gray_img = None
        self.rx_dict = {'name':re.compile(r'(?P<name>nam.*)'),
        'pan_no':re.compile(r'(?P<pano_no>pan.*)'),
        'father_name':re.compile(r'(?P<father_name>father.*)'),
        'relationship':re.compile(r'(?P<relationship>relation.*)'),
        'residential_addr':re.compile(r'(?P<residential_addr>resident.*)'),
        'period_stay':re.compile(r'(?P<period_stay>period.*)'),
        'tel_no':re.compile(r'(?P<tel_no>tel.*)'),
        'mobile_no':re.compile(r'(?P<mobile_no>mob.*)'),
        'email':re.compile(r'(?P<email>e.*ai.*)')}
        self.parse_dict = {'name':[],'pan_no':[],'father_name':[],'relationship':[],'residential_addr':[],'period_stay':[],'tel_no':[],
                            'mobile_no':[],'email':[]}
        #TODO:See if there is a better way to do this
        self.ret = False
        self.key = None

    def load_img(self):
        '''Function to load the image corresponding to the path defined by the user'''
        ap = argparse.ArgumentParser()
        ap.add_argument("-i","--image",required=True,help = "Path to input image")
        args = vars(ap.parse_args())

        #Load Image
        self.img = cv2.imread(args["image"])
        self.extract_table()
        # self.preprocess_img()
        # self.run_tesseract()

    def extract_table(self):
        '''
        Function for table extraction from a loan document
        '''

        #Image for drawing contours
        cnt_img = np.copy(self.img)

        #Local variables
        box_coord_ls = []
        merged_coord_ls = []
        threshold = 10
        flag = False

        display("Input image",self.img)

        self.gray_img = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        display("Grayscale image",self.gray_img)

        #Assume that the paper is white and the ink is black. 
        #TODO: The threshold value of 100 shouldn't be hardcoded
        _,thresh_img = cv2.threshold(self.gray_img,240,255,cv2.THRESH_BINARY_INV)
        display("Thresholded image",thresh_img)

        #Remove noise from the image
        # kernel = np.ones((5,5),np.uint8)
        # thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_GRADIENT, kernel)
        # display("thresh_img",thresh_img)


        #Adaptive Thresholding
        # thresh_img = cv2.adaptiveThreshold(self.gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        # utils.display("Adaptive Thresholding",thresh_img)

        #Finding Contours on the image and extracting the largest one
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        large_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1] #Extract only the table
        # print("len(contours)",len(contours))

        table_mask = np.zeros((self.img.shape[0],self.img.shape[1]),dtype = np.uint8)
        table_mask = cv2.drawContours(table_mask,large_contour,0,(255,255,255),-1)
        display("table_mask",table_mask)

        table_img = cv2.bitwise_and(thresh_img,thresh_img,mask=table_mask)
        display("table_img",table_img)


        #Now we find each contour within the table and give it as input to OCR
        contours, hierarchy = cv2.findContours(table_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        #TODO: Don't hardcode the index 1:35
        cell_contours = sorted(contours,key=cv2.contourArea,reverse=True)[1:35]

        for cnt in cell_contours:
            x,y,w,h = cv2.boundingRect(cnt)

            # if(w >= 300 and w <= 400):
            if(w >= 300):
                #Merge neighbouring contours
                box_coord = np.array([x,y,w,h]).reshape(4,-1)
                print("box_coord:",box_coord)
                

                for i in range(0,len(box_coord_ls)):
                # for box_coord_ls[i] in box_coord_ls:
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
                    # print("box_coord:",box_coord)
                flag = False
        
        #Draw rectangles
        cnt_img = np.copy(self.img)
        # display("Input image",self.img)
        for coord in box_coord_ls:
            ocr_img = self.img[coord[1,0]:coord[1,0] + coord[3,0],coord[0,0]:coord[0,0] + coord[2,0]]
            # display("OCR_Image",ocr_img)
            self.preprocess_img(ocr_img)
            # cv2.rectangle(cnt_img,(coord[0,0],coord[1,0]),(coord[0,0] + coord[2,0],coord[1,0] + coord[3,0]),(0,0,255),5)
        # display("Contour Image",cnt_img) 
        #Save the final dictionary as a CSV file

        #Error Checking
        # for key,values in self.parse_dict.items():
        #     if(len(values) < 4):
        #         ls_to_append = ['nan'] * (2 - len(values))
        #         self.parse_dict[key].extend(ls_to_append)pd.DataFrame.from_dict(data_dict,orient='index').T.dropna()
        
        print("self.parse_dict",self.parse_dict)
        utils.breakpoint()

        # df = pd.DataFrame(self.parse_dict)
        df = pd.DataFrame.from_dict(self.parse_dict,orient='index').T.dropna()
        df.to_csv('/home/varghese/Nanonets/OCR/csv_files/'+ str(os.getpid()) + ".csv")       

    def preprocess_img(self,cnt_img):
        '''
        Function to carry out preprocessing, call the function to run OCR using tesseract
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
            self.counter = (i + 1)
            self.run_tesseract(box_img)
        

        #Adaptive Thresholding
        # adaptive_img = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        # utils.display("Adaptive Thresholding",adaptive_img)
        

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
        
        self.parse_output()


    def parse_line(self,line):
        '''
        Function to parse a line of text using the compiled regular expression
        '''
        for key,rx in self.rx_dict.items():
            match = rx.search(line)
            if(match):
                return key,True
        
        #No matches
        return False,False


    def parse_output(self):
        '''
        Function to parse the text file using regex and save the output in a database
        '''
        f = open(str(os.getpid()) + ".txt","r")

        for line in f:
            line = line.strip()            
            line = line.lower()
            # print("line:",line)
            # print("self.counter:",self.counter)
            if(self.counter > 1):
                if(self.ret == True):
                    self.parse_dict[str(self.key)].append(str(line))
                    continue

            self.key,self.ret = self.parse_line(line)
            if(self.ret == True): #If the cell contains more than one line of text, break as soon as we find the first match
                break

        print("self.parse_dict:",self.parse_dict)


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
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        sys.exit()

if __name__ == '__main__':
    ocrutils_obj = ocrutils()
    ocrutils_obj.load_img()
