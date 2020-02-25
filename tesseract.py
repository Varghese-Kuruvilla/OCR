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
        self.extract_table()
        self.preprocess_img()
        self.run_tesseract()

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
        _,thresh_img = cv2.threshold(self.gray_img,220,255,cv2.THRESH_BINARY_INV)
        display("Thresholded image",thresh_img)

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
        cell_contours = sorted(contours,key=cv2.contourArea,reverse=True)[:30]

        for cnt in cell_contours:
            x,y,w,h = cv2.boundingRect(cnt)

            #Merge neighbouring contours
            box_coord = np.array([x,y,w,h]).reshape(4,-1)
            print("box_coord:",box_coord)
            

            #Compare 
            for i in range(0,len(box_coord_ls)):
            # for box_coord_ls[i] in box_coord_ls:
                if(abs((box_coord_ls[i][0,0] + box_coord_ls[i][2,0]) - box_coord[0,0]) < threshold and abs(box_coord_ls[i][1,0] - box_coord[1,0]) < threshold):
                    flag = True
                    print("Inside true condition")
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
                print("box_coord:",box_coord)
            flag = False
        
        #Draw rectangles
        for coord in box_coord_ls:
            cv2.rectangle(cnt_img,(coord[0,0],coord[1,0]),(coord[0,0] + coord[2,0],coord[1,0] + coord[3,0]),(0,0,255),5)
            display("Contour Image",cnt_img)
        
        
            #Filtering out false contours based on length of the rectangle
            # if(w >= 300 and w <= 400): #Assuming a controlled environment
            #if((w > 900 and w < 1050) and (h > 90 and h<130)):
                #For debug
                # cv2.drawContours(cnt_img, [cnt], 0, (0,255,0), 3)
                # cv2.rectangle(cnt_img,(x,y),(x+w,y+h),(0,255,0))
                # Pass this to OCR
            #    img_crop = self.img[y:y+h,x:x+w]
            #    display("img_crop",img_crop)
        #display("Contour Image",cnt_img)
        # breakpoint()


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
