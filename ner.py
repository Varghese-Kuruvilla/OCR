#Script to implement NER 
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import pytesseract
from PIL import Image
import spacy
from spacy.matcher import Matcher
from spacy import displacy

sys.path.append('/home/varghese/Nanonets/OCR/code')
from debug_utils import utils
import pickle
import re

class nerutils:
    '''
    Class to implement NER on the text input
    '''

    def __init__(self):

        self.nlp = spacy.load("en_core_web_sm")
        #Creating a matcher object
        self.matcher = Matcher(self.nlp.vocab)

        person_label = self.nlp.vocab.strings["PERSON"]
        date_label = self.nlp.vocab.strings["DATE"]
        country_label = self.nlp.vocab.strings["GPE"]

        self.pattern_name = [{"ENT_TYPE":person_label}]
        self.pattern_period_stay = [{"ENT_TYPE":date_label}]
        self.pattern_residential_addr = [{"ENT_TYPE":country_label}]
        # self.pattern_pan_no = [{"TEXT": {"REGEX": "^([a-zA-Z]){5}([0-9]){4}([a-zA-Z]){1}?$"}}]
        #TODO: Add entities like brother-in-law , sister-in-law and so on
        self.pattern_relationship = ["wife","husband","father","mother","grandfather","grandmother","brother","sister","uncle","aunt"] 
        self.pattern_mobile_no = [{"ORTH": "Hello"}] #TODO: Improve this regex
        # self.pattern_mobile_no = [{"TEXT": {"REGEX": r'^\+\d{2}\d{10}'}}] #TODO: Improve this regex


        self.dict_pattern = {'name':self.pattern_name,'pan_no':[],'father_name':self.pattern_name,'relationship':[],'residential_addr':self.pattern_residential_addr,'period_stay':self.pattern_period_stay,'tel_no':[],
                            'mobile_no':[],'email':[]}
        
        self.dict_regex = {'pan_no':re.compile(r'(?P<pan_no>^([a-zA-Z]){5}([0-9]){4}([a-zA-Z]){1}?$)'),
                           "mobile_no":re.compile(r'(?P<mobile_no>^91[-\s]??\d{10}$)'),
                           "email":re.compile(r'(?P<email>^([a-zA-Z0-9_\-\.]+)(@\s)?([a-zA-Z0-9_\-\.]+)(\.\s)?([a-zA-Z]{2,5})$)')}

        
        self.dict_cond = {}
        self.key = None

    def for_debug(self,matcher,doc,i,matches):
        
        self.dict_cond[self.key] = True
        print("doc",doc)
        # utils.breakpoint()
        # for match_id, start, end in matches[i]:
        #     span = doc[start:end]
        #     print("span.text",span.text)

    def check_ocr(self,dict_ocr):
        '''
        Function to check the output of OCR using rule matching and NER
        '''
       
        #For debug(Changing specific values in dict_ocr)
        dict_ocr['name'] = ['Mr Peter Brown','Mrs. Susan Brown']
        dict_ocr['residential_addr'] = ['NO 78, Downing Street West Sussex, England','NO 78, Downing Street West Sussex, England']
        dict_ocr['relationship'] = ['husband','wife']
        print("dict_ocr:",dict_ocr)

        for self.key, value_ls in dict_ocr.items():
            if(self.key == 'relationship'): #Here we simply carry out phrase matching using a dictionary
                print("dict_ocr[relationship]",dict_ocr['relationship'])
                for value in value_ls:
                    value = value.lower() #Converting to lowercase
                    if(value in self.pattern_relationship):
                        self.dict_cond[self.key] = True

            if(self.key == 'name' or self.key == 'period_stay' or self.key == 'residential_addr'
             or self.key == 'father_name'):
                print("self.key",self.key)
                self.matcher.add(str(self.key),self.for_debug,self.dict_pattern[self.key])
            
            if(self.key == 'pan_no' or self.key == 'mobile_no' or self.key == 'email'):
                for value in value_ls:
                    value = value.lower()
                    if(self.key == 'email'):
                        print("value",value)
                    match = self.dict_regex[self.key].search(value)
                    if(match != None):
                        self.dict_cond[self.key] = True

                


            if(self.matcher):
                for value in value_ls:
                    print("value:",value)
                    doc = self.nlp(value)
                    #For debug
                    # ents = list(doc.ents)
                    # print("ents",ents)
                    # for i in range(0,len(ents)):
                    #     print("{},{}".format(ents[i].text,ents[i].label_)) 
                    print([t.text for t in doc])

                    matches = self.matcher(doc)
                    print("matches:",matches)


                if("name" or "pan_no" or "period_stay" or "residential_addr" or "father_name" or "mobile_no" in self.matcher):
                    print("Inside condition make self.matcher=None")
                    print("self.key",self.key)
                    on_match, patterns = self.matcher.get(self.key)
                    print("patterns:",patterns)
                    utils.breakpoint()
                    # print("self.matcher contains name")
                    self.matcher = None
                    #Initialize self.matcher object again
                    self.matcher = Matcher(self.nlp.vocab)

        print("self.dict_cond",self.dict_cond)
                

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
        crop_img = img[:,624:img.shape[1]] 
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
        # if os.path.isfile(str(filename)):
        #     os.remove(str(filename))
        # else: 
        #     print("Error: %s file not found" % myfile)

        #Reading the contents of the text file
        # f = open(str(os.getpid()) + ".txt","r")
        #For debug
        f = open("/home/varghese/Nanonets/OCR/code/debug_spacy.txt","r")
        # text = f.read()
        text = "My name is Matthew. I work in apple" #Sort of hopeless

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
        #     print("doc.ents",doc.ents)

    def compare_ner(self,txt_file_path):
        '''
        Function to compare the performance of different NER models

        Parameters
        ------------------
        txt_file_path: Path of the text file containing text for NER

        '''
        #Performance of Spacy
        use_spacy = True
        if(use_spacy == True):
            nlp = spacy.load("en_core_web_sm")
        
            #Creating a matcher object
            matcher = Matcher(nlp.vocab)
            date_label = nlp.vocab.strings["DATE"]
            print("date_label",date_label)
            # pattern = [{"SHAPE": "dd","ENT_TYPE":date_label}, {"SHAPE":"dddd","LENGTH":{"IN": [6,7,8]}}]
            pattern = [{"SHAPE": "dd","ENT_TYPE":date_label}, {"SHAPE":"dddd","LENGTH": 6,"LENGTH":7,"LENGTH":8,"ENT_TYPE":date_label}]

            matcher.add("match_date",self.for_debug,pattern)
            f = open(str(txt_file_path),"r")
            for line in f:
                line = line.strip()
                print("line:",line)
                doc = nlp(line)
                print([t.text for t in doc])
                #For debug
                matches = matcher(doc)
                print("matches:",matches)
                ents = list(doc.ents)
                for i in range(0,len(ents)):
                    print("{},{}".format(ents[i].text,ents[i].label_))
                    utils.breakpoint()

    



#For debug
if __name__ == '__main__':
    nerutils_obj = nerutils()
    # img = cv2.imread("/home/varghese/Nanonets/OCR/images/Loan_application_scanned.jpg")
    # nerutils_obj.process_img(img)
    # nerutils_obj.compare_ner("/home/varghese/Nanonets/OCR/code_testing/compare_ner_testbench.txt")
    # nerutils_obj.compare_ner("/home/varghese/Nanonets/OCR/code_testing/test_spacy_address.txt")

    #Reading data from a pickle file
    pickle_in = open("/home/varghese/Nanonets/OCR/code/pickle_files/dict_ocr.pickle","rb")
    dict_ocr = pickle.load(pickle_in)
    nerutils_obj.check_ocr(dict_ocr)