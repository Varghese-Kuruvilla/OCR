#Script to implement NER 
import os
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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


        #Patterns which are to be fed to the spacy matcher
        self.pattern_name = [{"ENT_TYPE":person_label}]
        self.pattern_period_stay = [{"ENT_TYPE":date_label}]
        self.pattern_residential_addr = [{"ENT_TYPE":country_label}]

        
        self.pattern_relationship = ["wife","husband","father","mother","grandfather","grandmother","brother","sister","uncle","aunt"] 


        self.dict_pattern = {'name':self.pattern_name,'pan_no':[],'father_name':self.pattern_name,'relationship':[],'residential_addr':self.pattern_residential_addr,'period_stay':self.pattern_period_stay,'tel_no':[],
                            'mobile_no':[],'email':[]}
        
        #Regex for parsing pan_no, mobile_no and email
        self.dict_regex = {'pan_no':re.compile(r'(?P<pan_no>^([a-zA-Z]){5}([0-9]){4}([a-zA-Z]){1}?$)'),
                           "mobile_no":re.compile(r'(?P<mobile_no>^91[-\s]??\d{10}$)'),
                           "email":re.compile(r'(?P<email>^([a-zA-Z0-9_\-\.]+)(@\s)?([a-zA-Z0-9_\-\.]+)(\.\s)?([a-zA-Z]{2,5})$)')}

        
        self.dict_cond = {}
        self.key = None

    def for_debug(self,matcher,doc,i,matches):
        '''
        Callback function for the matcher object
        Sets the value corresponding to self.key in self.dict_cond equal to True
        '''
        self.dict_cond[self.key] = True

    def check_ocr(self,dict_ocr):
        '''
        Function to check the output of OCR using rule matching and NER
        '''
       
        #For debug(Changing specific values in dict_ocr)
        # dict_ocr['name'] = ['Mr Peter Brown','Mrs. Susan Brown']
        # dict_ocr['residential_addr'] = ['NO 78, Downing Street West Sussex, England','NO 78, Downing Street West Sussex, England']
        # dict_ocr['relationship'] = ['husband','wife']
        print("dict_ocr:",dict_ocr)

        for self.key, value_ls in dict_ocr.items():

            #For keys in the if condition below we use simple regex based pattern matching
            if(self.key == 'relationship' or self.key == 'pan_no' or self.key == 'mobile_no' or self.key == 'email'):
                for value in value_ls:
                    value = value.lower() #Converting to lowercase
                    if(self.key == 'relationship'):
                        if(value in self.pattern_relationship):
                            self.dict_cond[self.key] = True
                    else:
                        match = self.dict_regex[self.key].search(value)
                        if(match != None):
                            self.dict_cond[self.key] = True

            #For keys in the if condition below we make use of the Spacy matcher object
            if(self.key == 'name' or self.key == 'period_stay' or self.key == 'residential_addr'
             or self.key == 'father_name'):
                print("self.key",self.key)
                self.matcher.add(str(self.key),self.for_debug,self.dict_pattern[self.key])
            
            # if(self.key == 'pan_no' or self.key == 'mobile_no' or self.key == 'email'):
            #     for value in value_ls:
            #         value = value.lower()
            #         match = self.dict_regex[self.key].search(value)
            #         if(match != None):
            #             self.dict_cond[self.key] = True


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
                    #Clear the self.matcher object and reinitialize it 
                    self.matcher = None
                    self.matcher = Matcher(self.nlp.vocab)

        # print("self.dict_cond",self.dict_cond)
        return self.dict_cond
    
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