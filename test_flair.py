#Script to test state of the art NER model flair
import sys
# import cv2

from flair.data import Sentence
from flair.models import SequenceTagger




def test_flair():
    f = open("/home/varghese/Nanonets/OCR/code/debug_spacy.txt","r")
    text = f.read()

    sentence = Sentence(text)

    tagger = SequenceTagger.load('ner')

    # run NER over sentence
    tagger.predict(sentence)

    print(sentence)
    print('The following NER tags are found:')

    # iterate over entities and print
    for entity in sentence.get_spans('ner'):
        print(entity)
    





if __name__ == '__main__':
    test_flair()