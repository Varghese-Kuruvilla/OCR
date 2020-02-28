#Python utils
import cv2
import sys
def breakpoint():
    inp = input("Waiting for input")


def display(txt,img):
    '''Display an the img with namedwindow txt'''
    winname = txt
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,img)
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        sys.exit()