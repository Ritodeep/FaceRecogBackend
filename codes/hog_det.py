import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import dlib
import time
# data visualisation and manipulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from PIL import Image
import pandas as pd
 
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)


# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
from tqdm import tqdm
import os
import os.path

date = pd.to_datetime('today').strftime('%d-%m-%Y')

def face_detecthog(image):
    global date
    hog_face_detector = dlib.get_frontal_face_detector()
    start = time.time()

      # apply face detection (hog)
    faces_hog = hog_face_detector(image, 1)
    end = time.time()
    print("Execution Time (in seconds) :")
    print("HOG : ", format(end - start, '.2f'))
  
    i=0
    #loop over detected faces
    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

  # draw box over face
  
        if w < 50 or h < 50:
            continue
      
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        roi_color = image[y:y+h, x:x+h]
        print(roi_color)
  # display output image
        plt.imshow(roi_color)
        plt.show()

#      if not os.path.exists(dirName):
#          os.mkdir(dirName)
#          print("Directory " , dirName ,  " Created ")
        path1 = "F:/academic_project/project/HOG_Test"
        if not os.path.exists(path1):
            os.mkdir(path1)       
        path = path1 + "/" + date
        if not os.path.exists(path):
            os.mkdir(path)
        
        cv2.imwrite(path+'/'+str(i+1)+'.jpg',roi_color)
        i+=1
#      return roi_color
      
      
def assign_label(img,student_name):
    return student_name

def folderwise(student_name, DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,student_name)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_detecthog(img)
        break
    
def hog(DIR):
#    label=assign_label(img,student_name)
#    path = os.path.join(DIR,img)
    img = cv2.imread(DIR,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detecthog(img)
#    return roi

if __name__ == '__main__':
    Aishee = 'F:/academic_project/project/Original/Aishee'        
    folderwise('Aishee',Aishee)