#names = ["Aishee","Akanksha","Arkaprabha","Parnavi","Ritodeep","Rohit","Sayan","Sukrita"]

import pandas as pd
name_data = pd.read_csv("/home/ritodeep/Desktop/Project_backend/FaceRecog/codes/HOG2/name_database.csv")
names = list(name_data["Names"])


from tqdm import tqdm

import os

import numpy as np

import cv2                   
import os.path

import re

import os

from matplotlib import style
import seaborn as sns

#model selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
                
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)


def assign_label(img,student_name):
    return student_name

def make_train_data(X,Y,student_name,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,student_name)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,0)
        img = cv2.resize(img, (150,150))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        X.append(np.array(img))
        Y.append(str(label))
    
def train(path):
    X=[]
    Y=[]

#    path = 'F:/academic_project/project/HOG1/'
    
    for name in names:
        make_train_data(X,Y,name,path+'/'+name)
    
    X=np.array(X)
    X=X/255
    
    le=LabelEncoder()
    Y1=np.array(le.fit_transform(Y))
    
    x_train,x_test,y_train,y_test = train_test_split(X,Y1,test_size=0.1,random_state=15)
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(x_train,y_train)
    
#    face_recognizer.write("lbph.json")
    
    acc = evaluate(face_recognizer,x_test,y_test)
    
    return acc, face_recognizer

def evaluate(face_recognizer,x_test,y_test):
    pred = []
    for i in range(len(x_test)):
        pred.append(face_recognizer.predict(x_test[i]))
    
    acc_sum = 0 
    for i in range(0,len(pred)):
        if y_test[i] == pred[i][0]:
            acc_sum += 1
        
        acc = acc_sum/len(pred)

    print(f"\nEvaluation Accuracy: {acc}")
    return acc

def sort_int(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def test_data(DIR):
    xtest=[]
    paths=[]
    for img in tqdm(sort_int(os.listdir(DIR))):
        path = os.path.join(DIR,img)
        paths.append(path)
        img = cv2.imread(path,0)
        img = cv2.resize(img, (150,150))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        xtest.append(np.array(img))
        
    return xtest,paths

def predict(path):
#    pred=[]

#    path = 'F:/academic_project/project/HOG1/test'
    
    x_test, paths =test_data(path)
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#    face_recognizer.read("lbph.json")
    face_recognizer.read("lbph_15.json")
#    face_recognizer.update()
    
#    for i in range(len(x_test)):
#        pred.append(face_recognizer.predict(x_test[i]))
#        
#    pred_name = [names[pred[i][0]] for i in range(len(pred))]
    
    pred_name = []
    unknown_list = []
    for i in range(len(x_test)):
        prediction = face_recognizer.predict(x_test[i])
        if prediction[1] > 73.65:
#            print(prediction[1])
            pred_name.append("Unknown")
            unknown_list.append(paths[i])
        else:
            pred_name.append(names[prediction[0]])
    
    print(pred_name)
#    print(unknown_list)
#    print(len(unknown_list))
    return pred_name
    
def predict_single(path):
#    path = 'F:/academic_project/project/HOG1/test/1.jpg'
    img = cv2.imread(path,0)
    img = cv2.resize(img, (150,150))
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("lbph_15.json")
    pred = face_recognizer.predict(img)
    
#    print(names[pred[0]])
    
#    if pred[1] > 70:
#        print("Unknown")
#    else:
#        print(names[pred[0]])
        
    return names[pred[0]]
    
if __name__ == '__main__':
#    a,b=train("F:/academic_project/project/HOG2")
    predict('/home/ritodeep/Desktop/Project_backend/FaceRecog/codes/HOG2/test')
#    predict_single()