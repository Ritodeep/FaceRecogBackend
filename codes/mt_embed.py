import pandas as pd
name_data = pd.read_csv("F:/academic_project/project/HOG2/name_database.csv")
names = list(name_data["Names"])

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.models import load_model

from numpy import expand_dims
from sklearn.preprocessing import Normalizer

import cv2

from sklearn.externals import joblib
import numpy as np

from tqdm import tqdm

import os
import os.path
import re

from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

date = pd.to_datetime('today').strftime('%d-%m-%Y')

def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    
    faces = []
    for i in range(0,len(results)):
        x1, y1, width, height = results[i]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        faces.append(face_array)
        
        path = "F:/academic_project/project/HOG_Test/"+date
        if not os.path.exists(path):
            os.mkdir(path)
        
        cv2.imwrite(path+'/'+str(i+1)+'.jpg',face_array)
        
    return faces, results


def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]
#    return samples
    
def assign_label(img,student_name):
    return student_name

def make_train_data(X,Y,student_name,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,student_name)
        path = os.path.join(DIR,img)
        img = cv2.imread(path)
        img = cv2.resize(img, (160,160))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        X.append(np.array(img))
        Y.append(str(label))
    
def train(path):
    model = load_model('keras-facenet/model/facenet_keras.h5')
    model.load_weights('keras-facenet/weights/facenet_keras_weights.h5')
    
    X=[]
    Y=[]

#    path = 'F:/academic_project/project/HOG1/'
    
    for name in names:
        make_train_data(X,Y,name,path+'/'+name)

    le=LabelEncoder()
    Y1=le.fit_transform(Y)

    X=np.array(X)
    X=X/255
    # X=X[:,:,:,np.newaxis]

    x_train,x_val,y_train,y_val=train_test_split(X,Y1,test_size=0.1,random_state=42)
    
    newTrainX = list()
    for face_pixels in x_train:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)
    
    newTestX = list()
    for face_pixels in x_val:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)    
    newTestX = asarray(newTestX)
    print(newTestX.shape)
    
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(newTrainX)
    testX = in_encoder.transform(newTestX)
    
    modelsvc = SVC(kernel='linear', probability=True)
    modelsvc.fit(trainX, y_train)
    
    tacc,vacc = evaluate(modelsvc,trainX,testX,y_train,y_val)
    
    print(tacc,vacc)
    
#    joblib_file = "svcmodel.json"
#    joblib.dump(modelsvc, joblib_file)
    
    return tacc,vacc

def evaluate(modelsvc,trainX,testX,y_train,y_val):
    yhat_train = modelsvc.predict(trainX)
    yhat_test = modelsvc.predict(testX)
    
    score_train = accuracy_score(y_train, yhat_train)
    score_test = accuracy_score(y_val, yhat_test)
    
    return score_train, score_test

def test_im(path):
    model = load_model('keras-facenet/model/facenet_keras.h5')
    model.load_weights('keras-facenet/weights/facenet_keras_weights.h5')
    
    imgt = cv2.imread(path)
    imgt = cv2.resize(imgt, (160,160))
    imgt = cv2.normalize(imgt, None, 0, 255, cv2.NORM_MINMAX)
    
    testim = get_embedding(model,imgt)
    testim = expand_dims(testim, axis=0)
    
    in_encoder = Normalizer(norm='l2')
    testim = in_encoder.transform(testim)
    
    modelsvc = joblib.load('svcmodel.json')
    testr = modelsvc.predict(testim)
#    
    pred = names[testr[0]]
    
    print(pred)
    
def sort_int(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def test_data(DIR):
    xtest=[]
#    paths=[]
    for img in tqdm(sort_int(os.listdir(DIR))):
        path = os.path.join(DIR,img)
#        paths.append(path)
        img = cv2.imread(path)
        img = cv2.resize(img, (160,160))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        xtest.append(np.array(img))
        
    return xtest
    
def test_im_folder(path):
    x_test = test_data(path)
    
    model = load_model('keras-facenet/model/facenet_keras.h5')
    model.load_weights('keras-facenet/weights/facenet_keras_weights.h5')
    
    test = []
    for i in range(0,len(x_test)):
        testim = get_embedding(model,x_test[i])
        test.append(testim)
    
    in_encoder = Normalizer(norm='l2')
    test = in_encoder.transform(test)
    
    modelsvc = joblib.load('svcmodel.json')
    testr = modelsvc.predict(test)
#    
    pred = []
    
    for i in range(0,len(testr)):
        pred.append(names[testr[i]])
    
    print(pred)
    
    
if __name__ == "__main__":
    # pixels, details = extract_face('F:/academic_project/project/gr.jpg')
    path = "F:/academic_project/project/HOG2/test/1.jpg"
    test_im(path)
    # path = "F:/academic_project/project/HOG1/test"
    # test_im_folder(path)
    