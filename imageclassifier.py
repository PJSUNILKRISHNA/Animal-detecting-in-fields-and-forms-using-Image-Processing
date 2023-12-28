import time

import cv2
from pydub import AudioSegment
from pydub.playback import play

import numpy as np
import os
import pyttsx3

def talk(text):
    engine.say(text)
    engine.runAndWait()

engine=pyttsx3.init()
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
rate=engine.getProperty('rate')
engine.setProperty('rate',190)



path = 'Imagequery'
orb = cv2.ORB_create(nfeatures=1000)# By implementing our product to the agri field , the crops will be saved and won't be damaged by the animals , so there will no loss for the farmers . And animals also won't be affected by this ,

images = []
classNames = []
myList = os.listdir(path)
print('Total classes detected',len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findDes(images):
    desList = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findID(img,desList,thres=15):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:

        for des in desList:
            matches = bf.knnMatch(des,des2,k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    #print(matchList)
    if len(matchList)!=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal

desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture(0)

#class_audio_mapping = {
        #'pig.jpg': 'bell.mp3',  # Replace 'class1' with actual class name
        #'rat.jpg': 'cat.mp3',  # Replace 'class2' with actual class name
        #'elephant.jpg': 'tiger.mp3',
        # Add more entries for other classes
    #}


while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    id = findID(img2, desList)
    if id != -1:
        class_name = classNames[id]
        cv2.putText(imgOriginal, class_name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        talk(class_name)

        # Play audio file corresponding to the detected class
        if class_name in class_audio_mapping:
            audio_path = class_audio_mapping[class_name]
            audio = AudioSegment.from_file(audio_path)
            play(audio)

    cv2.imshow('img2', imgOriginal)
    cv2.waitKey(1)
0






