import cv2
import numpy as np
import os
import pyttsx3

#engine = pyttsx3.init()
#voices = engine.getproperty('voices')
#engine.setproperty('voice',voices[0].id)
#def talk(text):
 #   engine.say(text)
  #  engine.runAndWait()


path = 'Imagequery'
orb = cv2.ORB_create(nfeatures=1000)
## IMPORT IMAGES
images = []
classNames =[]
myList = os.listdir(path)
print('Total Classes Detected', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findDes(images):
    desList=[]
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findID(img, desList,thres=15):
    kp1, des1 = orb.detectAndCompute(img, None)
    kp2, des2 = orb.detectAndCompute(img, None)
    kp3, des3 = orb.detectAndCompute(img, None)
    kp4, des4 = orb.detectAndCompute(img, None)
    kp5, des5 = orb.detectAndCompute(img, None)
    kp6, des6 = orb.detectAndCompute(img, None)
    kp7, des7 = orb.detectAndCompute(img, None)
    kp8, des8 = orb.detectAndCompute(img, None)

    bf = cv2.BFMatcher()
    matchList=[]
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des1,des2,k=2)
            matches = bf.knnMatch(des3,des4,k=2)
            matches = bf.knnMatch(des5,des6,k=2)
            matches = bf.knnMatch(des7,des8,k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
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

while True:

    success,img = cap.read()
    imgOriginal = img.copy()
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    id = findID(img,desList)
    if id != -1:
        cv2.putText(imgOriginal,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        #cv2.putTalk(imgOriginal,classNames[id])

    cv2.imshow('img',imgOriginal)
    cv2.waitKey(1)




#bf = cv2.BFMatcher()
#matches = bf.knnMatch(des1,des2,k=2)
#matches = bf.knnMatch(des3,des4,k=2)
#matches = bf.knnMatch(des5,des6,k=2)
#matches = bf.knnMatch(des7,des8,k=2)

#good = []
#for m,n in matches:
 #   if m.distance < 0.75*n.distance:
  #      good.append([m])
#print(len(good))
#