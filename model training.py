import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

data_path='C:\\Users\\hp\\PycharmProjects\\attendance system\\face samples\\'
only_files=[f for f in listdir(data_path) if isfile(join(data_path,f))]

training_data,labels=[],[]

for i, files in enumerate(only_files):
    image_path=data_path+only_files[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(images,dtype=np.uint8))
    labels.append((i))

labels=np.asarray(labels,dtype=np.int32)

model=cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(training_data),np.asarray(labels))

print('model traning complete')

face_classifier=cv2.CascadeClassifier(r'C:\Users\hp\PycharmProjects\opencv work\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
    return img,roi

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
    ret,frame=cap.read()
    image,face=face_detector(frame)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)
        print(result)

        if result[1]<500:
            confidence=int(100*(1-(result[1])/300))
            display_string=str(confidence)+'% confidence it is user'
        cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(234,56,123),2)

        if confidence>75:
            cv2.cv2.putText(image,'unlocked',(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',image)
        else:
            cv2.cv2.putText(image, 'locked', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            cv2.imshow('Face Cropper', image)

    except:
        cv2.cv2.putText(image, 'face not found', (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()

