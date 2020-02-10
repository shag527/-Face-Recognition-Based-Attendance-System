import cv2

face_cascade=cv2.CascadeClassifier(r'C:\Users\hp\PycharmProjects\opencv work\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
count=0

def face_extractor(img):
    if img is not None:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)

    if faces is ():
        return None

    for(x,y,w,h) in faces:
        cropped_face=img[y:y+h,x:x+w]
    return cropped_face

while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_path='C:\\Users\\hp\\PycharmProjects\\attendance system\\face samples\\'+str(count)+'.jpg'
        cv2.imwrite(file_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('samples',face)
        if count==15:
            break

    else:
        print('Face not found !')
        pass

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()


