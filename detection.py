import numpy as np
import cv2


inhh = cv2.CascadeClassifier('inhalation_hazard.xml')
flal = cv2.CascadeClassifier('flammable_liquid.xml')
ngas = cv2.CascadeClassifier('nonflammable_gas.xml')


cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inhalation = inhh.detectMultiScale(gray, 1.3, 5)
    liquid = flal.detectMultiScale(gray, 1.3, 5)
    nongas = ngas.detectMultiScale(gray, 1.3, 5)    

    for (x,y,w,h) in inhalation:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.putText(img,'inhalation hazard',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    for (x,y,w,h) in liquid:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.putText(img,'flammable liquid',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    for (x,y,w,h) in nongas:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.putText(img,'nonflammable gas',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        


    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
