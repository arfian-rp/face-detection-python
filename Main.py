import cv2
videoCam = cv2.VideoCapture(0)#ngambil webcam sebagai input gambar
face = cv2.CascadeClassifier('facexml.xml')
eye = cv2.CascadeClassifier('eyexml.xml')

while True:
    cond, frame = videoCam.read()#membaca setiap frame webcam

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    muka = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in muka: #ngasih kotak ke muka
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)

        roi_warna = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        mata = eye.detectMultiScale(roi_gray)
        for (mx,my,mw,mh)in mata:#ngasih kotak ke mata
            cv2.rectangle(roi_warna, (mx,my), (mx+mw, my+mh), (0, 150, 0), 2)
    cv2.putText(frame, "credit by:ArfianRp", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,30,255), 3)
    cv2.imshow('Face dan Eye detection', frame)#munculin frame yg tdi

    k = cv2.waitKey(1) & 0xff
    if k == ord('x'):#klik x untuk close
        break

videoCam.release()
cv2.destroyAllWindows()
