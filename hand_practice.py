import os
import cv2
import time
import hand as htm
cap=cv2.VideoCapture(0)
Folder = "Fingers"
ptime = 0
lst = os.listdir(Folder)
lst_2 = []
for i in lst:
    image = cv2.imread(f"{Folder}/{i}")
    lst_2.append(image)
detector = htm.handDetector(detectionCon = 0.55)
fingerid = [4,8,12,16,20]
while True:
    ret,frame=cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    #print(lmList)
    if len(lmList)!=0:
        fingers = []
        if lmList[fingerid[0]][1] < lmList[fingerid[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[fingerid[id]][2] < lmList[fingerid[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        count = fingers.count(1)
        #print(count)
        h , w , c = lst_2[count-1].shape
        frame[0:h,0:w] = lst_2[count-1]
        cv2.rectangle(frame,(0,200),(175,400),(123,231,133),-1)
        cv2.putText(frame,f"{count}",(55,350),cv2.FONT_HERSHEY_SIMPLEX,3,(200,155,55),3)
    ctime = time.time()
    fps = 1 /(ctime-ptime)
    ptime = ctime
    cv2.putText(frame,f"FPS: {int(fps)}",(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(200,155,55),3)
    cv2.imshow("HandTracking",frame)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()