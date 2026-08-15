import cv2
import mediapipe as mp
import time


class HandDetector:

    def __init__(self, maxHands = 2 , mode = False , trackCon = 0.5 , detectionCon = 0.5):
        self.maxHands = maxHands
        self.trackCon = trackCon
        self.detectionCon = detectionCon
        self.mode = mode

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode = self.mode,
            max_num_hands = self.maxHands,
            min_detection_confidence = self.detectionCon,
            min_tracking_confidence = self.trackCon,
            model_complexity = 1
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self,img,draw=True):
        imgRBG = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRBG)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,draw=True,handNo=0):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id , lm in enumerate(myHand.landmark):
                h , w , c = img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),10,(253,123,100),3)
        return lmList

def main():
    cap=cv2.VideoCapture(0)
    detector = HandDetector()
    ptime = 0
    while True:
        success , img = cap.read()
        img = detector.findHand(img)
        lmList = detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[8])
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img,f"FPS : {str(int(fps))}",(0,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
        cv2.imshow("Hand",img)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ =="__main__":
    main()