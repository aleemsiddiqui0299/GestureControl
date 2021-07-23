import cv2
import mediapipe as mp
import time
import numpy as np
import math




class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands =maxHands
        self.detectionCon =detectionCon
        self.trackCon =trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        new_img = np.zeros(img.shape, dtype=np.uint8)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
                    self.mpDraw.draw_landmarks(new_img, handLMS, self.mpHands.HAND_CONNECTIONS)

        return img, new_img

    def findPosition(self, img, handNo = 0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id , lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return lmList





if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    detector = handDetector()
    pTime = 0
    cTime = 0


    vol = 0
    volBar = 400

    while True:
        blur_factor = 1
        success, img = cap.read()
        img , nimg= detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # print(lmList[2])
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx = (x1 + x2)//2
            cy = (y1 + y2)//2
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            cv2.circle(nimg, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(nimg, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(nimg, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(nimg, (x1, y1), (x2, y2), (255, 0, 255), 3)


            length = math.hypot(x2-x1, y2-y1)
            if(length <= 50):
                cv2.circle(img, (cx, cy), 12, (0,255,0), cv2.FILLED)
            volBar = np.interp(length, [50,300], [400, 150])

            blur_factor = np.interp(length, [50, 300], [1, 25])

        # print(lmList)

        cv2.rectangle(img, (50,150), (85, 400), (0,255,0), 3)
        cv2.rectangle(img, (50,int(volBar)), (85, 400), (0,255,0),  cv2.FILLED)

        cv2.rectangle(nimg, (50,150), (85, 400), (0,255,0), 3)
        cv2.rectangle(nimg, (50,int(volBar)), (85, 400), (0,255,0),  cv2.FILLED)


        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, "FPS: "+str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,2,(255,0,255), 3)
        cv2.putText(nimg, "FPS: "+str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,2,(255,0,255), 3)
        cv2.putText(img, "Brightness: " + str(int(blur_factor)), (400, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        cv2.putText(nimg, "Brightness: " + str(int(blur_factor)), (400,60 ), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        cv2.imshow("Image1", cv2.hconcat([cv2.blur(img,(int(blur_factor),int(blur_factor))), nimg]))
        # cv2.imshow("Image2", nimg)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyWindow()



