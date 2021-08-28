import cv2
import mediapipe as mp
import time
import PoseModule as pm

if __name__=='__main__':

    cap = cv2.VideoCapture(0)
    pTime = 0
    while True:
        success, img = cap.read()
        detector = pm.PoseDetector()
        img = detector.findpose(img)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 5, (255, 0, 0), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)