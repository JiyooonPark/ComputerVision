import cv2
import time
import PoseModule as pm
import numpy as np

wCam, hCam = 640, 480

# cap = cv2.VideoCapture('../video/workout_woman.mp4')
cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)

pTime = 0
count = 0
dir = 0

detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        angle = detector.findAngle(img, 11, 13, 15, True)
        per = np.interp(angle, (25, 165), (0, 100))
        bar = np.interp(angle, (25, 165), (img.shape[0] - 300, img.shape[0] - 20))

        # check for dumbbell curls
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0
        if 15 <= per <= 85:
            cv2.rectangle(img, (20, img.shape[0] - 10), (60, int(bar)), (100, 25, 100), cv2.FILLED)
        else:
            cv2.rectangle(img, (20, img.shape[0] - 10), (60, int(bar)), (100, 25, 255), cv2.FILLED)

    # show count
    cv2.putText(img, str(count), (25, img.shape[0] - 330), cv2.FONT_HERSHEY_PLAIN, 2, [100, 0, 0], 2)
    cv2.rectangle(img, (20, img.shape[0] - 10), (60, img.shape[0] - 300), [100, 25, 100])

    # extra information
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.rectangle(img, (img.shape[1] - 80, img.shape[0] - 40), (img.shape[1] - 10, img.shape[0] - 10), (255, 255, 255),
                  cv2.FILLED)
    cv2.putText(img, f"FPS:{int(fps)}", (img.shape[1] - 70, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    cv2.imshow("Image", img)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
