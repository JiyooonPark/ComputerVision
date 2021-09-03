import HandTrackingModule as htm
import time, os
import cv2

wCam, hCam = 640, 480

dark_purple = [42, 9, 68]
purple = [59, 24, 95]
magenta_pink = [161, 37, 104]
baby_yellow = [254, 194, 96]

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
# myList = os.listdir(folderPath)
myList = [i for i in range(0, 6)]

overlayList = []
for impath in myList:
    image = cv2.imread(f'{folderPath}/{impath}.jpg')
    print(f'{folderPath}/{impath}')
    overlayList.append(image)

pTime = 0
detector = htm.handDetector(detectionCon=0.75, maxHands=1)

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tip_ids[0]][1] < lmList[tip_ids[0] - 1][1]:  # use cv orientation
            fingers.append(1)
        else:
            fingers.append(0)
        # four fingers
        for id in range(1, 5):
            if lmList[tip_ids[id]][2] < lmList[tip_ids[id]-2][2]: # use cv orientation
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        # print number / image
        total_fingers = fingers.count(1)
        cv2.putText(img, f"{total_fingers}", (50, 40), cv2.FONT_HERSHEY_PLAIN, 3, list(reversed(magenta_pink)), 2)
        h, w, c = overlayList[total_fingers].shape
        img[100:100+h, 20:20+w] = overlayList[total_fingers]

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (570, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, list(reversed(purple)), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
