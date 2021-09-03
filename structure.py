import cv2
import time
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.rectangle(img, (img.shape[1] - 80, img.shape[0] - 40), (img.shape[1] - 10, img.shape[0] - 10), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, f"FPS:{int(fps)}", (img.shape[1] - 70, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    print(img.shape)
    cv2.imshow("Image", img)

    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
