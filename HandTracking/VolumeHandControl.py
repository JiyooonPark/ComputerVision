import cv2, time
import HandTrackingModule as htm
import numpy as np
import math

''' WINDOWS VOLUME CONTROL
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
'''

# VOLUME CONTROL FOR UBUNTU
from subprocess import call

###########################  CAM SETTING
wCam, hCam = 640, 480
pTime = 0
###########################  INIT PARAM
vol = 0
volBar = 400
volPer  =0
point_size = 7
line_size = 3
###########################  INPUT
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
###########################  COLOR
sky_blue = (224, 132, 78)
dark_blue = (253, 63, 44)
indigo_blue = (253, 143, 113)
indigo_pink = (106, 67, 216)


detector = htm.handDetector(detectionCon=0.8, maxHands=1)

''' WINDOWS VOLUME CONTROL
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
volume.GetMasterVolumeLevel()
print(volume.GetVolumeRange())
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]
'''

while True:

    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        x3, y3 = lmList[0][1], lmList[0][2] # to get length of hand
        cx, cy = (x1+x2) // 2, (y1+y2)//2

        hand_length = math.hypot(x3 - x1, y3 - y1)

        cv2.circle(img, (x1, y1), point_size, sky_blue, cv2.FILLED )
        cv2.circle(img, (x2, y2), point_size, sky_blue, cv2.FILLED )
        cv2.circle(img, (cx, cy), point_size, sky_blue, cv2.FILLED )

        cv2.line(img, (x1, y1), (x2, y2), sky_blue, line_size)
        length = math.hypot(x2-x1, y2-y1)

        vol = np.interp(length, [hand_length*0.1, hand_length], [0, 100])
        volBar = np.interp(length, [hand_length*0.1, hand_length], [400, 150])
        volPer = np.interp(length, [hand_length*0.1, hand_length], [0, 100])

        ''' WINDOWS VOLUME CONTROL
        vol = np.interp(length, [hand_length*0.1, hand_length*0.8],[minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)
        '''

        # VOLUME CONTROL FOR UBUNTU
        call(["amixer", "-D", "pulse", "sset", "Master", str(vol)+"%"])

        if volPer < 20 or volPer > 90:
            cv2.line(img, (x1, y1), (x2, y2), indigo_pink, line_size)
            cv2.rectangle(img, (50, 150), (85, 400), indigo_pink, 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), indigo_pink, cv2.FILLED)
        else:
            cv2.rectangle(img, (50, 150), (85, 400), dark_blue, 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), dark_blue, cv2.FILLED)

    cv2.putText(img, f"{int(volPer)}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, dark_blue, 3)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f"FPS:{int(fps)}", (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, dark_blue, 3)
    cv2.imshow("Img", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
