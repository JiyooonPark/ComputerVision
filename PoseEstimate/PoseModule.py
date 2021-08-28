import cv2
import mediapipe as mp
import time
class PoseDetector:
    def __init__(self,mode = False, complexity = 1, smooth = True, enable_segmentation = False,
        smooth_segmentation = True, detection_confidence = 0.5, tracking_confidence = 0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enableSeg = enable_segmentation
        self.smoothSeg = smooth_segmentation
        self.detectConf = detection_confidence
        self.trackConf = tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
    def findpose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList


if __name__=='__main__':

    cap = cv2.VideoCapture(0)
    pTime = 0
    while True:
        success, img = cap.read()
        detector = PoseDetector()
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