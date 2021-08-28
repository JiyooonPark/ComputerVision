import cv2
import mediapipe as mp
import time

class FaceDetection:
    def __init__(self, detectionConf=0.5, model_selection=0):
        self.detectionConf = detectionConf
        self.model_selection = model_selection

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.detectionConf, self.model_selection)

    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs=[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), int(bboxC.width*iw), int(bboxC.height*ih)
                bboxs.append([bbox, detection.score])
                if draw:
                    # self.mpDraw.draw_detection(img, detection)
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f"{int(detection.score[0]*100)}%", (bbox[0], bbox[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 100), 2)

        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=10, rt = 1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255, 0, 100), rt)

        # Top left x, y
        cv2.line(img, (x,y), (x+l, y), (255, 0, 100), t)
        cv2.line(img, (x,y), (x, y+l), (255, 0, 100), t)
        # Top right x, y
        cv2.line(img, (x1,y), (x1-l, y), (255, 0, 100), t)
        cv2.line(img, (x1,y), (x1, y+l), (255, 0, 100), t)

        # Bottom right x, y
        cv2.line(img, (x1,y1), (x1-l, y1), (255, 0, 100), t)
        cv2.line(img, (x1,y1), (x1, y1-l), (255, 0, 100), t)
        # Bottom left x, y
        cv2.line(img, (x,y1), (x+l, y1), (255, 0, 100), t)
        cv2.line(img, (x,y1), (x, y1-l), (255, 0, 100), t)
        return img


if __name__=='__main__':

    cap = cv2.VideoCapture('videos/2.mp4')
    pTime = 0
    while True:
        success, img = cap.read()
        detector = FaceDetection(0.75)
        img, bboxs = detector.findFace(img, draw=True)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS:{int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 100), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(20)