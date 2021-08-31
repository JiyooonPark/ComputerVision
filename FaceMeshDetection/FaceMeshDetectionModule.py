import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self, mode=False,num_faces=1,detectionConf=0.5,trackingConf=0.5):
        self.mode = mode
        self.num_faces = num_faces
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(mode,num_faces,detectionConf,trackingConf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.result = self.faceMesh.process(imgRGB)
        faces = []
        if self.result.multi_face_landmarks:

            for faceLms in self.result.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 100), 1)
                    face.append([x, y])
                faces.append(face)
        return img, faces

if __name__=='__main__':
    cap = cv2.VideoCapture('../video/mamamoo.mp4')
    pTime = 0
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = FaceMeshDetector(num_faces=4)
        img, faces = detector.findFaceMesh(img, draw=False)
        if len(faces) != 0:
            print(faces[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

