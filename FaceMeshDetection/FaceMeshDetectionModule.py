import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, mode=False, num_faces=1, detection_conf=0.5, tracking_conf=0.5):
        self.mode = mode
        self.num_faces = num_faces
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(mode, num_faces, detection_conf, tracking_conf)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    def find_face_mesh(self, img, draw=True):
        self.result = self.face_mesh.process(imgRGB)
        faces = []
        if self.result.multi_face_landmarks:

            for faceLms in self.result.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, faceLms, self.mp_face_mesh.FACEMESH_CONTOURS,
                                                landmark_drawing_spec=self.draw_spec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 100), 1)
                    face.append([x, y])
                faces.append(face)
        return img, faces


if __name__ == '__main__':
    # cap = cv2.VideoCapture('../video/2.mp4')
    cap = cv2.VideoCapture(0)
    pTime = 0
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = FaceMeshDetector(num_faces=4)
        img, faces = detector.find_face_mesh(img, draw=True)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
