import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # mp related
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # read hands
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # if there are hands
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # draw point on hands
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_no=0, draw=True):

        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def draw_with_finger(self, img, draw=True):

        x, y = self.lm_list[8][1:]
        cv2.circle(img, (x, y), 3, [0, 100, 100], 1)
        return img

    def fingers_up(self, img, draw=True):
        tip_ids = [4, 8, 12, 16, 20]

        if len(self.lm_list) != 0:
            fingers = []
            # Thumb
            if self.lm_list[tip_ids[0]][1] < self.lm_list[tip_ids[0] - 1][1]:  # use cv orientation
                fingers.append(1)
            else:
                fingers.append(0)
            # four fingers
            for id in range(1, 5):
                if self.lm_list[tip_ids[id]][2] < self.lm_list[tip_ids[id] - 2][2]:  # use cv orientation
                    fingers.append(1)
                else:
                    fingers.append(0)
            total_fingers = fingers.count(1)
            cv2.putText(img, f"{total_fingers}", (50, 400), cv2.FONT_HERSHEY_PLAIN, 3, [100, 20, 100], 2)
        return img, fingers


def main():
    # to calculate fps
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        # read video
        success, img = cap.read()
        img = detector.find_hands(img)
        lmList = detector.find_position(img)
        if len(lmList) != 0:
            print(lmList[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
