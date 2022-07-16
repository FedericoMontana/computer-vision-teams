import cv2
import mediapipe as mp
import time
from collections.abc import Iterable
from sklearn.preprocessing import normalize


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):

        self.hands = mp.solutions.hands.Hands(
            mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )

        self.results_ = None
        self.img_ = None

    def found_hands(self):
        return self.results_.multi_hand_landmarks != None

    def _get_positions(self):

        hands = []

        if self.results_.multi_hand_landmarks:
            for h_k, hand in enumerate(self.results_.multi_hand_landmarks):

                lm = {}
                for l_k, point in enumerate(hand.landmark):
                    h, w, c = self.img_.shape
                    cx, cy = int(point.x * w), int(point.y * h)
                    lm[l_k] = [cx, cy]

                hands.append(lm.copy())

        return hands

    def get_positions_normalized(self):

        if self.results_.multi_hand_landmarks:

            hands = self._get_positions()

            n = np.zeros(shape=(len(hands), len(hands[0])))

            for count, lm in enumerate(hands):
                x = normalize([[x[0] for x in (n for n in lm.values())]])
                y = normalize([[x[1] for x in (n for n in lm.values())]])

                n[count] = x + y

            return n

    def transform_connect_lines(self, img):

        drawer = mp.solutions.drawing_utils

        if self.results_.multi_hand_landmarks:
            for hand in self.results_.multi_hand_landmarks:
                drawer.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)

        return img

    def transform_draw(self, img):

        hands = self._get_positions()
        if self.results_.multi_hand_landmarks:
            for h in hands:
                for lm in h.values():
                    cv2.circle(img, (lm[0], lm[1]), 7, (255, 0, 255), cv2.FILLED)

        return img

    # Return the coordinate of a square around the hand's landmarks

    def get_hands_squared(self, space=0.1, correct_limits=True):

        squares = []
        if self.results_.multi_hand_landmarks:
            hands = self._get_positions()
            for lm in hands:
                x_max = max([x[0] for x in lm.values()])
                y_max = max([x[1] for x in lm.values()])
                x_min = min([x[0] for x in lm.values()])
                y_min = min([x[1] for x in lm.values()])

                if space is not None and space > 0:
                    size_x = int((x_max - x_min) * space)
                    size_y = int((y_max - y_min) * space)

                    x_max = x_max + size_x
                    x_min = x_min - size_x
                    y_max = y_max + size_y
                    y_min = y_min - size_y

                if correct_limits:
                    x_min = x_min if x_min >= 0 else 0
                    y_min = y_min if y_min >= 0 else 0

                    x_max = self.img_.shape[1] if x_max >= self.img_.shape[1] else x_max
                    y_max = self.img_.shape[0] if y_max >= self.img_.shape[0] else y_max

                squares.append([(x_min, y_min), (x_max, y_max)])

        return squares

    def transform_square(self, img):

        squares = self.get_hand_squared()

        for s in squares:
            cv2.rectangle(img, s[0], s[1], (0, 255, 0), 2)

        return img

    def fit(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # TODO ver si se guarda los valores de la imagen qeu interesan
        # en results_
        self.results_ = self.hands.process(imgRGB)
        self.img_ = img.copy()


import time
import cv2

print(cv2.__version__)
import numpy as np


if __name__ == "__main__":

    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()

        detector.fit(img)
        img = detector.transform_draw(img)
        # lala = detector._get_positions()
        # print(lala)
        # img = detector.findHands(img)
        # lmlist = detector.findPosition(img)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cap.release()
