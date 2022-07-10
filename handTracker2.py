import cv2
import mediapipe as mp
import math


class HandTracker():

    def __init__(self, mode=False, complexity=1, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.max_number_of_hands = maxHands
        self.detection_confidence = detectionConfidence
        self.tracking_confidence = trackConfidence
        self.modelComplexity = complexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_number_of_hands, self.modelComplexity, self.detection_confidence, self.tracking_confidence)
        self.tipIds = [4, 8, 12, 16, 20]
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

        return image


    def findPosition(self, image, handNumber=0, draw=True):
        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(image, (cx, cy), 5, (255,0,255), cv2.FILLED)
        return lmList


    def fingersUp(self):
        self.fingers = []
        if self.landmarks[self.tipIds[0]][1] > self.landmarks[self.tipIds[0] -1][1]:
            self.fingers.append(1)
        else:
            self.fingers.append(0)
        for id in range(1, 5):
            if self.landmarks[self.tipIds[id]][2] < self.landmarks[self.tipIds[id] -2][2]:
                self.fingers.append(1)
            else:
                self.fingers.append(0)

        return self.fingers

    def findDistance(self, p1, p2, image, draw=True, r=15, t=3):
        x1, y1 = self.landmarks[p1][1:]
        x2, y2 = self.landmarks[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), t)
            cv2.circle(image, (x1, y1), r, (0, 255, 0), cv2.FILLED)
            cv2.circle(image, (x2, y2), r, (0, 255, 0), cv2.FILLED)
            cv2.circle(image, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)

        return length, image, [x1, y1, x2, y2, cx, cy]
