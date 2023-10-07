import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
capture.set(3,680)
capture.set(4,460)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
id_points = {i:(0,0) for i in range(21)}
print('see',type(id_points))
while True:
    suc, img = capture.read()
    imgRGB =  cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results =   hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                id_points[id]=(cx,cy)
                val = id_points[18][1]-id_points[19][1]
                if(val>50): print('Welcome')
    img = cv2.flip(img,1)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
