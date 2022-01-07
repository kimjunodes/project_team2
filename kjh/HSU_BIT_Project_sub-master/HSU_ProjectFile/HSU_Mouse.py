import cv2
import mediapipe as mp
import math
import pyautogui
import time

cam = cv2.VideoCapture(0)

cam_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
cam_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

pyautogui.FAILSAFE = False

mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1-x2, 2)) + math.sqrt(math.pow(y1-y2, 2)) #제곱근(sqrt), 제곱(pow)

while True:
    success, img = cam.read()
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = my_hands.process(imgRGB)

    curmX, curmY = pyautogui.position()
    print("커서 {0}, {1}".format(curmX,curmY))

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # 관절 사이 계산
            gumx = handLms.landmark[8].x
            gumy = handLms.landmark[8].y
            umx = handLms.landmark[4].x
            umy = handLms.landmark[4].y
            print("검지 {0}, {1}".format(gumx,gumy))
            open = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[14].x, handLms.landmark[14].y) < \
                   dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[16].x, handLms.landmark[16].y)
            if open == False:
                if gumx <= 0.333 and gumy <= 0.333: #1
                    pyautogui.moveTo(curmX+30, curmY-30)
                elif gumx>0.333 and gumy<=0.333 and gumx<=0.666 and gumy<=0.333: #2
                    pyautogui.moveTo(curmX, curmY-30)
                elif gumx > 0.666 and gumy <= 0.333: #3
                    pyautogui.moveTo(curmX-30, curmY - 30)
                elif gumx<=0.333 and gumy>0.333 and gumy<=0.666: #4
                    pyautogui.moveTo(curmX+30, curmY)
                #elif gumx>0.333 and gumy>0.333 and gumx<=0.666 and gumy<=0.666: #click
                    #if dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[10].x, handLms.landmark[10].y) < dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[12].x, handLms.landmark[12].y):
                        #pyautogui.mouseDown()
                        #pyautogui.mouseUp()
                        #time.sleep(1)
                elif gumx>0.666 and gumy>0.333 and gumy<=0.666: #5
                    pyautogui.moveTo(curmX - 30, curmY)
                elif gumx<=0.333 and gumy>0.666: #6
                    pyautogui.moveTo(curmX + 30, curmY + 30)
                elif gumx>0.333 and gumy>0.666 and gumx<=0.666 and gumy>0.666: #7
                    pyautogui.moveTo(curmX, curmY + 30)
                elif gumx>0.666 and gumy>0.666: #8
                    pyautogui.moveTo(curmX - 30, curmY + 30)
                else:
                    continue
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    fimg = cv2.flip(img, 1)
    cv2.imshow('video', fimg)

    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
'''
while True:
    success, img = cap.read()
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = my_hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            if open == False:


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


#def drag_():
#    return 0


screenWidth, screenHeight = pyautogui.size() #모니터 크기
curmX,curmY = pyautogui.position() #현재 마우스 위치
print('{0},{1}'.format(screenWidth, screenHeight))
print('{0},{1}'.format(curmX, curmY))

pyautogui.move(0, 100) #마우스 옮기기
curmX,curmY = pyautogui.position() #현재 마우스 위치
print('{0},{1}'.format(curmX, curmY))
#pyautogui.dragTo(300, 300, 3, button='left') #드래그 한 상태로 x, y좌표로 3초간 이동.이건 좌클릭
#pyautogui.dragTo(500, 500, 3, button='right') #이건 우클릭

pyautogui.mouseDown()
time.sleep(5)
pyautogui.mouseUp()'''