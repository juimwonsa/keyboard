# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:24:36 2020

@author: gmlgn
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:04:48 2020

@author: PC
"""
import numpy as np
import cv2

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
capture = cv2.VideoCapture(0)

tableEdgeHeight = 1
tmpNumber = 1
pastDetectResult = False


    
kernel = np.ones((3,3),np.uint8)

while(True):
    ret, src = capture.read()
    src = cv2.resize(src, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    
    dst = src.copy()
    
    #cut image
    #dst = src[tableEdgeHeight:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    
    cv2.imshow("src",src) 
    cv2.imshow("dst",dst) 
    
    # BGR -> YCrCb 변환
    YCrCb = cv2.cvtColor(dst,cv2.COLOR_BGR2YCrCb)
    # 피부 검출
    mask_hand = cv2.inRange(YCrCb,np.array([0,145,77]),np.array([255,173,157]))
    # 피부 색 나오도록 연산
    mask_color = cv2.bitwise_and(dst,dst,mask=mask_hand)
    cv2.imshow("mask_color",mask_color) 
    
    # 계산을 위해 영상 이진화
    ret, thresh = cv2.threshold(mask_color,20,255, cv2.THRESH_BINARY)
    cv2.imshow("thresh",thresh) 
    erosion = cv2.erode(thresh,kernel,iterations = 1)
    # 침식연산(잡음 제거)
    cv2.imshow("erode",erosion) 
    # 잘린 영상의 밝기합을 이용하여 입력여부 계산
    threshsum = (int)(np.sum(thresh))
    # 손의 위치 테두리를 계산하기 위한 블러 연산
    blur2 = cv2.blur(thresh,(5,5))
    cv2.imshow("blur2",blur2)
    blur2h = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY);
    contours, hierarchy = cv2.findContours(blur2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if (((float)(threshsum)>(float)(100000)) & (pastDetectResult == False)):
        print(tmpNumber)
        tmpNumber += 1
        print("입력 감지됨")
        pastDetectResult = True
            
    elif(((float)(threshsum)<=(float)(100000)) & (pastDetectResult == True)):
        print("손 뗌")
        pastDetectResult = False
        
    
    if cv2.waitKey(1) == ord('q'): break

capture.release()
cv2.destroyAllWindows()