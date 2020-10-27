# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:32:36 2020

@author: gmlgn
"""


import numpy as np
import cv2
import copy
import math

#영상 크기 조정
WIDTH = 640
HEIGHT = 480
cap2 = cv2.VideoCapture(0) #상단 카메라
cap2.set(3,WIDTH) 
cap2.set(4,HEIGHT)

#피부 경계값
skin_lower = np.array([0,140,90])
skin_upper = np.array([255,173,157])
kernel = np.ones((5,5),np.uint8)

#임시 키보드 좌표 고정
aextBot = [625, 260]
aextLeft = [10, 263]
aextRight = [569, 82] 
aextTop = [87, 89]
pts1 = np.float32([aextBot, aextLeft, aextRight, aextTop])
pts2 = np.float32([[0,0],[WIDTH,0],[0,HEIGHT],[WIDTH,HEIGHT]])
matrix = cv2.getPerspectiveTransform(pts1,pts2) 

#용도 미확인 및 임시 변수i
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
DETECT_THRESH = 80000

cap1 = cv2.VideoCapture(2)
cap1.set(3,IMAGE_WIDTH) 
cap1.set(4,IMAGE_HEIGHT)

tableEdgeHeight = 1
tmpNumber = 1
Hand = 8
handBound = [1,125,160,190,250,320,355,392,500]
handBoundMax = 500

#검사할 손
targetHand = 0
pastDetectResult = False



def distanceBetweenTwoPoints(start, end):
    
    x1,y1 = start
    x2,y2 = end
 
    return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))

def calculateAngle(A, B):

    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    C = np.dot(A,B)

    angle = np.arccos(C/(A_norm*B_norm))*180/np.pi
    return angle

def getFingerPosition(max_contour, img_result, debug):
    points1 = []


    M = cv2.moments(max_contour)
    if(M['m00'] == 0):
        M['m00'] = 1
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])


    max_contour = cv2.approxPolyDP(max_contour,0.02*cv2.arcLength(max_contour,True),True)
    hull = cv2.convexHull(max_contour)

    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0])) 

    if debug:
        cv2.drawContours(img_result, [hull], 0, (0,255,0), 2)
        for point in points1:
            cv2.circle(img_result, tuple(point), 15, [ 0, 0, 0], -1)


        # STEP 6-2
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)

    if defects is None:
        return -1,None

    points2=[]
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        angle = calculateAngle( np.array(start) - np.array(far), np.array(end) - np.array(far))

        if angle < 90:
            if start[1] < cy:
                points2.append(start)
      
            if end[1] < cy:
                points2.append(end)

    if debug:
        cv2.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
        for point in points2:
            cv2.circle(img_result, tuple(point), 20, [ 0, 255, 0], 5)


    # STEP 6-3
    points = points1 + points2
    points = list(set(points))


    # STEP 6-4
    new_points = []
    for p0 in points:
    
        i = -1
        for index,c0 in enumerate(max_contour):
            c0 = tuple(c0[0])

            if p0 == c0 or distanceBetweenTwoPoints(p0,c0)<20:
                i = index
                break

        if i >= 0:
            pre = i - 1
            if pre < 0:
                pre = max_contour[len(max_contour)-1][0]
            else:
                pre = max_contour[i-1][0]
      
            next = i + 1
            if next > len(max_contour)-1:
                next = max_contour[0][0]
            else:
                next = max_contour[i+1][0]


            if isinstance(pre, np.ndarray):
                pre = tuple(pre.tolist())
            if isinstance(next, np.ndarray):
                next = tuple(next.tolist())


        angle = calculateAngle( np.array(pre) - np.array(p0), np.array(next) - np.array(p0))     

        if angle < 90:
            new_points.append(p0)
  
    return 1,new_points

def cutImage(image):
    image_result = cv2.warpPerspective(image,matrix,(WIDTH,HEIGHT))        
    return image_result

def dctSkin(image):
    # BGR -> YCrCb 변환
    YCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    # 피부 검출
    mask_hand = cv2.inRange(YCrCb,skin_lower,skin_upper)
    # 마스크 침식
    mask_hand = cv2.erode(mask_hand, kernel)
    # 마스크 팽창
    #mask_hand = cv2.dilate(mask_hand, kernel)
     # 피부 색 나오도록 연산
    mask_color = cv2.bitwise_and(image,image,mask=mask_hand)
    return mask_color

def divImageInHalf(image,direction):
    image_right = image[0:HEIGHT, 0:(int)(WIDTH/2)]
    image_left = image[0:HEIGHT, (int)(WIDTH/2):WIDTH]
    if(direction == 'left'):
        return image_left
    elif(direction == 'right'):
        return image_right
    else:
        print('방향을 정해주세요 (left, right)')
        pass

def getPoint(image,hand_num):
    # hand_num = 0:왼손 새끼 -> 1:왼손 약지 -> ... -> 7:오른손 새끼
    # 오류 발생시 -1, -1 return
    
    resultPoint = [-1,-1]
    isLeft = 0;
    
    if(hand_num>=0 and hand_num<=3):
        isLeft =1;
        image = divImageInHalf(image,'left')
    elif(hand_num>=4 and hand_num<=7):
        hand_num = hand_num - 4
        image = divImageInHalf(image,'right')
        
    # get the coutours
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh1 = copy.deepcopy(image)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    ci = 0
    
    if length > 1:
        for i in range(length):  # find the biggest contour (ording to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
        res = contours[ci]
        drawing = np.zeros(image.shape, np.uint8)
        ret,points = getFingerPosition(res,drawing,debug=False)

        if ret > 0 and len(points) > 0:  
            points.sort()
            try:
                if(hand_num == 0):
                    resultPoint = points[0]
                elif(hand_num  == 1):
                    resultPoint = points[1]
                elif(hand_num  == 2):
                    resultPoint = points[2]
                elif(hand_num  == 3):
                    resultPoint = points[3]
                else:
                    print('손가락의 번호가 범위를 넘어감')
                    return -1, -1
            except:
                print('해당하는 손을 현재 찾지 못함')
                return -1, -1
            if(isLeft):
                resultPoint = (resultPoint[0] +(int)(WIDTH/2), resultPoint[1])
            return resultPoint
    else:
        print('손의 형상을 찾지 못함')
        return -1, -1
    
def getHandPosition(image,hand_num):
    ret, image_ori = cap2.read()
    #cv2.imshow("original", image_ori)
    
    #키보드 사이즈에 맞게 영상 자르기
    image_affine = cutImage(image_ori)
    #cv2.imshow("affine", image_affine)
    
    #피부색만 검출
    image_detected = dctSkin(image_affine)
    cv2.imshow("skin detect",image_detected) 
        
    #손의 좌표 찾기
    return getPoint(image_detected, hand_num) 


 
while(True):
    ret, src = cap1.read()
    dst = src.copy()
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray) 
    
    blur = cv2.blur(gray, (5,5))
    cv2.imshow("blur",blur) 
    
    canny = cv2.Canny(blur, 500, 150, apertureSize = 5, L2gradient = True)    
    cv2.imshow("canny",canny) 
    
    lines = cv2.HoughLinesP(canny, 0.8, np.pi / 180, 90, minLineLength = 30, maxLineGap = 2)
    
    if lines is not None:
        for i in lines:
            cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 2)
            tableEdgeHeight = (lines[0][0][1])
            print("height = ")
            print(tableEdgeHeight)
            
    cv2.imshow("test",dst)
    #if cv2.waitKey(1) == ord('q'):
    cv2.destroyAllWindows()
    break

    
kernel = np.ones((5,5),np.uint8)

while(True):
    ret, src = cap1.read()    
    dst = src.copy()
    
    #cut image
    dst = src[140:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    #dst = src[tableEdgeHeight:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
        
    # BGR -> YCrCb 변환
    YCrCb = cv2.cvtColor(dst,cv2.COLOR_BGR2YCrCb)
    # 피부 검출
    mask_hand = cv2.inRange(YCrCb,np.array([0,145,90]),np.array([255,173,157]))
    # 피부 색 나오도록 연산
    mask_color = cv2.bitwise_and(dst,dst,mask=mask_hand)
    #cv2.imshow("mask_color",mask_color) 
    
    # 계산을 위해 영상 이진화
    ret, thresh = cv2.threshold(mask_color,20,255, cv2.THRESH_BINARY)
    #cv2.imshow("thresh",thresh)
    
    erosion = cv2.erode(thresh,kernel,iterations = 1)
    # 침식연산(잡음 제거)
    #cv2.imshow("erode",erosion)
    
    # 잘린 영상의 밝기합을 이용하여 입력여부 계산
    threshsum = (int)(np.sum(thresh))
    
    # 손의 위치 테두리를 계산하기 위한 블러 연산
    blur2 = cv2.blur(thresh,(5,5))
    #cv2.imshow("blur2",blur2)

    #손가락 화면 분할
    hand = [thresh[0:(int)(IMAGE_HEIGHT),
                   (int)(IMAGE_WIDTH*(handBound[i]/handBoundMax)):(int)(IMAGE_WIDTH*(handBound[i+1]/handBoundMax))]
            for i in range(Hand)]

    
    cv2.imshow("Hand1",hand[0])
    cv2.imshow("Hand2",hand[1])
    cv2.imshow("Hand3",hand[2])
    cv2.imshow("Hand4",hand[3])
    cv2.imshow("Hand5",hand[4])
    cv2.imshow("Hand6",hand[5])
    cv2.imshow("Hand7",hand[6])
    cv2.imshow("Hand8",hand[7])
    

    threshSum = [(int)(np.sum(hand[i])) for i in range(Hand)]
    
    #손가락 클릭 여부 확인

    if ((max(threshSum) >= (int)(DETECT_THRESH)) and (pastDetectResult == False)):
        targetHand = threshSum.index(max(threshSum))
        print(tmpNumber, "입력 감지 -> ", targetHand)
        tmpNumber = tmpNumber + 1
        pastDetectResult = True
        
        detectedHandPos = getHandPosition(cap2,7 - targetHand)
        print(detectedHandPos)
        
    elif((max(threshSum)) < (int)(DETECT_THRESH)) and (pastDetectResult == True):
        print("손 뗌")
        pastDetectResult = False
        
    if cv2.waitKey(1) == ord('q'): break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

















