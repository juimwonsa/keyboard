import numpy as np
import cv2
import imutils
import copy
import math
import finger_log

WIDTH = 400
HEIGHT = 300

lower = np.array([0,10,0], dtype="uint8")
upper = np.array([20,255,255], dtype="uint8")

cap = cv2.VideoCapture(1)

#*********************************************************
#**************************setup**************************
#*********************************************************

while(True):
    #1 - load the image
    ret, image_ori = cap.read()

    #2 - Gary
    image_gray = cv2.cvtColor(image_ori,cv2.COLOR_BGR2GRAY)
    
    #3 - Blur
    image_blur = cv2.medianBlur(image_gray,9)#<<odd numbers only, prefer ~(WIDTH/20)~
    #image_blur = cv2.GaussianBlur(image_gray,(9,9),0)
    #medianBlur를 이용하여 아래에서 컨투어를 구할 때 키보드 자판 하나하나들로 나뉘는 것을 막음
    # TODO: 가우시안 블러는 혹시 쓸 수도 있어서 남겨놓은것 나중에 지울것
    #cv2.imshow("3-blur", image_blur)
    
    #4 - edge detection (Canny)
    image_edge = cv2.Canny(image_blur, 100, 200, 3)
    #cv2.imshow("4-edge dection", image_edge)
    
    #5 - Rotate
    image_rotated = imutils.rotate_bound(image_edge, 45) 
    #cv2.imshow("5-Rotate 45 Degrees", image_rotated)
    
    
    #6 - find contours, get largest one, get extrime points
    cnts = cv2.findContours(image_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea) 
    # determine the most extreme points along the contour
    extLeft = tuple( c [   c[0:, :, 0].argmin()   ] [0] )
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    '''
#    extLeft = tuple(c[c[:, :, 0].argmin()][0])
#    extRight = tuple(c[c[:, :, 0].argmax()][0])
#    extTop = tuple(c[c[:, :, 1].argmin()][0])
#    extBot = tuple(c[c[:, :, 1].argmax()][0])
    '''
    image_rotated_color = imutils.rotate_bound(image_ori, 45)
    image_rotated_color2 = image_rotated_color.copy()
    #위의 두개는 바로 아래에서 이미지에 낙서를 해서 하나더만든거임
    #TODO: 아래의 코드는 확인용이므로 지워도됨 그러나 프로젝트 끝날 때 까지는 살림
    # draw the outline of the object, then draw each of the
    cv2.drawContours(image_rotated_color2, [c], -1, (0, 255, 255), 2)
    cv2.circle(image_rotated_color2, extLeft, 6, (0, 0, 255), -1)
    cv2.circle(image_rotated_color2, extRight, 6, (0, 255, 0), -1)
    cv2.circle(image_rotated_color2, extTop, 6, (255, 0, 0), -1)
    cv2.circle(image_rotated_color2, extBot, 6, (255, 255, 0), -1)
    cv2.imshow("6-contour image", image_rotated_color2)
    
    
    #TODO: 객체가 화면밖으로 나가는순간 max() arg is an empty sequence 에러가 뜨면서 정지하는 버그있음
    #7 - affine 
    #add extra pixel 
    OFFSET = 0
    aextLeft = np.asarray(extLeft)
    aextRight = np.asarray(extRight)
    aextTop = np.asarray(extTop)
    aextBot = np.asarray(extBot)
#    print(aextLeft) #
#    print(aextRight)
#    print(aextTop)
#    print(aextBot)

    aextTop[0] = aextTop[0]+(aextTop[0]-aextLeft[0])
    aextTop[1] = aextTop[1]-(aextLeft[1]-aextTop[1])
    aextRight[0] = aextRight[0]+(aextRight[0]-aextBot[0])
    aextRight[1] = aextRight[1]-(aextBot[1]-aextRight[1])

    aextLeft[0] = aextLeft[0]-int(np.round_(0.1*(aextTop[0]-aextLeft[0])))
    aextLeft[1] = aextLeft[1]+int(np.round_(0.1*(aextLeft[1]-aextTop[1])))
    aextBot[0] = aextBot[0]-int(np.round_(0.1*(aextRight[0]-aextBot[0])))
    aextBot[1] = aextBot[1]+int(np.round_(0.1*(aextBot[1]-aextRight[1])))
    aextLeft += [-OFFSET,0]
    aextRight += [OFFSET,0]
    aextTop += [0,-OFFSET]
    aextBot += [0,OFFSET]
    
    pts1 = np.float32([[578., 589.],[233., 200.],[678., 434.],[399., 124.]])
    pts2 = np.float32([[0,0],[WIDTH,0],[0,HEIGHT],[WIDTH,HEIGHT]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    image_affine = cv2.warpPerspective(image_rotated_color,matrix,(WIDTH,HEIGHT))    
    
    cv2.imshow("7-affine ", image_affine)

    # Display the resulting frame
    cv2.imshow("3-blur", image_blur)
    cv2.imshow("4-edge dection", image_edge)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()



#*********************************************************
#**************************loop***************************
#*********************************************************
    


# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 70  #  BINARY threshold
blurValue = 20  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

framenum = 0    # current frame number
beforecX = 0    # before X point
beforecY = 0    # before Y point
cX = 0          # current X point
cY = 0          # current Y point
beforeVec = 0   # before Vector
beforeAcc = 0   # before Acc
fingerIndex = 0 # finger Number Index


# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


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


################################################################

cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 250, printThreshold)
cv2.namedWindow('blurval')
cv2.createTrackbar('trh2', 'trackbar', blurValue, 50, printThreshold)
while (True):
    
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    blurValue = cv2.getTrackbarPos('trh2', 'trackbar')
    #1 - read
    ret, frame = cap.read()
    #gaussian blur 적용
    frame1 = cv2.blur(frame,(10,10))
    
    #영상을 RGB에서 HSV로 바꿈
    img_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    #살색 히스토그램에 대한 Range적용
    img_hand = cv2.inRange(img_hsv, lower, upper)
    
    #이진화된 이미지의 잡음 제거
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.erode(img_hand, kernel, iterations = 1)
    
    cv2.imshow("result",result)
    #2 - Rotate
    image_rotated = imutils.rotate_bound(frame, 45)
    image_rotated1 = imutils.rotate_bound(result, 45)
    #3 - Affine
    frame = cv2.warpPerspective(image_rotated,matrix,(WIDTH,HEIGHT))
    result = cv2.warpPerspective(image_rotated1,matrix,(WIDTH,HEIGHT))
    #4 - Gary
    image_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    
    
    
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    
    #frame = cv2.flip(frame, 1)  # flip the frame horizontally
    
    
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        #img = img[0:int(cap_region_y_end * frame.shape[0]),
        #            int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue*2+1, blurValue*2+1), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        


        ####################
        thresh = cv2.bitwise_and(thresh,result)
        
        
        #######################
        
        cv2.imshow('ori', thresh)
        
        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        secondMaxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (ording to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
                elif area > secondMaxArea:
                    secondMaxArea = area
                    cii = i

            res = contours[ci]
            hull = cv2.convexHull(res)
                
            drawing = np.zeros(img.shape, np.uint8)
            
            ret,points = getFingerPosition(res,drawing,debug=False)
            
            if ret > 0 and len(points) > 0:  
                points.sort()
                for point in points:
                    cv2.circle(drawing, point, 20, [ 255, 0, 255], 5)
                    cv2.putText(drawing, str(fingerIndex), point, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
                    beforecX, beforecY, beforeVec, beforeAcc, acc, isclick = finger_log.fingerLog(fingerIndex, framenum, beforecX, beforecY, point[0], point[1], beforeVec, beforeAcc)
                    if isclick == True:
                        print(point)
                        isclick = False
                    fingerIndex = fingerIndex + 1                        
                fingerIndex = 0
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal,cnt = calculateFingers(res,drawing)
            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print (cnt)
                    #app('System Events').keystroke(' ')  # simulate pressing blank space
                    

            cv2.imshow('output', drawing)
        
        ##
    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        cap.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print ('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print ('!!!Trigger On!!!')