import cv2
import numpy as np
import copy
import math
import finger_log
#from appscript import app

# Environment:
# OS    : Mac OS EL Capitan
# python: 3.5
# opencv: 2.4.13

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
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

########################################################
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


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

# init




while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)


        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (ording to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

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

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
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