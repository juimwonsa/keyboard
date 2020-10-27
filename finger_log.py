# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 02:03:38 2020


@author: gmlgn
"""

import math

def fingerLog(fingerIndex, framenum, beforecX, beforecY, cX, cY, beforeVec, beforeAcc):
    
    #로그


    f = open('./log/test.txt','a')
    
    vecX = cX - beforecX
    vecY = cY - beforecY
    ori = None
    

    #
    acc = getVec(cX, cY, beforecX, beforecY) + beforeAcc
    if(isVaild(vecX, vecY)):
        
        if(vecY > 0):
            ori = "down"
        else:
            ori = "up"
            acc = 0
    else:
        ori = "unvaild"
    
    f.write(  "Frame   : " + str(framenum) + "\n" + "BeforeX : " + str(beforecX)+ "\n" 
            + "BeforeY : " + str(beforecY) + "\n" + "CurrentX: " + str(cX) + "\n" 
            + "CurrentY: " + str(cY) + "\n" + ori + "\n" 
            + "acc     : " + str(acc) + "\n" + "fingerIndex:"+str(fingerIndex)+"\n\n")
    
    f.close()
    
    isclick = isClicked(ori, acc)
    
    return cX, cY, beforeVec, beforeAcc, acc, isclick

def isVaild(vecX, vecY):
    if((vecX<30) and (vecY<30)):
        return True

def isClicked(ori, acc):
    if((ori == "down") & (acc > 10)):
        return True

def getVec(cX, cY, beforecX, beforecY):
    return math.sqrt((cX-beforecX)**2 + 3*((cY-beforecY)**2))
