#!/usr/bin/env python
import numpy as np
import cv2
import math
import csv
import contour_annotation as contour_an


def Mouse_Event(event, x, y, flags, param):
    global first, vertices, px_cm_ratio, prev_x, prev_y, contour_area, contour_perimeter
    
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if first: # Vertices array initialization with first point
            vertices = np.array([[x,y]])
            cv2.circle(img, (x, y), 3, (0,255,0), -1)
            cv2.imshow('Draw contour', img)
            first = False
            prev_x = x
            prev_y = y
        else: # Continue filling vertices array with selected points
            vertices = np.append(vertices, np.array([[x,y]]), axis=0)
            cv2.line(img,(prev_x,prev_y),(x,y),(0,255,0),3)
            cv2.imshow('Draw contour', img)
            prev_x = x
            prev_y = y
    #Join last point with initial + Stop getting vertices
    if event == cv2.EVENT_RBUTTONDOWN:
        pts = vertices.reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,0), 3)
        cv2.imshow('Draw contour', img)
        # print("Area (px): ", cv2.contourArea(vertices))
        # print("Area (cm): ", cv2.contourArea(vertices)/px_cm_ratio_area)
        # print("Perimeter (pc): ", cv2.arcLength(vertices, True))
        # print("Perimeter (cm): ", cv2.arcLength(vertices, True)/px_cm_ratio)
        contour_area = cv2.contourArea(vertices)
        contour_perimeter = cv2.arcLength(vertices, True)

#Should be in code for Unfolding/Folding
#pts = np.array( [[10,50], [400,50], [90,200], [50,500]], np.int32)# Let's now reshape our points in form  required by polylines
#pts = pts.reshape((-1,1,2))
#cv2.polylines(image, [pts], True, (0,0,255), 3)

def draw_contour(trial_img):
    print("\033[96m Draw contour... \033[0m")

    global contour_area, contour_perimeter, first
    
    #Set variables
    contour_perimeter = 0
    contour_area = 0
    first = True

    # # Image to draw contour
    # print("\033[94m Reading image: \033[0m", img_path)
    # img = cv2.imread(img_path)
    # # Resize image so it fits on the screen
    # print("Image dim: ", img.shape)
    # scale_percent = resize_percent # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('Draw contour', trial_img)
    
    # filei =open('test.csv','w')
    # writer=csv.writer(filei)

    # set Mouse Callback method
    param = trial_img
    cv2.setMouseCallback('Draw contour', Mouse_Event, param)
    
    print("\033[95m Action required! \033[0m Please, define the contour of the cloth")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\033[96m Ended drawing contour \033[0m")

    # # Save points in csv
    # filei.close()
    # # Create folder for team (input team name)
    # cv2.imwrite('rulebook/results/towel_wrinkle1.jpg', img) # Save with trial number
    
    contour_img = trial_img
    
    return contour_img, contour_perimeter, contour_area#, vertices

## Test code
#img_path = 'test/IMG_20221007_173646.jpg'
#contour_img, contour_perimeter, contour_area = draw_contour(img_path)
#print("Contour area: ", contour_area)
#print("Contour perimeter: ", contour_perimeter)
#cv2.imshow("Contour Image", contour_img)
#cv2.waitKey(0)



## REFS
# https://analyticsindiamag.com/real-time-gui-interactions-with-opencv-in-python/
