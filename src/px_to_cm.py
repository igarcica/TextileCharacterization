#!/usr/bin/env python
import numpy as np
import cv2
import math
import csv

activate_print=True
show_imgs = True
crop_images= False


##################################################################################################
## UTIL FUNCTIONS

def print_info(activate, arg1, arg2="", arg3="", arg4="", arg5="", arg6=""):
    if(activate):
        print(str(arg1) + str(arg2) + str(arg3) + str(arg4) + str(arg5) + str(arg6))

#def ordenar_puntos(puntos):
#    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
#    y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
#    x1_order = y_order[:2]
#    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
#    x2_order = y_order[2:4]
#    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])
#    
#    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

###############################
def transform_perspective(aruco_img):
    print("\033[96m Transforming perspective... \033[0m")

    pixel_cm_ratio = 0
    pixel_cm_area_ratio = 0

    # print("\033[94m Reading image with Aruco layout: \033[0m", aruco_img_path)
    # img = cv2.imread(aruco_img_path) # Load image with aruco layout
    # print_info(activate_print, "Original image dim: ", img.shape)
    # #Resize image to fit screen
    # scale_percent = resize_percentage # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    # print_info(activate_print, "Resized image to ", resize_percentage, "% -> New dim: ", img.shape)
    
    # Load Aruco detector
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    
    # Get Aruco marker
    corners, ids, _ = cv2.aruco.detectMarkers(aruco_img, aruco_dict, parameters=parameters)
    
    # for (markerCorner, markerID) in zip(corners, ids):
    #     # extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
    #     if(markerID==10 or markerID==12 or markerID==16 or markerID==18):
    #         corners = markerCorner.reshape((4, 2))
    #         (topLeft, topRight, bottomRight, bottomLeft) = corners
    #         # convert each of the (x, y)-coordinate pairs to integers
    #         topRight = (int(topRight[0]), int(topRight[1]))
    #         bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    #         bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    #         topLeft = (int(topLeft[0]), int(topLeft[1]))
    #         # draw the bounding box of the ArUCo detection
    #         cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
    #         cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
    #         cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
    #         cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)
    #         # compute and draw the center (x, y)-coordinates of the
    #         # ArUco marker
    #         cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    #         cY = int((topLeft[1] + bottomRight[1]) / 2.0)
    #         cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)
    #         # draw the ArUco marker ID on the img
    #         cv2.putText(img, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         # show the output img
    #         if(markerID==18):#12):
    #             topL_x = cX
    #             topL_y = cY
    #         if(markerID==16):#18
    #             topR_x = cX
    #             topR_y = cY
    #         if(markerID==12):#10
    #             botL_x = cX
    #             botL_y = cY
    #         if(markerID==10):#16(umich)
    #             botR_x = cX
    #             botR_y = cY
    
    # print("Detecting corner markers")
    # cv2.imshow("Frame", img)
    # cv2.waitKey(0)
    
    # print("Correcting image perspective")
    # cv2.circle(img, (topL_x, topL_y), 4, (0, 255, 0), -1)
    # resta = topR_x-topL_x
    # pts1 = np.float32([[topL_x, topL_y],[topR_x, topR_y],[botL_x, botL_y],[botR_x,botR_y]])
    # pts2 = np.float32([[topL_x,topL_y],[topL_x+resta,topL_y],[topL_x,topL_y+resta],[topL_x+resta,topL_y+resta]])
    # M = cv2.getPerspectiveTransform(pts1,pts2)
    # dst = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)

    # print("Getting px/cm ratio")
    # ## Get center Aruco marker and compute px/cm ratio
    # corners, ids, _ = cv2.aruco.detectMarkers(dst, aruco_dict, parameters=parameters)
    print_info(activate_print, "Location of aruco corners: ", corners)
    print_info(activate_print,"Aruco IDs: ", ids)
    for (markerCorner, markerID) in zip(corners, ids):
        if(markerID==10): #center marker 14
            # Draw polygon around the marker
            int_corners = np.int0(markerCorner)
            cv2.polylines(aruco_img, int_corners, True, (0, 255, 0), 5)
            if(show_imgs):
                cv2.imshow('aruco', aruco_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Aruco Perimeter
            aruco_perimeter = cv2.arcLength(markerCorner[0], True) # Aruco perimeter size in pixels
            
            # Pixel to cm ratio -> # Aruco px to cm relation based on detected perimeter size
            pixel_cm_ratio = aruco_perimeter / 20 # 20 is the Aruco perimeter in cm
    
            #Aruco area
            aruco_area = cv2.contourArea(corners[0])
            pixel_cm_area_ratio = aruco_area / 25 # 25 is the Aruco area in cm
            print("\033[33m --> px to cm ratio: ", pixel_cm_ratio, "\033[0m")
            print("\033[33m  --> px to cm AREA ratio: ", pixel_cm_area_ratio, "\033[0m")

    # #Debug
    # x,y,w,h = cv2.boundingRect(corners[0])
    # rect_area = w*h
    # print("Rect: ", rect_area)
    # print(rect_area/pixel_cm_ratio)
    # print(w)
    # print(w/pixel_cm_ratio)
    # print(h)
    # print(h/pixel_cm_ratio)
    # pixel_cm_ratio = rect_area / 25
    # print("RAtio: ", pixel_cm_ratio)
    # print(rect_area)
    # print(rect_area/pixel_cm_ratio)
    # print(w)
    # print(w/pixel_cm_ratio)
    # print(h)
    # print(h/pixel_cm_ratio)

    # # Draw objects boundaries
    # for cnt in contours:
    #     # Get rect
    #     rect = cv2.minAreaRect(cnt)
    #     (x, y), (w, h), angle = rect
    #     # Get Width and Height of the Objects by applying the Ratio pixel to cm
    #     object_width = w / pixel_cm_ratio
    #     object_height = h / pixel_cm_ratio

    #     cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    #     cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)


    return pixel_cm_ratio, pixel_cm_area_ratio

def get_px_cm_ratio(aruco_img_path, resize_percentage):
    print("\033[96m Getting px to cm ratio... \033[0m")

    pixel_cm_ratio = 0
    pixel_cm_area_ratio = 0

    print("\033[94m Reading image with Aruco layout: \033[0m", aruco_img_path)
    img = cv2.imread(aruco_img_path) # Load image with aruco layout
    print_info(activate_print, "Original image dim: ", img.shape)
    #Resize image to fit screen
    img = cv2.resize(img, (int(img.shape[1]*resize_percentage),int(img.shape[0]*resize_percentage)), interpolation = cv2.INTER_AREA) 
    print_info(activate_print, "Resized image to ", resize_percentage*100, "% -> New dim: ", img.shape)
    
    # Load Aruco detector
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    
    # Get Aruco marker
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    
    print_info(activate_print, "Location of aruco corners: ", corners)
    print_info(activate_print,"Aruco IDs: ", ids)
   
    for (markerCorner, markerID) in zip(corners, ids):
        if(markerID==14):
            # Draw polygon around the marker
            print(markerID)
            print(markerCorner)
            int_corners = np.int0(markerCorner)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)
            if(show_imgs):
                cv2.imshow('aruco', img)
                cv2.waitKey(0)
            
            # Aruco Perimeter
            aruco_perimeter = cv2.arcLength(markerCorner, True)
            
            # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / 20 # 20 is the Aruco perimeter in cm
            print("Aruco perimeter: ", aruco_perimeter)
            print("\033[33m --> px to cm ratio: \033[0m", pixel_cm_ratio)
    
            #Aruco area
            aruco_area = cv2.contourArea(markerCorner)#corners[0])
            pixel_cm_area_ratio = aruco_area / 25 # 20 is the Aruco perimeter in cm
            print("Aruco area: ", aruco_area)
            print("\033[33m  --> px to cm AREA ratio: \033[0m", pixel_cm_area_ratio)


    return pixel_cm_ratio, pixel_cm_area_ratio


##################################################################################################
def crop_img(img):
    print("Original img size: ", img.shape)
    #crop_img = img[1450:3600, 10:3000]
    crop_img = img[0:1575, 50:2950]
    print("Cropped image size: ", crop_img.shape)
    return crop_img

## Test code
test_img_path = './EOS_cut/aruco.jpg'
if(crop_images):
    imgg = cv2.imread(test_img_path)
    imgc = crop_img(imgg)
    write_image = './EOS_cut/aruco.jpg'
    cv2.imwrite(write_image, imgc)
else:
    px_cm_ratio, px_cm_area_ratio = get_px_cm_ratio(test_img_path, 0.3)
#px_cm_ratio, px_cm_area_ratio = transform_perspective(test_img_path,30)



## REFS
# https://pysource.com/2021/05/28/measure-size-of-an-object-with-opencv-aruco-marker-and-python
# https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
