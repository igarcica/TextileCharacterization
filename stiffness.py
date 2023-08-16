import cv2
import os
import math

data_dir = "/home/irene/Desktop/stiffness/"
write_dir = "./AOS/"

all_files = False #
cloth = "test" #
cloth_dims = (17,23) # Size of object to compute real area
plate_diam = 10 #plate diameter
px_cm_ratio = 367
use_plate = False
plate_image = data_dir + "brown_plate" + str(plate_diam) + ".jpg" #
cloth_image = data_dir + cloth + str(plate_diam) + "_2.jpg" #
write_image = write_dir + cloth + str(plate_diam) + "_res.jpg" #
resize_percentage = 0.3

write_image = write_dir + cloth + ".jpg" #
cloth_image = data_dir + cloth + ".jpg" #
#cloth_image = "./test/blue_lines_v.jpg" #

activate_print = True #
save_img = True #
show_imgs = True

#########################################################

def do_things(img):
    
    img = cv2.resize(img, (int(img.shape[1]*resize_percentage),int(img.shape[0]*resize_percentage)), interpolation = cv2.INTER_AREA) 

    # convert the image to grayscale format
#    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Setting parameter values
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold
      
    ## Applying the Canny Edge filter
    edge = cv2.Canny(img, 3000, 150, apertureSize=5)
    if(show_imgs):
        cv2.imshow('original', img)
        cv2.imshow('edge', edge)
    
    ## test canny parameters
    #for i in range(3000,4000,10):
    #    print(i)
    #    edge = cv2.Canny(img, 3000, 150, apertureSize=5)
    #  
    #    cv2.imshow('original', img)
    #    cv2.imshow('edge', edge)
    #    cv2.waitKey(0)
    #    print("hola")
    #cv2.destroyAllWindows()
    
    #Dilate canny edges to join and close contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilated = cv2.dilate(edge, kernel)
    if(show_imgs):
        cv2.imshow('dilated', dilated)

    # Find largest contour to filter noise
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key = cv2.contourArea) #key=len
    
    contour_img = cv2.drawContours(img, contour, -1, (0,255,0), 3)
    if(show_imgs):
        cv2.imshow("Contours", contour_img)
    #cv2.imwrite('./test2/blue_lines_v.jpg', contour_img)

#    convexHull = cv2.convexHull(contour, False)
#    img2 = cv2.drawContours(img, [convexHull], -1, (255, 0, 0), 2)
#    cv2.imshow("Convex", img2)



    cv2.waitKey(0)


    # Compute AREA value
    measured_area = cv2.contourArea(contour)
    print("Measured contour area: ", measured_area)

    return measured_area, contour_img



def get_px_cm_ratio(measured_plate_area):

    plate_real_area_cm = math.pi*pow((plate_diam/2),2)
    px_cm = plate_measured_area / plate_real_area_cm
    print("PX / CM: ", px_cm)

    ## Comparison
#    plate_measured_area_cm = plate_measured_area/px_cm_ratio
#    plate_measured_r_cm = math.sqrt(plate_measured_area_cm/math.pi)
#    print("Measured plate radius (cm): ", plate_measured_r_cm)
#    print("Measured plate area (cm2): ", plate_measured_area_cm)
#    print("Real plate area (cm2): ", plate_real_area_cm)
#
#    print_info(activate_print, "Real plate area (cm): ", real_plate_area_cm)
#    px_cm = measured_plate_area / real_plate_area_cm
#
#    measured_plate_area_cm = measured_plate_area / px_cm
#    print_info(activate_print,"Measured plate area (cm): ", measured_plate_area_cm)
#
#    measured_plate_r_cm = math.sqrt(measured_plate_area_cm/math.pi)
#    print_info(activate_print,"Measured plate radius (cm): ", measured_plate_r_cm)
#
#    print("PX / CM: ", px_cm)

    return px_cm

def compute_drape_ratio(plate_area, cloth_total_area, cloth_measured_area):

    # Compute drape ratio
    drape = (cloth_measured_area-plate_area) / (cloth_total_area-plate_area)

    return drape


def print_info(activate, arg1, arg2=""):
    if(activate):
        print(arg1, arg2)


#########################################################

if(use_plate): ## Get px cm ratio using the plate image as reference
    print("\033[94m Measuring px cm ratio...\033[0m")
    plate_img = cv2.imread(plate_image)
    plate_measured_area, contour_img = do_things(plate_img)
    
    px_cm = get_px_cm_ratio(plate_measured_area)
    plate_area_cm = plate_measured_area_cm
else: ## Use the given px cm ratio + measure real plate area
    plate_real_area_cm = math.pi*pow((plate_diam/2),2)
    plate_area_cm = plate_real_area_cm


## Use all images in the fodler "dara_dir"
if(all_files):
    print("all files")
    for filename in sorted(os.listdir(data_dir)):
        f = os.path.join(data_dir, filename)
        if os.path.isfile(f) and filename.endswith('.jpg'):
            print("\033[94m Measuring drape area of ", f, "\033[0m ")
            img = cv2.imread(f)
            cloth_measured_area, contour_img = do_things(img)
            print(filename)
    
            ## Compute areas in cm
            cloth_measured_area_cm = cloth_measured_area/px_cm_ratio
            print("Cloth measured area (cm): ", cloth_measured_area_cm)
            cloth_total_area_cm = cloth_dims[0]*cloth_dims[1]
            print("Plate real area (cm2): ", plate_area_cm)
            print("Cloth real area (cm): ", cloth_total_area_cm)

            ## Compute drape
            drape_ratio = compute_drape_ratio(plate_area_cm, cloth_total_area_cm, cloth_measured_area_cm)
            print("\033[92m DRAPE RATIO (%): ", drape_ratio*100, "% \033[0m")
            ## Save image
            if(save_img):
                write_image = write_dir + "res_"+ filename
                cv2.imwrite(write_image, contour_img)

## Use a unique image ("cloth_image")
else: 
    print("\033[94m Measuring drape area of ", cloth_image, "\033[0m")
    img = cv2.imread(cloth_image)
    cloth_measured_area, contour_img = do_things(img)
    
    ## Compute areas in cm
    cloth_measured_area_cm = cloth_measured_area/px_cm_ratio
    print("Cloth measured area (cm): ", cloth_measured_area_cm)
    cloth_total_area_cm = cloth_dims[0]*cloth_dims[1]
    print("Plate real area (cm2): ", plate_area_cm)
    print("Cloth real area (cm): ", cloth_total_area_cm)
    
    ## Compute drape
    drape_ratio = compute_drape_ratio(plate_area_cm, cloth_total_area_cm, cloth_measured_area_cm)
    print("\033[92m DRAPE RATIO (%): ", drape_ratio*100, " % \033[0m")
    ## Save image
    if(save_img):
        cv2.imwrite(write_image, contour_img)





#########################################################

## REFS
#https://stackoverflow.com/questions/62039403/python-opencv-finding-the-biggest-contour
#https://www.geeksforgeeks.org/python-opencv-canny-function/
#https://stackoverflow.com/questions/43009923/how-to-complete-close-a-contour-in-python-opencv


## Process / Results
# Pink sample has a lot of texture that is detected as edge by canny -> Necessary to change canny parameters. -> Solved using dilation and max area contour.
# Blue sample detects clearly the contour but as two contours, so a dilation is necessary to join them in 1 contour and get the area.
# Red sample finds well the contour but also some edges on the inside. Without dilation it detects the outer contour perfectly but with dilation (for blue sample) it joins the edges of the inside. -> Solved as pink sample
# Yellow sample works perfect with and without dilation -> but if we change parameters (for pink sample) maybe not.

###
## To Do
# OK - Close contours to get Area -> Dilation
# Not necessary (previous problem was solved using max area instead of length) - Try another method to close contours -> Convex hull - https://answers.opencv.org/question/74777/how-to-use-approxpolydp-to-close-contours/
# Generalization method for different textured samples + different lightnings

#- Output failure in case measured cloth area > real cloth area or < plate area
# - Poner imagen borrosa para que detecte mejor el borde y obvie texturas interiores
# - Draw manually contour in image to compare between automatic and manual area
# - Write CSV results (measured cloth in px and cm, cloth real area, used px_cm ratio, used plate diameter)
