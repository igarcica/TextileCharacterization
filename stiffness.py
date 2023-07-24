import cv2
import os
import math

data_dir = "./data"
all_files = False
test_image = data_dir + "/flowers10.jpg"
cloth_dims = (17,24)

plate_image = data_dir + "/brown_plate10.jpg"
plate_r = 10/2
#cloth_measured_total_area = 0

#########################################################

def do_things(img):
    #img = cv2.imread("pink.jpg")  # Read image
    img = cv2.resize(img, (int(img.shape[1]*0.1),int(img.shape[0]*0.1)), interpolation = cv2.INTER_AREA) 

    # convert the image to grayscale format
#    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Setting parameter values
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold
      
    ## Applying the Canny Edge filter
    edge = cv2.Canny(img, 3000, 150, apertureSize=5)
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
    cv2.imshow('dilated', dilated)
    
    # Find largest contour to filter noise
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key = cv2.contourArea) #key=len
    
    contourImg = cv2.drawContours(img, contour, -1, (0,255,0), 3)
    cv2.imshow("Contours", contourImg)

#    convexHull = cv2.convexHull(contour, False)
#    img2 = cv2.drawContours(img, [convexHull], -1, (255, 0, 0), 2)
#    cv2.imshow("Convex", img2)



    cv2.waitKey(0)


    # Compute AREA value
    measured_area = cv2.contourArea(contour)
    print("Measured contour area: ", measured_area)

    return measured_area



def get_px_cm_ratio(measured_plate_area):

    real_plate_area_cm = math.pi*pow((plate_r),2)
    print("Real plate area (cm): ", real_plate_area_cm)

    px_cm = measured_plate_area / real_plate_area_cm
    print(px_cm)

    measured_plate_area_cm = measured_plate_area / px_cm
    print("Measured plate area (cm): ", measured_plate_area_cm)

    measured_plate_r_cm = math.sqrt(measured_plate_area_cm/math.pi)
    print("Measured plate radius (cm): ", measured_plate_r_cm)

    print("PX / CM: ", px_cm)

    return px_cm

def compute_drape_ratio(plate_area, cloth_total_area, cloth_measured_area):

    # Compute drape ratio
    drape = (cloth_measured_area-plate_area) / (cloth_total_area-plate_area)

    return drape



#########################################################

# Get px cm ratio using the plate as reference
print("\033[94m Measuring px cm ratio...\033[0m")
print(plate_image)
plate_img = cv2.imread(plate_image)
plate_measured_area = do_things(plate_img)
px_cm = get_px_cm_ratio(plate_measured_area)
plate_measured_area_cm = plate_measured_area/px_cm
plate_real_area_cm = math.pi*pow((plate_r),2)

if not (all_files):
    print("\033[94m Measuring drape area of ", test_image, "\033[0m")
    img = cv2.imread(test_image)
    cloth_measured_area = do_things(img)
    cloth_measured_area_cm = cloth_measured_area/px_cm
    print("Cloth measured area (cm): ", cloth_measured_area_cm)
    cloth_total_area_cm = cloth_dims[0]*cloth_dims[1]
    print("Cloth real area (cm): ", cloth_total_area_cm)
    drape_ratio = compute_drape_ratio(plate_measured_area_cm, cloth_total_area_cm, cloth_measured_area_cm)
    print("\033[92m DRAPE RATIO (%): ", drape_ratio*100, " % \033[0m")

if(all_files):
    print("all files")
    for filename in sorted(os.listdir(data_dir)):
        f = os.path.join(data_dir, filename)
        if os.path.isfile(f) and filename.endswith('.jpg'):
            print("\033[94m Measuring drape area of ", f, "\033[0m ")
            img = cv2.imread(f)
            cloth_measured_area = do_things(img)
            cloth_measured_area_cm = cloth_measured_area/px_cm
            print("Cloth measured area (cm): ", cloth_measured_area_cm)
            cloth_total_area_cm = cloth_dims[0]*cloth_dims[1]
            print("Cloth real area (cm): ", cloth_total_area_cm)
            drape_ratio = compute_drape_ratio(plate_real_area_cm, cloth_total_area_cm, cloth_measured_area_cm)
            print("\033[92m DRAPE RATIO (%): ", drape_ratio*100, "% \033[0m")




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
