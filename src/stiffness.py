import cv2
import csv
import sys
import math
import argparse
import numpy as np

data_dir = "./data/"
write_dir = "./results/"

px_cm_ratio = 370 # Pixel to centimeter ratio, obtained with px_to_cm.py
t_lower = 0 # Lower Canny threshold
t_upper = 1000  # Upper Canny threshold
resize_percentage = 0.3 # Resize image so it can be seen in computer

csv_file = "./results/stiffness_data.csv"

activate_print = False # Print complementary info
show_imgs = True # Open contour image
save_img = True # Save images with contour and CSV data

dilate = True

#########################################################

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--aruco", required=True, help="path to aruco image")
ap.add_argument("-i", "--input", required=True, help="path to input image")
#ap.add_argument("-o", "--output", required=False, type=str, help="path to output folder")
ap.add_argument("-p", "--plate", required=True, type=int, help="Plate diameter")
ap.add_argument('-s', "--size", required=True, nargs=2, type=int, help='Cloth dimensions')
args = vars(ap.parse_args())

aruco = args["aruco"]
cloth = args["input"]
plate_diam = args["plate"]
cloth_dims = args["size"]

cloth_image_path = data_dir + args["input"]
write_image_path = write_dir + args["input"]
aruco_image_path = data_dir + args["aruco"]

try:
    with open(cloth_image_path) as f:
        pass
except FileNotFoundError:
    print("\033[91m[ERROR]\033[0m Image {} does not exist".format(args["input"]))
    sys.exit(0)


#########################################################

def measure_draped_area(img):
    ## Obtaines contour of draped cloth through Canny edge filter and measures area
    ## Input: Zenital image of draped cloth
    ## Output: Measured area in pixels and image with detected contour 
    
    img = cv2.resize(img, (int(img.shape[1]*resize_percentage),int(img.shape[0]*resize_percentage)), interpolation = cv2.INTER_AREA) 

    ## convert the image to grayscale format
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    ## Apply the Canny Edge filter
    edge = cv2.Canny(img, t_lower, t_upper, apertureSize=5)
    # if(show_imgs):
    #     cv2.imshow('original', img)
    #     cv2.imshow('edge', edge)
    
    
    #Dilate canny edges to join and close contours
    if(dilate):
        print_info(activate_print, "Dilated")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        dilated = cv2.dilate(edge, kernel)
        edge = dilated
        # if(show_imgs):
        #     cv2.imshow('dilated', edge)

    # Find largest contour to filter noise
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key = cv2.contourArea) #key=len
    
    contour_img = cv2.drawContours(img, contour, -1, (0,255,0), 3)
    if(show_imgs):
        cv2.imshow("Contours", contour_img)

    # convexHull = cv2.convexHull(contour, False)
    # img2 = cv2.drawContours(img, [convexHull], -1, (255, 0, 0), 2)
    # cv2.imshow("Convex", img2)

    cv2.waitKey(0)

    # Compute perimenter in pixels
    measured_perimeter = cv2.arcLength(contour, True)
    print_info(activate_print, "Measured perimeter (px): ", measured_perimeter)

    # Compute AREA value
    measured_area = cv2.contourArea(contour)
    print_info(activate_print, "Measured contour area (px): ", measured_area)

    return measured_area, contour_img

def check_measurements_coherence(plate_area, cloth_total_area, cloth_measured_area):
    ## Checks for coherence between measured area and real areas
    
    new_plate_area = plate_area
    new_cloth_total_area = cloth_total_area
    new_cloth_measured_area = cloth_measured_area

    if(cloth_total_area < cloth_measured_area):
    #Set cloth_measured_area to total (cloth_total_area is usually < than real cloth area due to usage, elasticity, etc)
        print("\033[93m[WARN] Cloth flat area is less than cloth measured area \033[0m")
        new_cloth_measured_area = cloth_total_area
    if(cloth_measured_area < plate_area):
        print("\033[93m[WARN] Cloth measured area is less than plate area \033[0m")
    if(cloth_total_area < plate_area):
        print("\033[93m[WARN] Cloth flat area is less than plate area \033[0m")

    return new_plate_area, new_cloth_total_area, new_cloth_measured_area

def compute_drape_ratio(plate_area, cloth_total_area, cloth_measured_area):
    ## Computes stiffness value through paper's formula
    ## Input: plate area in centimeters (A2), cloth real area in centimeters (A1) and cloth measured area in centimeters (A3)

    # Compute drape ratio
    drape = (cloth_measured_area-plate_area) / (cloth_total_area-plate_area)

    print_info(activate_print, "Dividend: ", cloth_measured_area-plate_area)
    print_info(activate_print, "Divisor: ", cloth_total_area-plate_area)

    return drape

def save_data_csv(drape, measured_area):

    db = csv.reader(open(csv_file))
    database = np.empty([0,10])
    for row in db:
        database = np.vstack([database, row])

    data = [cloth, drape, cloth_dims[0], cloth_dims[1], measured_area, plate_diam, px_cm_ratio, t_lower, t_upper, resize_percentage]
    database = np.vstack([database, data])

    np.savetxt(csv_file, database, delimiter=",", fmt='%s')

def get_px_cm_ratio(aruco_img_path):
    ## Obtains the pixel to centimeter ratio to have a common unit (centimeters) for all camera brands and setups
    ## Input: Image path of aruco pattern
    ## Output: Pixel to centimeter ratio (area)

    print("\033[94mGetting px to cm ratio from \033[0m", aruco_img_path)

    pixel_cm_ratio = 0
    pixel_cm_area_ratio = 0

    img = cv2.imread(aruco_img_path) # Load image with aruco layout
    img = cv2.resize(img, (int(img.shape[1]*resize_percentage),int(img.shape[0]*resize_percentage)), interpolation = cv2.INTER_AREA) #Resize image to fit screen
    
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
            int_corners = np.int0(markerCorner)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)
            # if(show_imgs):
            #     cv2.imshow('aruco', img)
            #     cv2.waitKey(0)
            
            # Aruco Perimeter
            aruco_perimeter = cv2.arcLength(markerCorner, True)
            
            # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / 20 # 20 is the Aruco perimeter in cm
            print_info(activate_print, "Aruco perimeter: ", aruco_perimeter)
            print_info(activate_print, "\033[33m --> px to cm ratio: \033[0m", pixel_cm_ratio)
    
            #Aruco area
            aruco_area = cv2.contourArea(markerCorner)#corners[0])
            pixel_cm_area_ratio = aruco_area / 25 # 20 is the Aruco perimeter in cm
            print_info(activate_print, "Aruco area: ", aruco_area)
            print_info(activate_print, "\033[33m  --> px to cm AREA ratio: \033[0m", pixel_cm_area_ratio)


    return int(pixel_cm_area_ratio)

def print_info(activate, arg1, arg2=""):
    if(activate):
        print(arg1, arg2)


#########################################################

## Compute plate area (A2)
plate_real_area_cm = math.pi*pow((plate_diam/2),2)
plate_area_cm = plate_real_area_cm

## Compute cloth reat area (A1)
cloth_total_area_cm = cloth_dims[0]*cloth_dims[1]

## Measure draped area in pixels
print("\033[94mMeasuring drape area of \033[0m", cloth_image_path)
img = cv2.imread(cloth_image_path)
cloth_measured_area, contour_img = measure_draped_area(img)
    
## Obtain measured rapped area in centimeters (A3)
px_cm_ratio = get_px_cm_ratio(aruco_image_path)
cloth_measured_area_cm = cloth_measured_area/px_cm_ratio
plate_area_cm, cloth_total_area_cm, cloth_measured_area_cm_corr = check_measurements_coherence(plate_area_cm, cloth_total_area_cm, cloth_measured_area_cm)
print("A1 - Cloth real area (cm2): ", cloth_total_area_cm)
print("A2 - Plate real area (cm2): ", plate_area_cm)
print("A3 - Cloth measured area (cm): ", cloth_measured_area_cm)

## Compute drape
drape_ratio = compute_drape_ratio(plate_area_cm, cloth_total_area_cm, cloth_measured_area_cm_corr)
print("\033[92m ---DRAPE RATIO: ", round(drape_ratio*100, 1), " % ---\033[0m")
## Save image
if(save_img):
    cv2.imwrite(write_image_path, contour_img)
    save_data_csv(round(drape_ratio*100,1), round(cloth_measured_area_cm))





#########################################################

## REFS
#https://stackoverflow.com/questions/62039403/python-opencv-finding-the-biggest-contour
#https://www.geeksforgeeks.org/python-opencv-canny-function/
#https://stackoverflow.com/questions/43009923/how-to-complete-close-a-contour-in-python-opencv
#https://pysource.com/2021/05/28/measure-size-of-an-object-with-opencv-aruco-marker-and-python
#https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/

