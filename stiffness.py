import cv2
import csv
import os
import io
import math
import argparse
import numpy as np

data_dir = "./HCOS/"
write_dir = "./HCOS_res/"

all_files = False #
cloth = "test" #
cloth_dims = (50,90) # Size of object to compute real area
plate_diam = 27 #plate diameter
px_cm_ratio = 370
use_plate = False
#plate_image = data_dir + "brown_plate" + str(plate_diam) + ".jpg" #
#cloth_image = data_dir + cloth + str(plate_diam) + "_2.jpg" #
#write_image = write_dir + cloth + str(plate_diam) + "_res.jpg" #
resize_percentage = 0.3

write_image = write_dir + cloth + "_res.jpg" #
cloth_image = data_dir + cloth + ".jpg" #
csv_file = "./stiffness_data.csv"
#my_file = open(csv_file, "w")
#wr = csv.writer(my_file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)

activate_print = True #
save_img = True #Save images with contour and CSV data
show_imgs = True

dilate = True

#########################################################

# Get image with Aruco layout
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--ar_input", required=True, help="path to input image containing ArUCo layout")
ap.add_argument("-i", "--input", required=True, help="path to input image")
ap.add_argument("-o", "--output", required=False, type=str, help="path to output folder")
ap.add_argument("-p", "--plate", required=True, type=int, help="Plate diameter")
ap.add_argument('-s', "--size", nargs='+', type=int, help='Cloth dimensions')
#ap.add_argument("-tt", "--trial", required=True, type=int, default=1, help="Trial numbber")
# number of correct corners
args = vars(ap.parse_args())

cloth_image = data_dir + args["input"] + ".jpg" 
write_image = write_dir + args["input"] + "_res.jpg"
cloth = args["input"]
cloth_dims = args["size"]
plate_diam = args["plate"]

t_lower = 0 #Lower Canny threshold
t_upper = 400  #Upper Canny threshold

#########################################################

def do_things(img):
    
    img = cv2.resize(img, (int(img.shape[1]*resize_percentage),int(img.shape[0]*resize_percentage)), interpolation = cv2.INTER_AREA) 

    # convert the image to grayscale format
#    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    ## Applying the Canny Edge filter
    edge = cv2.Canny(img, t_lower, t_upper, apertureSize=5)
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
    if(dilate):
        print("Dilated")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        dilated = cv2.dilate(edge, kernel)
        edge = dilated
        if(show_imgs):
            cv2.imshow('dilated', edge)

    # Find largest contour to filter noise
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key = cv2.contourArea) #key=len
    
    contour_img = cv2.drawContours(img, contour, -1, (0,255,0), 3)
    if(show_imgs):
        cv2.imshow("Contours", contour_img)
    #cv2.imwrite('./test2/blue_lines_v.jpg', contour_img)

#    convexHull = cv2.convexHull(contour, False)
#    img2 = cv2.drawContours(img, [convexHull], -1, (255, 0, 0), 2)
#    cv2.imshow("Convex", img2)



    cv2.waitKey(0)

    # Compute perimenter in pixels
    measured_perimeter = cv2.arcLength(contour, True)
    print("Measured perimeter (px): ", measured_perimeter)

    # Compute AREA value
    measured_area = cv2.contourArea(contour)
    print("Measured contour area (px): ", measured_area)

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

def check_measurements_coherence(plate_area, cloth_total_area, cloth_measured_area):
    
    new_plate_area = plate_area
    new_cloth_total_area = cloth_total_area
    new_cloth_measured_area = cloth_measured_area

    if(cloth_total_area < cloth_measured_area):
    #Set cloth_measured_area to total (cloth_total_area is usually < than real cloth area due to usage, elasticity, etc)
        print("\033[93mError: Cloth flat area is less than cloth measured area \033[0m")
        new_cloth_measured_area = cloth_total_area
    if(cloth_measured_area < plate_area):
        print("\033[93mError: Cloth measured area is less than plate area \033[0m")
    if(cloth_total_area < plate_area):
        print("\033[93mError: Cloth flat area is less than plate area \033[0m")

    return new_plate_area, new_cloth_total_area, new_cloth_measured_area



def compute_drape_ratio(plate_area, cloth_total_area, cloth_measured_area):

    # Compute drape ratio
    drape = (cloth_measured_area-plate_area) / (cloth_total_area-plate_area)

    print_info(activate_print, "Dividend: ", cloth_measured_area-plate_area)
    print_info(activate_print, "Divisor: ", cloth_total_area-plate_area)

    return drape

def save_data_csv(drape, measured_area):
    # Read CSV files with groundtruth and results
    #db = csv.reader(open(csv_file))
    #data = []
    #for rows in db:
    #    data.append(rows)
    #my_data = np.genfromtxt(csv_file, delimiter=',')
    #print(my_data)
    #my_data = np.append([my_data], [[3,2,1]])
    #print(my_data)
    #my_data=[1 2 3]
    #np.savetxt("foo.csv", my_data, delimiter=",")

    #headers = ["File", "short_size", "long_size", "drape", "measured_area_cm","plate_diam", "px_cm", "t_lower", "t_upper"]
    #wr.writerow(headers)
    db = csv.reader(open(csv_file))
    database = np.empty([0,10])
    #print(database)
    n=0
    #database = [1,1,1,1,1,1,1,1,1,1]
    for row in db:
        #print("row", row)
        #wr.writerow(row)
        #database[n] = row
        #print(database)
        database = np.vstack([database, row])
        #print("dat", database)
        #n+=1

    data = [cloth, drape, cloth_dims[0], cloth_dims[1], measured_area, plate_diam, px_cm_ratio, t_lower, t_upper, resize_percentage]
    #wr.writerow(data)
    database = np.vstack([database, data])
    #database = np.append(database, [data])
    #database[n]=data

    #print("db", database)
    #my_file2 = open("./hola.csv", "w")
    #wr = csv.writer(my_file2, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    np.savetxt(csv_file, database, delimiter=",", fmt='%s')
    

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
            plate_area_cm, cloth_total_area_cm, cloth_measured_area_cm = check_measurements_coherence(plate_area_cm, cloth_total_area_cm, cloth_measured_area_cm)

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
    print("Cloth real area (cm2): ", cloth_total_area_cm)
    plate_area_cm, cloth_total_area_cm, cloth_measured_area_cm_corr = check_measurements_coherence(plate_area_cm, cloth_total_area_cm, cloth_measured_area_cm)
    
    ## Compute drape
    drape_ratio = compute_drape_ratio(plate_area_cm, cloth_total_area_cm, cloth_measured_area_cm_corr)
    print("\033[92m DRAPE RATIO (%): ", round(drape_ratio*100, 1), " % \033[0m")
    ## Save image
    if(save_img):
        cv2.imwrite(write_image, contour_img)
        save_data_csv(round(drape_ratio*100,1), round(cloth_measured_area_cm))





#########################################################

## REFS
#https://stackoverflow.com/questions/62039403/python-opencv-finding-the-biggest-contour
#https://www.geeksforgeeks.org/python-opencv-canny-function/
#https://stackoverflow.com/questions/43009923/how-to-complete-close-a-contour-in-python-opencv


#- Output failure in case measured cloth area > real cloth area or < plate area
# - Poner imagen borrosa para que detecte mejor el borde y obvie texturas interiores
# - Draw manually contour in image to compare between automatic and manual area
# - Write CSV results (measured cloth in px and cm, cloth real area, used px_cm ratio, used plate diameter)
