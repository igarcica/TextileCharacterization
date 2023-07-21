import cv2
import os

data_directory = "./test"
all_files = False
test_image = "./test/plate.jpg"

total_sample_area = 0
plate_area = 5174.5
plate_area_cm = 153.9380
px_cm = plate_area / plate_area_cm

#########################################################

def do_things(img):
    #img = cv2.imread("pink.jpg")  # Read image
    img = cv2.resize(img, (int(img.shape[1]*0.1),int(img.shape[0]*0.1)), interpolation = cv2.INTER_AREA) 
      
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
    contour = max(contours, key = cv2.contourArea) #len
    
    contourImg = cv2.drawContours(img, contour, -1, (0,255,0), 3)
    cv2.imshow("Contours", contourImg)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Compute AREA value
    measured_area = cv2.contourArea(contour)
    print(measured_area)

    # Compute drape ratio
    #drape = (measured_area-plate_area) / (total_sample_area-plate_area)



#########################################################

if not (all_files):
    img = cv2.imread(test_image)
    do_things(img)

if(all_files):
    print("all files")
    for filename in sorted(os.listdir(data_directory)):
        f = os.path.join(data_directory, filename)
        if os.path.isfile(f) and filename.endswith('.jpg'):
            print(f)
            img = cv2.imread(f)
            do_things(img)


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
