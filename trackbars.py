# Source: https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
import cv2
import numpy as np

def nothing(x):
    pass

# Variables
show_imgs=True
HSV_segm = False
canny_segm = True

# Load image
resize_percentage=0.3
img = cv2.imread('./EOS_cut/strange_print_v.jpg')
image = cv2.resize(img, (int(img.shape[1]*resize_percentage),int(img.shape[0]*resize_percentage)), interpolation = cv2.INTER_AREA) 
image2 = image

#### Slides for HSV segmentation ####
# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

#### Slides for Canny segmentation ####
# Create a window
cv2.namedWindow('Canny')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('t_lower', 'Canny', 0, 179, nothing)
cv2.createTrackbar('t_upper', 'Canny', 0, 255, nothing)
cv2.createTrackbar('A', 'Canny', 0, 50000, nothing)
cv2.createTrackbar('B', 'Canny', 0, 10000, nothing)
cv2.createTrackbar('i', 'Canny', 0, 150, nothing)
cv2.createTrackbar('get_contour', 'Canny', 0,1, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('t_lower', 'Canny', 50)
cv2.setTrackbarPos('t_upper', 'Canny', 150)
cv2.setTrackbarPos('A', 'Canny', 5000)
cv2.setTrackbarPos('B', 'Canny', 5000)
cv2.setTrackbarPos('i', 'Canny', 0)
cv2.setTrackbarPos('get_contour', 'Canny', 0)

pt_low = pt_up = 0

first_run = True


def HSV_segmentation(hMin, sMin, vMin, hMax, sMax, vMax):
    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result

def canny_segmentation(input_img, t_low, t_up, A, B):

    ## convert the image to grayscale format
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Setting parameter values
    t_lower = t_low  # Lower Threshold
    t_upper = t_up  # Upper threshold
      
    ## Applying the Canny Edge filter
    edge = cv2.Canny(input_img, A, B, apertureSize=5)
    
    #Dilate canny edges to join and close contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    dilated = cv2.dilate(edge, kernel)
    

    return dilated

while(1):

    if(HSV_segm):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')
    
        result = HSV_segmentation(hMin, sMin, vMin, hMax, sMax, vMax)
    
        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax
    
        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    if(canny_segm):
        t_low = cv2.getTrackbarPos('t_lower', 'Canny')
        t_up = cv2.getTrackbarPos('t_upper', 'Canny')
        A = cv2.getTrackbarPos('A', 'Canny')
        B = cv2.getTrackbarPos('B', 'Canny')
        i = cv2.getTrackbarPos('i', 'Canny')
        get_contour = cv2.getTrackbarPos('get_contour', 'Canny')
        
        #img33 = cv2.imread('./DOS/tank_white.jpg')
        img33=img
        image33 = cv2.resize(img33, (int(img33.shape[1]*resize_percentage),int(img33.shape[0]*resize_percentage)), interpolation = cv2.INTER_AREA) 
        result = canny_segmentation(image33, t_low, t_up, A, B)
        

        if((pt_low != t_low) | (pt_up != t_up) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            #first_run = True
            pt_low = t_low
            pt_up = t_up

        # Find largest contour to filter noise
        if(get_contour):
            contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key = cv2.contourArea) #key=len
            cont_img = cv2.drawContours(image2, contour, -1, (0,255,0), 3)
            cv2.imshow('Contour', image2)
            result = cont_img
            print("t_low:", A)
            print("t_up: ", B)
        else:
            image2 = image33

        # Display result image
        cv2.imshow('edge', result)
        cv2.imshow('original', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
