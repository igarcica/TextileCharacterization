# Draw contour
# Measure area in cm
# Compute unfolding error --> Given the object name and a dictionary
# Based on threshold, compute points

import numpy as np
import pandas as pd
import cv2
import argparse
import os
import sys
sys.path.insert(1, './px_cm/')
import px_to_cm
import contour_annotation as contour_an
import manipulation_scoring as scoring

resize_percent = 130
activate_print=False
test=False
define_canonical = False

# Get image with Aruco layout
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--ar_input", required=True, help="path to input image containing ArUCo layout")
#ap.add_argument("-x", "--can_input", required=True, help="path to input image containing flat cloth (canonical)")
ap.add_argument("-i", "--input", required=True, help="path to input image containing result of the trial (folded or flat cloth)")
ap.add_argument("-o", "--team", required=True, type=str, default="Team", help="Team name")
ap.add_argument("-t", "--task", required=True, type=str, default="u", help="Task to score: Task 2.1 Unfolding (u) ot Task 2.2. Folding (f)")
ap.add_argument("-nt", "--trial", required=True, type=int, default=1, help="Trial number")
ap.add_argument("-obj", "--object", type=str, default="med_towel", help="Object from the Household Cloth Object Set")
#first or second fold for task folding (o f1, f2)
args = vars(ap.parse_args())


if not os.path.exists(args["team"]):
#    os.mkdir("./ " + args["team"])
    os.mkdir(args["team"])
team = args["team"]
print(team)

trial = args["trial"]

if args["task"] != "u" and args["task"] != "f" and args["task"] != "f1" and args["task"] != "f2":
    print("[INFO] Not a manipulation task. Please select unfolding (u) or folding (f). ")
    sys.exit(0)
else:
    if args["task"] == "u":
        print("Evaluating Task 2.1 Unfolding")
        task="Unfolding"
        tk="u"
    else:
        print("Evaluating Task 2.2 Folding")
        task="Folding"
        if args["task"] == "f1":
            tk="f1"
        else:
            tk="f2"

#Household Cloth Object Set dictionary
CLOTH_SIZE = {
        "small_towel": (30,50),
        "med_towel": (50,90),
        "big_towel":(90, 150),
        "bedhseet": (160, 280),
        "fit_bedsheet": (90, 200),
        "sq_pillowcase": (60, 60),
        "rect_pillowcase": (45, 110),
        "rect_tablecloth": (170, 250),
        "round_tablecloth": 200,
        "cotton_napkin": (50, 50),
        "linen_napkin": (50, 50),
        "towel_rag": (50, 70),
        "linen_rag": (50, 70),
        "waffle_rag": (50, 70),
        "chekered_rag": (50, 70),
        "paper": (21, 29.7)
        }

if CLOTH_SIZE.get(args["object"], None) is None:
        print("[INFO] Object '{}' is not from the Household Cloth Object Set".format(args["object"]))
        print("\033[95mSelect one of the following list: \033[0m ")
        for name in CLOTH_SIZE:
            print(" ", name)
        sys.exit(0)
object_dims = CLOTH_SIZE.get(args["object"], None)

if test:
    team_trials_path = "test/"
    output_path="test/scoring/"
    trial_img_path=team_trials_path+args["input"]
    aruco_img_path=team_trials_path+args["ar_input"]
else: 
    team_trials_path = "teams/"+team+"/"+task+"/"
    output_path="teams/"+team+"/"+task+"/scoring/"
    trial_img_path=team_trials_path+args["input"]
    aruco_img_path=team_trials_path+args["ar_input"]
print("Input path: ", team_trials_path)
print("Output path: ", output_path)


#my_file = open("vertices.csv", "wb")
#vertices_wr = csv.writer(my_file, delimiter=",")

##############################################################
## UTIL FUNCTIONS

def print_info(activate, arg1, arg2="", arg3="", arg4="", arg5="", arg6=""):
    if(activate):
        print(str(arg1) + str(arg2) + str(arg3) + str(arg4) + str(arg5) + str(arg6))

##### RESIZE IMAGE to fit screen and draw GT #####
def decrease_image(img_path):
    print("\033[94m Reading image: \033[0m", img_path)
    print("\033[94m Resizing image to a ", resize_percent, " % \033[0m")
    img = cv2.imread(img_path) # Load image with aruco layout
    print_info(activate_print, "Original image dim: ", img.shape)
    width = int(img.shape[1] * resize_percent / 100)
    height = int(img.shape[0] * resize_percent / 100)
    dim = (width, height)
    res_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    print_info(activate_print, "Resized image to ", resize_percent, "% -> New dim: ", res_img.shape)

    return res_img

def increase_image(img):
    print("\033[94m Restoring image size: \033[0m")
    #print_info(activate_print, "Original image dim: ", img.shape)
    #print("\033[94m Resizing image to a ", resize_percent, " % \033[0m")
    width = int(img.shape[1] / (resize_percent/100))
    height = int(img.shape[0] / (resize_percent/100))
    dim = (width, height)
    res_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    print_info(activate_print, "Resized image: ", res_img.shape)

    return res_img

##### SAVE RESULTS #####
def save_results(contour_vert, img, results_data):
    print("\033[94m Saving results \033[0m")

    # Save measured results
    np.savetxt(output_path+str(trial)+"_"+str(tk)+"_results_data"+".csv", results_data, fmt='%s', delimiter=",")

    # Save vertices of defined contour
    np.savetxt(output_path+str(trial)+"_"+str(tk)+"_vertices.csv", contour_vert, fmt='%s', delimiter=",")   

    # Save image with defined contour
    error = results_data[2][1]
    text_loc = img.shape
    text = "Error: " + str(round(abs(error),2)) +"%"
    cv2.putText(img, text, (int((text_loc[1]/2)-100), int(text_loc[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imwrite(output_path+str(trial)+"_"+str(tk)+"_result.png", img)                



##############################################################

##### RESIZING IMAGES TO FIT SCREEN #####
aruco_img = decrease_image(aruco_img_path)
trial_img = decrease_image(trial_img_path)

##### GET PX/CM RATIO ####
print("\033[32m GETTING PIXEL/CENTIMETER RATIO \033[0m")
px_cm_ratio, px_cm_area_ratio = px_to_cm.transform_perspective(aruco_img)


# #### CANONICAL ####
# if define_canonical:
#     # Get init area in pixels and save image and value for other trials
#     # Get starting configuration cloth perimeter and area in pixels
#     print("\032[32m DEFINE INITIAL CONFIG CONTOUR \033[0m")
#     # trial_img_path=team_trials_path+"trial_start_"+str(trial)+".png"
#     trial_img_path=team_trials_path+args["can_input"]
#     can_contour_img, can_cloth_per_px, can_cloth_area_px, can_vertices = contour_an.draw_contour(trial_img_path, resize_percent)
#     print_info(activate_print, "CANONICAL Measured cloth perimeter (px): ", can_cloth_per_px)
#     print_info(activate_print, "CANONICAL Measured cloth area (px): ", can_cloth_area_px)

#     # Compute perimeter and area in cm
#     can_cloth_per_cm = can_cloth_per_px/px_cm_ratio
#     can_cloth_area_cm = can_cloth_area_px/px_cm_area_ratio
#     print("CANONICAL Measured cloth perimeter (cm): ", can_cloth_per_cm)
#     print("CANONICAL Measured cloth area (cm): ", can_cloth_area_cm)

#     #Save defined contour for next trials
#     np.savetxt("can_vertices.csv", can_vertices.astype(int), fmt="%s", delimiter=",")
#     #print("Defined can vertices: ", can_vertices)

# else:
#     ## Get saved canonical contour
#     print("\032[32m GETTING INITIAL CONFIG CONTOUR \033[0m")
#     #can_vertices = np.genfromtxt('can_vertices.csv', delimiter=',')
#     #can_vertices = np.array([[476, 361], [492, 441], [567, 380]])
#     df = pd.read_csv('can_vertices.csv', header=None)
#     can_vertices = df.to_numpy()
#     #print("Read can vertices: ", can_vertices)
#     can_cloth_area_px = cv2.contourArea(can_vertices)
#     can_cloth_per_px = cv2.arcLength(can_vertices, True)
    
#     print_info(activate_print, "CANONICAL Measured cloth perimeter (px): ", can_cloth_per_px)
#     print_info(activate_print, "CANONICAL Measured cloth area (px): ", can_cloth_area_px)

#     # Compute perimeter and area in cm
#     can_cloth_per_cm = can_cloth_per_px/px_cm_ratio
#     can_cloth_area_cm = can_cloth_area_px/px_cm_area_ratio
#     print("CANONICAL Measured cloth perimeter (cm): ", can_cloth_per_cm)
#     print("CANONICAL Measured cloth area (cm): ", can_cloth_area_cm)

#     # print("Read contour are: ", contour_area)
#     # print("Read contour perimeter: ", contour_perimeter)


#### UNFOLDING ####
# Scoring fol UNFOLDING
if args["task"] == "u":
    # Get starting configuration cloth perimeter and area in pixels
    print("\032[32m DEFINE INITIAL CONFIG CONTOUR \033[0m")
    # trial_img_path=team_trials_path+"trial_start_"+str(trial)+".png"
    u_contour_img, u_cloth_per_px, u_cloth_area_px, u_vertices = contour_an.draw_contour(trial_img)
    
    # print_info(activate_print, "UNFOLDED Measured cloth perimeter (px): ", u_cloth_per_px)
    # print_info(activate_print, "UNFOLDED Measured cloth area (px): ", u_cloth_area_px)

    # Compute perimeter and area in cm
    print("\033[94mGetting the cloth perimeter and area\033[0m")
    u_cloth_per_cm = u_cloth_per_px/px_cm_ratio
    u_cloth_area_cm = u_cloth_area_px/px_cm_area_ratio
    # print("UNFOLDED Measured cloth perimeter (cm): ", u_cloth_per_cm)
    # print("UNFOLDED Measured cloth area (cm): ", u_cloth_area_cm)
    
    #call unfolding scoring code
    print("\033[94mScoring Task 2.1. Unfolding!\033[0m")
    u_percentage_area_error = scoring.unfolding(object_dims, u_cloth_per_px, u_cloth_per_cm, u_cloth_area_px, u_cloth_area_cm)

    # ## Save results
    # np.savetxt(output_path+"u_vertices.csv", u_vertices, fmt='%s', delimiter=",")   # Save vertices of defined contour
    # text_loc = u_contour_img.shape
    # #cv2.rectangle(u_contour_img, (int((text_loc[1]/2)-100), int(text_loc[0]-50)), (int(text_loc[1]/2), int(text_loc[0])), (255,255,255), -1)
    # text = "Error: " + str(round(abs(u_percentage_area_error),2)) +"%"
    # cv2.putText(u_contour_img, text, (int((text_loc[1]/2)-100), int(text_loc[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    # cv2.imwrite(output_path+"u_result.jpg", u_contour_img)                # Save image with defined contour

    # Save results
    u_results = [["Measured perimeter (cm)", u_cloth_per_cm], ["Measured area (cm)", u_cloth_area_cm], ["Area error (%)", u_percentage_area_error]] #measured perimeter (cm), measured area (cm), error area (%)
    save_results(u_vertices, u_contour_img, u_results)

#### FIRST FOLD (A/2) ####
# Scoring for FIRST FOLD
elif args["task"] == "f1":
    # Get cloth perimeter and area in pixels
    print("\033[32m DEFINE CONTOUR FIRST FOLD \033[0m")
    f1_contour_img, f1_cloth_per_px, f1_cloth_area_px, f1_vertices = contour_an.draw_contour(trial_img)
    print_info(activate_print, "FIRST measured cloth perimeter (px): ", f1_cloth_per_px)
    print_info(activate_print, "FIRST measured cloth area (px): ", f1_cloth_area_px)

    # Compute perimeter and area in cm
    #print("\033[94mGetting the cloth perimeter and area\033[0m")
    f1_cloth_per_cm = f1_cloth_per_px/px_cm_ratio
    f1_cloth_area_cm = f1_cloth_area_px/px_cm_area_ratio
    print("FIRST FOLD Measured cloth perimeter (cm): ", f1_cloth_per_cm)
    print("FIRST FOLD Measured cloth area (cm): ", f1_cloth_area_cm)

    print("\033[94m Scoring Task 2.2. Folding! - First fold (A/2) \033[0m")
    f1_percentage_area_error = scoring.folding(object_dims, f1_cloth_per_px, f1_cloth_per_cm, f1_cloth_area_px, f1_cloth_area_cm)

    # ## Save results
    # np.savetxt(output_path+"f1_vertices.csv", f1_vertices, fmt='%s', delimiter=",") # Save defined contour
    # text_loc = f1_contour_img.shape
    # text = "Error: " + str(round(abs(f1_percentage_area_error),2)) +"%"
    # cv2.putText(f1_contour_img, text, (int((text_loc[1]/2)-100), int(text_loc[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    # cv2.imwrite(output_path+"f1_result.jpg", f1_contour_img)                # Save image with defined contour

    # Save results
    f1_results = [["Measured perimeter (cm)", f1_cloth_per_cm], ["Measured area (cm)", f1_cloth_area_cm], ["Area error (%)", f1_percentage_area_error]] 
    save_results(f1_vertices, f1_contour_img, f1_results)

#### SECOND FOLD (A/4) ####
# Scoring for SECOND FOLD
elif args["task"] == "f2":
    # Get cloth perimeter and area in pixels
    print("\033[94m DEFINE CONTOUR SECOND FOLD \033[0m")
    # trial_img_path=team_trials_path+"trial_final_"+str(trial)+".png"
    f2_contour_img, f2_cloth_per_px, f2_cloth_area_px, f2_vertices = contour_an.draw_contour(trial_img)
    print_info(activate_print,"SECOND FOLD  Measured cloth perimeter (px): ", f2_cloth_per_px)
    print_info(activate_print,"SECOND FOLD Measured cloth area (px): ", f2_cloth_area_px)

    # Compute perimeter and area in cm
    f2_cloth_per_cm = f2_cloth_per_px/px_cm_ratio
    f2_cloth_area_cm = f2_cloth_area_px/px_cm_area_ratio
    print("SECOND FOLD Measured cloth perimeter (cm): ", f2_cloth_per_cm)
    print("SECOND FOLD Measured cloth area (cm): ", f2_cloth_area_cm)

    print("\033[94m Scoring Task 2.2. Folding! - Second fold (A/4) \033[0m")
    f2_percentage_area_error = scoring.folding2(object_dims, f2_cloth_per_px, f2_cloth_per_cm, f2_cloth_area_px, f2_cloth_area_cm)
    # print("\033[92m Seccond fold perimeter error ", f2_cloth_area_px/(can_cloth_area_px/4), "\033[0m")

    # ## Save results
    # np.savetxt(output_path+"f2_vertices.csv", f2_vertices, fmt='%s', delimiter=",") # Save defined contour
    # text_loc = f2_contour_img.shape
    # text = "Error: " + str(round(abs(f2_percentage_area_error),2)) +"%"
    # cv2.putText(f2_contour_img, text, (int((text_loc[1]/2)-100), int(text_loc[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    # cv2.imwrite(output_path+"f2_result.jpg", f2_contour_img)                # Save image with defined contour

    # Save results
    f2_results = [["Measured perimeter (cm)", f2_cloth_per_cm], ["Measured area (cm)", f2_cloth_area_cm], ["Area error (%)", f2_percentage_area_error]] 
    save_results(f2_vertices, f2_contour_img, f2_results)


# #### SCORING ####
# # Scoring fol UNFOLDING
# if args["task"] == "u":
#     #call unfolding scoring code
#     print("\033[94mScoring Task 2.1. Unfolding!\033[0m")
#     size = (50,90)
#     scoring.unfolding(CLOTH_SIZE.get(args["object"], None), cloth_per_cm)
# # Scoring for FIRST FOLD
# elif args["task"] == "f1":
#     print("\033[94m Scoring Task 2.2. Folding!\033[0m")
#     obj_real_perimeter, obj_real_area, error_perimeter_f1, error_area_f1 = scoring.folding(CLOTH_SIZE.get(args["object"], None), init_cloth_area_px, f1_cloth_per_px, f1_cloth_per_cm, f1_cloth_area_px, f1_cloth_area_cm)
#     print("\033[92m First fold perimeter error ", f1_cloth_area_px/(init_cloth_area_px)/2, "\033[0m")
#     print(f1_cloth_area_px, " / ", (init_cloth_area_px)/2, " / ", f1_cloth_area_px/init_cloth_area_px/2)
# # Scoring for SECOND FOLD
# elif args["task"] == "f":
#     # #call folding scoring code
#     # print("\033[94mScoring Task 2.2. Folding!\033[0m")
#     # obj_real_perimeter, obj_real_area, error_perimeter_f1, error_area_f1 = scoring.folding(CLOTH_SIZE.get(args["object"], None), init_cloth_area_px, f1_cloth_per_px, f1_cloth_per_cm, f1_cloth_area_px, f1_cloth_area_cm)
#     # print("\033[92m First fold perimeter error ", f1_cloth_area_px/(init_cloth_area_px)/2, "\033[0m")
#     # print(f1_cloth_area_px, " / ", (init_cloth_area_px)/2, " / ", f1_cloth_area_px/init_cloth_area_px/2)
#     obj_real_perimeter_f2, obj_real_area_f2, error_perimeter_f2, error_area_f2 = scoring.folding2(CLOTH_SIZE.get(args["object"], None), init_cloth_area_px, f2_cloth_per_px, f2_cloth_per_cm, f2_cloth_area_px, f2_cloth_area_cm)
#     print("\033[92m Seccond fold perimeter error ", f2_cloth_area_px/(init_cloth_area_px/4), "\033[0m")
# else:
#     print("[INFO] Not a manipulation task. Please select unfolding (u) or folding (f). ")
#     sys.exit(0)

# # Save results
# # Contour image
# cv2.imwrite(output_path + "/trial" + str(trial) + "_cont.jpg", contour_img) # Save with trial number
# # Measured perimeter and area
# filei =open(output_path + "/trial" + str(trial) + '.csv','w')
# writer=csv.writer(filei)
# row = [px_cm_ratio, px_cm_area_ratio, cloth_per_px, cloth_area_px, cloth_per_cm, cloth_area_cm]
# writer.writerow(row)
# # Info trial (team, trial, config, object, ...), Ratios, perimeter + area (in px and cm), error, points
# #Towel Area: 4500cm2 = 45cm2, Perimeter: 280cm
# filei.close()

##TODO
# Compare area with initial!
# Take into account the % of elasticity of each cloth. The sizes in CLOTH_SIZE should be with +- a percentage. Error should be 0% when it is inside this tolerance (due to elasticity)
