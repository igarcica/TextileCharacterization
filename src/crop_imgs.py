import cv2
import os

data_dir = "./EOS/"
write_dir = "./EOS_cut/"
save_img=False

def crop_img(img):
    print("Original img size: ", img.shape)
    #crop_img = img[420:970,180:710]
    crop_img = img[1400:3575, 50:2950]
    print("Cropped image size: ", crop_img.shape)
    return crop_img

## Use all images in the fodler "dara_dir"
print("all files")
for filename in sorted(os.listdir(data_dir)):
    f = os.path.join(data_dir, filename)
    if os.path.isfile(f) and filename.endswith('.jpg'):
        print("\033[94m Measuring drape area of ", f, "\033[0m ")
        print(filename)
        img = cv2.imread(f)
        c_img = crop_img(img)
        ## Save image
        if(save_img):
            write_image = write_dir + filename
            cv2.imwrite(write_image, c_img)
