import cv2
import os

input_path = "./data/11.21/raw/1"
output_path = "./data/11.21/marked"
imgs = os.listdir(input_path)

for img_dir in imgs:
    img = cv2.imread(input_path + "/" + img_dir)
    print("img size:", img.shape)
    cropped = img[550:1550, 400:1900]
    print("cropped size:", cropped.shape)
    cv2.imwrite(output_path + "/" + img_dir, cropped)
