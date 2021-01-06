import cv2
import os

path = './Data/Positive'

imgs = os.listdir(path)
for img in imgs:
    save_name = img.split('.')[0] + '_rot180' + '.png'
    this_img = cv2.imread(path + '/' + img)
    this_img = cv2.cvtColor(this_img, cv2.COLOR_BGR2RGB)
    this_img = cv2.rotate(this_img, cv2.ROTATE_180)
    cv2.imwrite(path + '/' + save_name, this_img)