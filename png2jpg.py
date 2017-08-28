from glob import glob
import cv2
import os

car_dirs = os.listdir("data/vehicles/")
print(car_dirs)
for image_type in car_dirs:
    pngs = glob('data/vehicles/'+image_type+'/*.png')

    for png in pngs:
        img = cv2.imread(png)
        cv2.imwrite(png[:-3]+'jpg',img)


notcar_dirs = os.listdir("data/non-vehicles/")
print(notcar_dirs)
for image_type in notcar_dirs:
    pngs = glob('data/non-vehicles/'+image_type+'/*.png')

    for png in pngs:
        img = cv2.imread(png)
        cv2.imwrite(png[:-3]+'jpg',img)

