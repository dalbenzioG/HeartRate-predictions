import os
import numpy as np
from random import sample
import cv2
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from skimage.feature import hog
#
os.chdir('/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/data/thermal')
list_hog_cropped_thermal = []
sample_thermal = []
lst=os.listdir('./')
for i in range(0,len(sorted(lst))):
    im = cv2.imread(lst[i])
    cropped_image = im[80:430 , 10:260]  # Slicing to crop the image
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    # cropped_image = cropped_image.astype(np.uint8)
    min_in = 112
    max_in = np.amax(cropped_image)
    # ret ,im_thresh= cv2.threshold(cropped_image,min_in,255,cv2.THRESH_TOZERO)
    # im_thresh = np.lib.pad(im_thresh, 5, 'constant', constant_values=0)
    # cv2.imwrite(
    #    '/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/croppedFrames/rgb/' + str(
    #       lst[i]), cropped_image)
    # cv2.waitKey()
#     features = cv2.resize(im_thresh, (50,50))
#     hog_image = hog(features, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
#     list_hog_cropped_thermal.append(hog_image)
# #
plt.axis("off")
plt.imshow(cropped_image)
plt.show()
#
# for k in sample(range(len(list_hog_cropped_thermal)),int(round(len(list_hog_cropped_thermal)*(1)))):
#     sample_thermal.append(list_hog_cropped_thermal[k])

# Calculate the mean, std of the complete dataset
import glob
import cv2
import numpy as np
import tqdm
import random

# calculating 3 channel mean and std for image dataset

means = np.array([0, 0, 0], dtype=np.float32)
stds = np.array([0, 0, 0], dtype=np.float32)
total_images = 0
randomly_sample = 200
for f in tqdm.tqdm(random.sample(glob.glob("/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/croppedFrames/rgb/**.jpg", recursive = True), randomly_sample)):
    img = cv2.imread(f)
    means += img.mean(axis=(0,1))
    stds += img.std(axis=(0,1))
    total_images += 1
means = means / (total_images * 255.)
stds = stds / (total_images * 255.)
print("Total images: ", total_images)
print("Means: ", means)
print("Stds: ", stds)

