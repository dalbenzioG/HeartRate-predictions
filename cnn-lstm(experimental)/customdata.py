import numpy
import torch
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision

from torch.utils.data import DataLoader,Dataset
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

#df= pd.read_csv('/home/gabriella/Documents/PhD-IVS/Courses/Advanced Topic in AI for Intelligent System/cropped/target/features_6.csv')
#train_set,test_set=train_test_split(df,test_size=0.1)

#img_folder='/media/gabriella/New Volume2/Dataset-2-conditions/User6/conditions/abnormal/thermal/'
#BATCH_SIZE=1


class ImageDataset(Dataset) :
    def __init__(self , csv , img_folder , transform) :
        self.csv = csv
        self.transform = transform
        self.img_folder = img_folder

        self.image_names = self.csv[:]['Name']
        self.labels = numpy.array(self.csv['Labels'])

    def __len__(self) :
        return len(self.image_names)

    def __getitem__(self , index) :
        image = cv2.imread(self.img_folder + self.image_names.iloc[index])
        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)
        image = self.transform(image)
        targets = self.labels[index]

        # sample = {'image' : image , 'labels' : targets}

        return image, targets


# train_transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((128, 128)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomRotation(degrees=45),
#                 transforms.ToTensor()])
#
# test_transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((128, 128)),
#                 transforms.ToTensor()])
#
# train_dataset=ImageDataset(train_set,img_folder,train_transform)
# test_dataset=ImageDataset(test_set,img_folder,test_transform)
#
# train_dataloader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )
#
# test_dataloader = DataLoader(
#     test_dataset,
#     batch_size=4,
#     shuffle=True
# )
#
# def imshow(inp, title=None):
#     """imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     plt.show()
#

# Get a batch of training data
#images = next(iter(test_dataloader))

# Make a grid from batch
#output = torchvision.utils.make_grid(images['image'])

#imshow(output)





