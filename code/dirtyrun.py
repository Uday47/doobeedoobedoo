import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import copy
import os
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import sys
from torch.optim.lr_scheduler import StepLR
from model import *

# Constants
classes = [
    'Burglary', 'Normal', 'Fighting', 'Abuse', 'Assault', 'Explosion', 'Arson',
    'Arrest'
]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mean, std = [0.4889, 0.4887, 0.4891], [0.2074, 0.2074, 0.2074]
im_size = 128

model = resnet50(class_num=8).to(device)

os.makedirs('C:/sih/videos/Abuse/', exist_ok=True)




decoder = {}
for i in range(len(classes)):
    decoder[classes[i]] = i
encoder = {}
for i in range(len(classes)):
    encoder[i] = classes[i]

seq = 16
frame_num = 0

path = r"C:\sih\videos\Abuse007_x264_SparkVideo.mp4"
des = r"C:/sih/videos/Abuse/"

cap = cv2.VideoCapture(path)

# Get the frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the total numer of frames in the video.
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
skip = frame_count//seq

# print(frame_count)

# while frame_num + (seq * skip) <= frame_count:
count = 1
frame_number = frame_num
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # optional
success, image = cap.read()
img1 = cv2.resize(image, (128, 128))
while count < seq:
    print(count)
    try:
        frame_number += skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # optional
        success, image = cap.read()
        image = cv2.resize(image, (128, 128))
    except:
        print(path)
    # print(frame_num)
    # print(frame_count)
    img1 = np.append(img1, image, axis=1)
    count += 1
    # frame_num += frame_number
    # k += 1

# cv2.imwrite(des + str(frame_number) +".jpg",img1)
cv2.imwrite("C:/sih/videos/Abuse/image.jpg", img1)
frame_num = frame_number
cv2.imshow('', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = img1



img = cv2.imread("C:/sih/videos/Abuse/image.jpg")
seq_img = list()
for i in range(16):
  img1 = img[:,128*i:128*(i+1),:].astype("uint8")
  img1 = torchvision.transforms.functional.to_tensor(img1)
  img1 = transforms.Normalize(mean,std)(img1)
  img1 = transforms.Resize((im_size,im_size))(img1)
  img1 = img1.to('cuda' if torch.cuda.is_available() else 'cpu')
  seq_img.append(img1)
seq_image = torch.stack(seq_img)
seq_image = seq_image.reshape(3,16,im_size,im_size)
seq_image = seq_image.reshape([1,3, 16, 128, 128])
# print(seq_image.shape)
# seq_image = seq_image.cuda()

from model import resnet50
model = resnet50(class_num=8).to('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(r'C:\sih\models\c3d_11_0.01.h5',map_location=torch.device(device)))
model.eval()
model(seq_image)
pred = (model(seq_image)).argmax()
# tar.append(encoder[pred.item()])
print(encoder[pred.item()])
print(device)
