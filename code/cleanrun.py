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
import numpy as np
import matplotlib.pyplot as plt
from model import *

#CONSTANTS
CLASSES = [
    'Burglary', 'Normal', 'Fighting', 'Abuse', 'Assault', 'Explosion', 'Arson',
    'Arrest'
]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
IM_SIZE = 128
IMAGE_NAME = "frame.jpg"
NUM_CHANNELS = 3
NUM_CLASSES = len(CLASSES)
TRIM_FRAME_RATE = 15



#ENCODES LABEL NAMES TO INTEGER LABELS
decoder = {}
for i in range(NUM_CLASSES):
    decoder[CLASSES[i]] = i
encoder = {}
for i in range(NUM_CLASSES):
    encoder[i] = CLASSES[i]


#EXTRACTS FRAMES FROM VIDEOS
def preprocessvideo(SEQUENCE_LENGTH, VIDEO_PATH, FRAME_DES):
    trimmed_video = []
    frame_num = 0
    count = 1
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS of the video
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT
                          )  # Get the total numer of frames in the video.
    skip = frame_count // SEQUENCE_LENGTH
    frame_number = frame_num
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image = cap.read()
    cv2.imwrite(os.path.join(FRAME_DES,"snap.jpg"), image) ####
    trimmed_video.append(image)
    out = cv2.VideoWriter('trimmed_video.avi',cv2.VideoWriter_fourcc(*'MJPG'), TRIM_FRAME_RATE, trimmed_video[0].shape[:2][::-1])
    # print(trimmed_video[0].shape[:2])
    for i in range(TRIM_FRAME_RATE):
    	out.write(trimmed_video[0])
    img1 = cv2.resize(image, (IM_SIZE,IM_SIZE))
    while count < SEQUENCE_LENGTH:
        try:
            frame_number += skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, image = cap.read()
            cv2.imwrite(os.path.join(FRAME_DES,"snap" + str(count)) + ".jpg", image) ####
            trimmed_video.append(image)
            for i in range(TRIM_FRAME_RATE):
            	out.write(trimmed_video[count])
            size = trimmed_video[0].size
            image = cv2.resize(image, (IM_SIZE, IM_SIZE))
        except:
            print(path)
        img1 = np.append(img1, image, axis=1)
        count += 1
    cv2.imwrite(os.path.join(FRAME_DES,IMAGE_NAME), img1) # Write sequenced frames as as image

    img = cv2.imread(os.path.join(FRAME_DES,IMAGE_NAME)) # Preprocess frame image
    seq_img = list()
    for i in range(SEQUENCE_LENGTH):
        img1 = img[:, IM_SIZE * i:IM_SIZE * (i + 1), :].astype("uint8")
        img1 = transforms.functional.to_tensor(img1)
        STD = img1.std(dim = [1,2]).numpy()
        MEAN = img1.mean(dim = [1,2]).numpy()
        img1 = transforms.Normalize(MEAN, STD)(img1)
        img1 = transforms.Resize((IM_SIZE, IM_SIZE))(img1)
        img1 = img1.to(DEVICE)
        seq_img.append(img1)
    seq_image = torch.stack(seq_img)
    seq_image = seq_image.reshape(NUM_CHANNELS, SEQUENCE_LENGTH, IM_SIZE, IM_SIZE)
    seq_image = seq_image.reshape([1, NUM_CHANNELS, SEQUENCE_LENGTH, IM_SIZE, IM_SIZE])
    return seq_image, trimmed_video


def infercrime(SEQ_IMAGE, MODEL_PATH):
	model = resnet50(class_num = NUM_CLASSES).to(DEVICE)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
	model.eval()
	model(SEQ_IMAGE)
	pred = (model(SEQ_IMAGE)).argmax()
	return encoder[pred.item()]


