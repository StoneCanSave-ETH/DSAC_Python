from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import math
import time
import torch

from Hypothesis import Hypothesis
from properties import GlobalProperties
import cnn
import TYPE


class RGB_DATASET(Dataset):
    def __init__(self, dataset, inputSize, transform, trainingImages, trainingPatches):
        self.dataset = dataset
        self.inputSize = inputSize
        self.transform = transform
        self.trainingImages = trainingImages
        self.trainingPatches = trainingPatches

    def __len__(self):
        return self.trainingImages * self.trainingPatches

    def __getitem__(self, item):

        # Get raw data
        imgBGR = self.dataset.rand_getBGR(item)
        BGRinfo = self.dataset.rand_bgrFiles[item]

        # Randomly select patch and pixel
        pixel, depth = self.dataset.rand_getRGBpix(i=item, inputSize=self.inputSize)
        label = self.dataset.rand_getcoord(i=item, pix=pixel, depth=depth)
        data = self.dataset.rand_getsubsample(pix=pixel, inputSize=self.inputSize, imgBGR=imgBGR)

        # transform data
        img = self.transform(data)

        return img, label, pixel, BGRinfo


class GetCoorData(Dataset):
    def __init__(self, sampling, patchsize, colorData, transform):
        self.patchsize = patchsize
        self.sampling = sampling
        self.colorData = colorData
        self.transform = transform

    def __len__(self):
        return np.size(self.sampling, 0) * np.size(self.sampling, 1)

    def __getitem__(self, item):
        (width_samp, height_samp) = np.shape(self.sampling[:, :, 0])
        x_samp, y_samp = int(item % height_samp), int(item // height_samp)
        (origX, origY) = self.sampling[y_samp, x_samp, :]
        data = self.colorData[int(origY - self.patchsize/2):int(origY + self.patchsize/2),
               int(origX - self.patchsize/2):int(origX + self.patchsize/2), :]
        data_tensor = self.transform(data)
        return data_tensor


# Function for SCORE DATASET
def getRandHyp(gaussRot, gaussTrans):
    np.random.seed()
    trans = np.array([np.random.normal(loc=0, scale=gaussTrans),
                      np.random.normal(loc=0, scale=gaussTrans),
                      np.random.normal(loc=0, scale=gaussTrans)])
    rotAxis = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    rotAxis = rotAxis/np.linalg.norm(rotAxis)
    rotAxis = rotAxis * np.random.normal(loc=0, scale=gaussRot) * math.pi / 180.0
    # Construct rot vector and translation vector
    RotVec = np.zeros(6)
    RotVec[:3] = rotAxis
    RotVec[3:6] = trans
    # Construct a hypothesis
    h = Hypothesis()
    h.RodvecandTrans(RotVec)
    return h


class SCORE_DATASET(Dataset):
    def __init__(self, dataset, objInputSize, rgbInputSize, model, temperature, transform,
                 trainingImages, trainingPatches):
        self.dataset = dataset
        self.objInputSize = objInputSize
        self.rgbInputSize = rgbInputSize
        self.model = model
        self.temperature = temperature
        self.transform = transform
        self.trainingImages = trainingImages
        self.trainingPatches = trainingPatches

    def __len__(self):
        return self.trainingImages * self.trainingPatches

    def __getitem__(self, item):

        # Get random data
        imgBGR = self.dataset.rand_getBGR(item)
        info = self.dataset.rand_getInfo(item)

        # Get parameter
        gp = GlobalProperties()
        camMat = gp.getCamMat()

        # Sampling for reprojection error image
        sampling = cnn.stochasticSubSample(imgBGR, targetsize=self.objInputSize, patchsize=self.rgbInputSize)
        estObj = cnn.getCoordImg(imgBGR, sampling, self.rgbInputSize, self.model)

        # Generate data
        poseGT = Hypothesis()
        poseGT.Info(info)
        np.random.seed()
        driftLevel = np.random.randint(low=0, high=3)
        if not driftLevel:
            poseNoise = poseGT * getRandHyp(2, 2)
        else:
            poseNoise = poseGT * getRandHyp(10, 100)

        data = cnn.getDiffMap(TYPE.our2cv([poseNoise.getRotation(), poseNoise.getTranslation()]),
                              estObj, sampling, camMat)
        data_norm = self.transform(data)

        # Generate GroundTruth Label
        label = -1 * self.temperature * max(poseGT.calcAngularDistance(poseNoise),
                                            np.linalg.norm(poseGT.getTranslation() - poseNoise.getTranslation())/10.0)
        return data_norm, label


class END2END(Dataset):
    def __init__(self, dataset, trainingImages, trainingPatches,
                 objInputSize, rgbInputSize):
        self.dataset = dataset
        self.trainingImages = trainingImages
        self.trainingPatches = trainingPatches
        self.objInputSize = objInputSize
        self.rgbInputSize = rgbInputSize

    def __len__(self):
        return self.trainingImages * self.trainingPatches

    def __getitem__(self, item):

        # Get raw data
        imgBGR = self.dataset.rand_getBGR(item)
        info = self.dataset.rand_getInfo(item)
        poseGT = Hypothesis()
        poseGT.Info(info)

        # Sampling for reprojection error image
        sampling = cnn.stochasticSubSample(imgBGR, targetsize=self.objInputSize, patchsize=self.rgbInputSize)
        estObj = cnn.getCoordImg1(imgBGR, sampling, self.rgbInputSize, self.model)

        return imgBGR, poseGT, sampling, estObj

