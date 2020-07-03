import numpy as np
import cv2
from tkinter import _flatten
import time

import Hypothesis
import util
import properties
import read_data
import TYPE


def pxToEye(x, y, depth):
    eye = np.zeros(3)
    if not depth:
        return eye
    gp = properties.GlobalProperties()
    eye[0] = (x-gp.imageWidth/2.0-gp.xShift)/(gp.focalLength/depth)
    eye[1] = -1*(y-gp.imageHeight/2.0-gp.yShift)/(gp.focalLength/depth)
    eye[2] = -1*depth
    return eye


def onObj(pt):
    return(any(pt))


class Dataset(object):
    def __init__(self, bgrFiles=[], depthFiles=[], infoFiles=[], objID=[],
                 rand_bgeFiles=[], rand_depthFiles=[], rand_infoFiles=[],):
        self.bgrFiles = bgrFiles
        self.depthFiles = depthFiles
        self.infoFiles = infoFiles
        self.objID = objID
        self.rand_bgrFiles = rand_bgeFiles
        self.rand_depthFiles = rand_depthFiles
        self.rand_infoFiles = rand_infoFiles

    def readFileNames(self, basePath):
        bgrPath = basePath + '/rgb_noseg/'
        bgrSuf = '.png'
        dPath = basePath + '/depth_noseg/'
        dSuf = '.png'
        infoPath = basePath + '/poses/'
        infoSuf = '.txt'
        self.bgrFiles = util.getFiles(bgrPath, bgrSuf)
        self.depthFiles = util.getFiles(dPath, dSuf)
        self.infoFiles = util.getFiles(infoPath, infoSuf, True)

    def randFiles(self, trainingImages, trainingPatches):
        random_indx = np.random.randint(low=0, high=self.size(), size=trainingImages)
        random_bgrFiles, random_depthFiles, random_infoFiles = [], [], []
        for i in range(trainingImages):
            random_bgrFiles.append([self.bgrFiles[random_indx[i]]] * trainingPatches)
            random_depthFiles.append([self.depthFiles[random_indx[i]]] * trainingPatches)
            random_infoFiles.append([self.infoFiles[random_indx[i]]] * trainingPatches)
        self.rand_bgrFiles = list(_flatten(random_bgrFiles))
        self.rand_depthFiles = list(_flatten(random_depthFiles))
        self.rand_infoFiles = list(_flatten(random_infoFiles))

    def SetObjID(self, objid):
        self.objID = objid

    def mapDepthTORGB(self, x, y, depth):
        gp = properties.GlobalProperties()
        eye = np.ones(4)
        eye[0] = (x-gp.imageWidth/2.0-gp.rawXShift)/(gp.secondaryFocalLength/float(depth))
        eye[1] = -1*(y-gp.imageHeight/2.0-gp.rawYShift)/(gp.secondaryFocalLength/float(depth))
        eye[2] = -1*depth
        eye = np.dot(gp.sensorTrans, eye)
        pix = np.zeros(2)
        pix[0] = int(eye[0]*gp.focalLength/float(depth)+gp.imageWidth/2.0+gp.xShift+0.5)
        pix[1] = int(-1*eye[1]*gp.focalLength/float(depth)+gp.imageHeight/2.0+gp.yShift+0.5)
        pix_int = pix.astype(int)
        return pix_int

    def getObjID(self):
        return self.objID

    def size(self):
        return len(self.bgrFiles)

    def getFileName(self, i):
        return self.bgrFiles[i]

    def rand_getFileName(self, i):
        return self.rand_bgrFiles[i]

    def getInfo(self, i):
        return read_data.readData_info(self.infoFiles[i])

    def rand_getInfo(self, i):
        return read_data.readData_info(self.rand_infoFiles[i])

    def getBGR(self, i):
        return read_data.readData_bgr(self.bgrFiles[i])

    def rand_getBGR(self, i):
        return read_data.readData_bgr(self.rand_bgrFiles[i])

    def getDepth(self, i):
        img = read_data.readData_depth(self.depthFiles[i])
        gp = properties.GlobalProperties()
        if gp.rawData:
            depthMapped = np.zeros(np.shape(img))
            for x in range(np.shape(img)[1]):
                for y in range(np.shape(img)[0]):
                    depth = img[y, x]
                    if not depth: continue
                    pix = self.mapDepthTORGB(x, y, depth)
                    depthMapped[pix[1], pix[0]] = depth
            img = depthMapped.astype('uint16')
        return img

    def rand_getDepth(self, i):
        img = read_data.readData_depth(self.rand_depthFiles[i])
        gp = properties.GlobalProperties()
        if gp.rawData:
            depthMapped = np.zeros(np.shape(img))
            for x in range(np.shape(img)[1]):
                for y in range(np.shape(img)[0]):
                    depth = img[y][x]
                    if depth == 0: continue
                    pix = self.mapDepthTORGB(x, y, depth)
                    depthMapped[pix[1], pix[0]] = depth
            img = depthMapped.astype('uint16')
        return img

    def getBGRD(self, i):
        img = TYPE.imag_brgd_t()
        img.bgr = self.getBGR(i)
        img.depth = self.getDepth(i)

    def rand_getBGRD(self, i):
        img = TYPE.imag_brgd_t()
        img.bgr = self.rand_getBGR(i)
        img.depth = self.rand_getDepth(i)

    def getObj(self, i):
        time_start = time.time()
        depthData = self.getDepth(i)
        poseData = self.getInfo(i)
        h = Hypothesis.Hypothesis()
        h.Info(poseData)
        img_cam = np.zeros([np.shape(depthData)[0], np.shape(depthData)[1], 3])
        img_obj = np.zeros([np.shape(depthData)[0], np.shape(depthData)[1], 3])
        for x in range(np.shape(depthData)[1]):
            for y in range(np.shape(depthData)[0]):
                if not depthData[y, x]:
                    img_cam[y, x, :] = np.zeros(3)
                    continue
                img_cam[y, x, :] = pxToEye(x, y, depthData[y, x])
                img_obj[y, x, :] = h.invTransform(img_cam[y, x, :])
        time_end = time.time()
        print('time in loading obj:', time_end - time_start)
        return img_obj

    def rand_getObj(self, i):
        depthData = self.rand_getDepth(i)
        poseData = self.rand_getInfo(i)
        h = Hypothesis.Hypothesis()
        h.Info(poseData)
        img_cam = np.zeros([np.shape(depthData)[0], np.shape(depthData)[1], 3])
        img_obj = np.zeros([np.shape(depthData)[0], np.shape(depthData)[1], 3])
        for x in range(np.shape(depthData)[1]):
            for y in range(np.shape(depthData)[0]):
                if not depthData[y, x]:
                    img_cam[y, x, :] = np.zeros(3)
                    continue
                img_cam[y, x, :] = pxToEye(x, y, depthData[y, x])
                img_obj[y, x, :] = h.invTransform(img_cam[y, x, :])
        return img_obj

    def getEye(self, i):
        imgDepth = self.getDepth(i)
        img = np.zeros([np.shape(imgDepth)[0], np.shape(imgDepth)[1], 3])
        for x in range(np.shape(imgDepth)[1]):
            for y in range(np.shape(imgDepth)[0]):
                img[y, x, :] = pxToEye(x, y, imgDepth[y, x])
        return img

    def rand_getEye(self, i):
        imgDepth = self.rand_getDepth(i)
        img = np.zeros([np.shape(imgDepth)[0], np.shape(imgDepth)[1], 3])
        for x in range(np.shape(imgDepth)[1]):
            for y in range(np.shape(imgDepth)[0]):
                img[y, x, :] = pxToEye(x, y, imgDepth[y, x])
        return img

    def rand_subsample(self, imgBGR, imgObj, inputSize):
        np.random.seed()
        width = np.size(imgBGR, 1)
        height = np.size(imgBGR, 0)
        x = np.random.randint(inputSize / 2, width - inputSize / 2)
        y = np.random.randint(inputSize / 2, height - inputSize / 2)
        data = imgBGR[int(y - inputSize / 2): int(y + inputSize / 2),
               int(x - inputSize / 2): int(x + inputSize / 2), :]
        label = imgObj[y, x, :]/1000.0
        return data, label

    # used for corner points selection
    """def rand_getRGBpix(self, i, inputSize):

        # Read Raw data
        img = read_data.readData_depth(self.rand_depthFiles[i])
        (height, width) = np.shape(img)
        img_rgb = self.rand_getBGR(i)
        keymap, _ = cnn.CornerDetector(img_rgb)

        # Collect points in the range 
        keymap_range_stacked = []
        for j in range(int(inputSize/2), int(height - inputSize/2)):
            for i in range(int(inputSize/2), int(width - inputSize/2)):
                if(keymap[j, i]):
                    keymap_range_stacked.append([i, j]) #(x, y)

        # For random selecting pixel
        random.seed()
        keypoint = random.sample(keymap_range_stacked, 1)[0]
        x = keypoint[0]
        y = keypoint[1]

        depth = img[y, x]
        if not depth:
            return np.array([x, y]), depth
        else:
            pix = self.mapDepthTORGB(x, y, depth)
            if pix[0] < inputSize / 2 or pix[0] > width - inputSize / 2 \
                or pix[1] < inputSize / 2 or pix[1] > height - inputSize / 2:
                return np.array([x, y]), 0
            else:
                return pix, depth"""
    
    # used for no corner points selection
    def rand_getRGBpix(self, i, inputSize):

        # Read Raw data
        img = read_data.readData_depth(self.rand_depthFiles[i])
        (height, width) = np.shape(img)

        # For random selecting pixel
        np.random.seed()
        x = np.random.randint(inputSize / 2, width - inputSize / 2)
        y = np.random.randint(inputSize / 2, height - inputSize / 2)

        depth = img[y, x]
        if not depth:
            return np.array([x, y]), depth
        else:
            pix = self.mapDepthTORGB(x, y, depth)
            if pix[0] < inputSize / 2 or pix[0] > width - inputSize / 2 \
                or pix[1] < inputSize / 2 or pix[1] > height - inputSize / 2:
                return np.array([x, y]), 0
            else:
                return pix, depth

    def rand_getcoord(self, pix, depth, i):
        x, y = pix[0], pix[1]
        poseData = self.rand_getInfo(i)
        h = Hypothesis.Hypothesis()
        h.Info(poseData)
        if not depth:
            return np.zeros(3)
        else:
            img_cam = pxToEye(x, y, depth)
            img_obj = h.invTransform(img_cam)
            return img_obj/1000.0

    def rand_getsubsample(self, imgBGR, inputSize, pix):
        x, y = pix[0], pix[1]
        data = imgBGR[int(y - inputSize / 2): int(y + inputSize / 2),
               int(x - inputSize / 2): int(x + inputSize / 2), :]
        return data



# time_start = time.time()
# d = Dataset()
# path = '/home/open/eth/2020spring/3DV/Project/C++/DSAC/7scenes/7scenes_chess/test/scene'
# d.readFileNames(path)
# # print(d.mapDepthTORGB(631, 35, 2304))
# img = read_data.readData_depth(d.depthFiles[2])
# getObj(2, d)
# time_end = time.time()
# print('cost time:', time_end - time_start)
# print(c.size)
# b = d.getObj(2)
# c = torch.from_numpy(b[30, 30, :]).unsqueeze(0)
# print(b[30, 30, :])
# print(c.shape)
# pathd = path+'/depth_noseg/frame-000002.depth.png'
# img = read_data.readData_depth(pathd)
# print(img[20][200])
# b = d.getDepth(2)
#
# print('b', b[39][205])
# print(b.dtype)
# print(b.astype('uint16').dtype)
# cv2.imwrite('/home/yzy/Pictures/xx.png', b.astype('uint16'))
# cv2.imshow('s', b)
# cv2.waitKey(0)

# print(d.getDepth(2)[480][20])
# print(len(d.bgrFiles))

# print('map', d.mapDepthTORGB(200, 20, 2274))
# print('raw', img[205][39])

#
# c = cv2.imread('/home/yzy/Pictures/xx.png', -1)
# print(c[39][205])