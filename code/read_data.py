import cv2
import numpy as np
import os
from numpy.linalg import inv

import TYPE


def readData_depth(dFile):
    img = cv2.imread(dFile, -1)
    return img


def readData_bgr(bgrFile):
    img = cv2.imread(bgrFile)
    return img


def readData_rgbd(bgrFile, dFile):
    img = TYPE.imag_brgd_t()
    img.bgr = readData_bgr(bgrFile)
    img.depth = readData_depth(dFile)
    return img


def readData_info(infoFile):
    info = TYPE.info_t()
    if not os.path.exists(infoFile):
        info.visible = False
        return False
    i = 0
    trans = np.eye(4)
    file_info = open(infoFile)
    lines_info = file_info.readlines()
    for line in lines_info:
        trans[i, :] = np.array(list(map(float, line.split())))
        i += 1
    transfile = './translation.txt'
    file_trans = open(transfile)
    lines_trans = file_trans.readlines()
    trans[:3, 3] -= np.array(list(map(float, lines_trans[0].split())))
    correction = np.eye(4)
    correction[:, 1] = -correction[:, 1]
    correction[:, 2] = -correction[:, 2]
    trans = inv(np.dot(trans, correction))
    info.rotation = trans[:3, :3]
    info.center = trans[:3, 3]
    info.extent = 10*np.ones(3)
    info.visible = True
    info.occlusion = 0
    return info




