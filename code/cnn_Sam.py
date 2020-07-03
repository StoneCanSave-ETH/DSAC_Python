import numpy as np
import random
import cv2
from torchvision import transforms
import torch
import time
import copy

import properties
import TYPE
import Hypothesis
import Model_obj
import Customized_Datasets
import Model_score
import maxloss as ml


CNN_OBJ_MAXINPUT = 100.0
BATCHSIZE = 64
CNN_RGB_PATCHSIZE = 42
CNN_OBJ_PATCHSIZE = 40
EPS = 1e-9
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Transform_OBJ():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform


def Transform_SCORE():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform


def containsNaNs(m):
    return np.any(np.isnan(m))


def entropy(dist):
    temp = -1*dist*np.log2(dist)
    return sum(temp)


def draw(probs, randomdraw):
    cumProb = []
    cumProbIdx = []
    probsum = 0
    for i in range(len(probs)):
        if probs[i] < EPS: continue
        probsum += probs[i]
        cumProb.append(probsum)
        cumProbIdx.append(i)
    cumProb, cumProbIdx = np.array(cumProb), np.array(cumProbIdx)
    gp = properties.GlobalProperties()
    if randomdraw:
        rand = random.uniform(0, probsum)
        idx = np.argwhere(cumProb > rand)[0]
        return int(cumProbIdx[idx])
    else:
        return np.argmax(probs)


def expectedMaxLoss(gt, hyps, probs):
    loss = 0
    losses = np.zeros(len(hyps))
    for i in range(len(hyps)):
        jpHyp = TYPE.cv2our(hyps[i])
        hyp = Hypothesis.Hypothesis()
        hyp.RotandTrans(jpHyp[0], jpHyp[1])
        losses[i] = ml.maxLoss(gt, hyp)
        loss = loss + probs[i] * losses[i]
    return loss, losses


def safeSolvePnP(objPts, imgPts, camMat, disCoeffs, methodFlag, rot, trans):
    """
    we should transfer the form of objPts and imgPts from list to np.array before using it
    :param objPts:
    :param imgPts:
    :param camMat:
    :param disCoeffs:
    :param methodFlag:
    :return:
    """
    """if not rot.all():
        retval, _, _, _ = cv2.solvePnPRansac(objPts, imgPts, camMat, disCoeffs, None, None, 0, 500, 10, 0.99, None, methodFlag)
        if not retval:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
        else:
            _, rvec, tvec, _ = cv2.solvePnPRansac(objPts, imgPts, camMat, disCoeffs, None, None, 0, 500, 10, 0.99, None, methodFlag)
        return [rvec, tvec]
    else:
        retval, _, _, _ = cv2.solvePnPRansac(objPts, imgPts, camMat, disCoeffs, rot, trans, 0, 500, 10, 0.99, None, methodFlag)
        if not retval:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
        else:
            _, rvec, tvec, _ = cv2.solvePnPRansac(objPts, imgPts, camMat, disCoeffs, rot, trans, 0, 500, 10, 0.99, None, methodFlag)
        return [rvec, tvec]"""
    if not rot.all():
        retval, _, _ = cv2.solvePnP(objPts, imgPts, camMat, disCoeffs, None, None, 0, methodFlag)
        if not retval:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
        else:
            _, rvec, tvec = cv2.solvePnP(objPts, imgPts, camMat, disCoeffs, None, None, 0, methodFlag)
        return [rvec, tvec]
    else:
        retval, _, _ = cv2.solvePnP(objPts, imgPts, camMat, disCoeffs, rot, trans, 0, methodFlag)
        if not retval:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
        else:
            _, rvec, tvec = cv2.solvePnP(objPts, imgPts, camMat, disCoeffs, rot, trans, 0, methodFlag)
        return [rvec, tvec]


def dPNP(imgPts, objPts, eps = 0.1):
    if len(imgPts) == 4:
        pnpMethod = cv2.SOLVEPNP_P3P
    else:
        pnpMethod = cv2.SOLVEPNP_ITERATIVE
    gp = properties.GlobalProperties()
    camMat = gp.getCamMat()
    imgPts = np.array(imgPts, np.int64)
    jacobean = np.zeros([6, len(objPts)*3])
    for i in range(len(objPts)):
        for j in range(3):
            # Forward step
            if j == 0: objPts[i][0] += eps
            elif j == 1: objPts[i][1] += eps
            elif j == 2: objPts[i][2] += eps
            objPts = np.array(objPts, np.float64)
            _, rot_f, tvec_f = safeSolvePnP(objPts, imgPts, camMat, None, pnpMethod)
            Trans_f = TYPE.cv2our([rot_f, tvec_f])
            h_f = Hypothesis.Hypothesis()
            h_f.RotandTrans(Trans_f[0], Trans_f[1])
            fstep = h_f.getRodVecAndTrans()

            # Backward step
            if j == 0: objPts[i][0] -= 2*eps
            elif j == 1: objPts[i][1] -= 2*eps
            elif j == 2: objPts[i][2] -= 2*eps
            objPts = np.array(objPts, np.float64)
            _, rot_b, tvec_b = safeSolvePnP(objPts, imgPts, camMat, None, pnpMethod)
            Trans_b = TYPE.cv2our([rot_b, tvec_b])
            h_b = Hypothesis.Hypothesis()
            h_b.RotandTrans(Trans_b[0], Trans_b[1])
            bstep = h_b.getRodVecAndTrans()

            # Back to normal state
            if j == 0: objPts[i][0] += eps
            elif j == 1: objPts[i][1] += eps
            elif j == 2: objPts[i][2] += eps

            # Gradient calculation
            for k in range(len(fstep)):
                jacobean[k][3*i+j] = (fstep[k] - bstep[k])/(2*eps)
            if containsNaNs(jacobean[:, 3*i+j]):
                return np.zeros([6, 3*objPts])
    return jacobean


def getAvg(mat):
    return np.average(np.abs(mat))


def getMax(mat):
    return np.max(np.abs(mat))


def getMed(mat):
    return np.median(np.abs(mat))


def getCoordImg(colorData, sampling, patchsize, model):
    transform = Transform_OBJ()
    DATA = Customized_Datasets.GetCoorData(sampling=sampling, patchsize=patchsize,
                                              colorData=colorData, transform=transform)
    data_loader = torch.utils.data.DataLoader(DATA, batch_size=BATCHSIZE, shuffle=False, num_workers=8)
    for idx, (BatchData) in enumerate(data_loader):
        BatchData = BatchData.to(DEVICE)
        pred = Model_obj.forward(model, BatchData, DEVICE)
        if not idx:
            patch = pred
        else:
            patch = np.vstack((patch, pred))
    (height, width) = np.shape(sampling[:, :, 0])
    modeImg = patch.reshape([height, width, 3]) * 1000.0
    return modeImg

def CornerDetector(img):
    cornermap = np.zeros((img.shape[0],img.shape[1]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    thres = 0.01*dst.max()
    cornermap[dst>thres] = 1
    cornerVector = []
    for j in range(0, img.shape[0]):
        for i in range(0, img.shape[1]):
            if cornermap[j, i]:
                cornerVector.append(np.array([j, i]))
    print("the length of cornervector is", len(cornerVector))
    return cornermap, cornerVector

def stochasticSubSamplewithoutC(inputMap, targetsize, patchsize):
    np.random.seed()
    width = np.shape(inputMap)[1]
    height = np.shape(inputMap)[0]
    sampling = np.zeros([targetsize, targetsize, 2])
    xStride = (width - patchsize)/targetsize
    yStride = (height - patchsize)/targetsize
    xrange = np.zeros(targetsize + 1)
    yrange = np.zeros(targetsize + 1)
    for i in range(targetsize + 1):
        xrange[i] = patchsize/2 + i * xStride
        yrange[i] = patchsize/2 + i * yStride
    for x in range(targetsize):
        for y in range(targetsize):
            sampling[y, x, 0] = int(xrange[y] + (xrange[y + 1] - xrange[y]) * np.random.random())
            sampling[y, x, 1] = int(yrange[x] + (yrange[x + 1] - yrange[x]) * np.random.random())
    return sampling

def stochasticSubSample2(img, targetsize, patchsize):
    inputMap, _ = CornerDetector(img)
    np.random.seed()
    width = np.shape(inputMap)[1]
    height = np.shape(inputMap)[0]
    sampling = np.zeros([targetsize, targetsize, 2])
    key_collection = []
    # create a classification container for keypoints
    for m in range(0, targetsize*targetsize):
        key_collection.append([])
    # to define range of sampling
    xStride = (width - patchsize)/targetsize
    yStride = (height - patchsize)/targetsize
    xrange = np.zeros(targetsize + 1)
    yrange = np.zeros(targetsize + 1)
    for i in range(targetsize + 1):
        xrange[i] = patchsize/2 + i * xStride
        yrange[i] = patchsize/2 + i * yStride
    xrange_f = int(patchsize/2)
    yrange_f = int(patchsize/2)
    xrange_n = int(width - patchsize/2)
    yrange_n = int(height - patchsize/2)

    # collect and classify keypoints
    for j in range(yrange_f, yrange_n):
        for i in range(xrange_f, xrange_n):
            if inputMap[j, i]:
                block_locationX = int((i - patchsize//2) // xStride)
                block_locationY = int((j - patchsize//2) // yStride)
                key_collection[block_locationY*targetsize+block_locationX].append(np.array([i, j]))
    # if the cell contains keypoints, choose one; else sample randomly
    for x in range(targetsize):
        for y in range(targetsize):
            if not len(key_collection[y*targetsize+x]):
                sampling[y, x, 0] = int(xrange[y] + (xrange[y + 1] - xrange[y]) * np.random.random())
                sampling[y, x, 1] = int(yrange[x] + (yrange[x + 1] - yrange[x]) * np.random.random())
            else:
                # print("the len of keypoint is:", len(key_collection[y*targetsize+x]), "which is located in",[y, x])
                random.shuffle(key_collection[y*targetsize+x])
                sampling[y, x, 0] = key_collection[y*targetsize+x][0][0]
                sampling[y, x, 1] = key_collection[y*targetsize+x][0][1]
    return sampling

def stochasticSubSample(img, targetsize, patchsize):
    np.random.seed()
    inputMap, _ = CornerDetector(img)
    sampling = np.zeros([targetsize, targetsize, 2])
    width = np.shape(inputMap)[1]
    height = np.shape(inputMap)[0]
    xrange_f = int(patchsize/2)
    yrange_f = int(patchsize/2)
    xrange_n = int(width - patchsize/2)
    yrange_n = int(height - patchsize/2)
    keycorner = []
    for j in range(yrange_f, yrange_n):
        for i in range(xrange_f, xrange_n):
            if inputMap[j, i]:
                keycorner.append(np.array([i, j]))
    if len(keycorner) < 1600:
        sampling = stochasticSubSample2(img, targetsize, patchsize)
    if len(keycorner) >= 1600:
        random.shuffle(keycorner)
        for x in range(targetsize):
            for y in range(targetsize):
                sampling[y, x, 0] = keycorner[y*targetsize+x][0]
                sampling[y, x, 1] = keycorner[y*targetsize+x][1]
    return sampling


def getDiffMap(hyp, objectCoordinates, sampling, camMat):
    width = np.size(sampling[:, :, 0], 1)
    height = np.size(sampling[:, :, 0], 0)
    diffMap = np.zeros([height, width, 1])
    CamMat = camMat.astype(np.float)
    for i in range(width * height):
        x = i % width
        y = i // width
        projections, _ = cv2.projectPoints(objectPoints=objectCoordinates[y, x, :],
                                            rvec=hyp[0], tvec=hyp[1], cameraMatrix=CamMat, distCoeffs=None)

        curPt = projections - sampling[y, x, :]
        diffMap[y, x, :] = min(np.linalg.norm(curPt), CNN_OBJ_MAXINPUT)
    return diffMap


def project(pt, obj, rot, t, camMat):
    f = camMat[0][0]
    ppx = camMat[0][2]
    ppy = camMat[1][2]
    objMat = np.dot(rot, obj)+t
    # Since we calculate the groudtruth object coordinate by using OPENCV coordinate
    # as can be seen in dataset.py 'getobj'
    px = -1 * f * objMat[0]/objMat[2] + ppx
    py = f * objMat[1]/objMat[2] + ppy
    pxy = np.array([px, py])
    return min(np.linalg.norm(pxy-pt), CNN_OBJ_MAXINPUT)


def dProjectObj(pt, obj, rot, t, camMat):
    f = camMat[0][0]
    ppx = camMat[0][2]
    ppy = camMat[1][2]
    objMat = np.dot(rot, obj) + t

    # Prevent division by zero
    if np.abs(objMat)[2] < np.finfo(float).eps: return np.zeros(3)
    px = -1 * f * objMat[0]/objMat[2] + ppx
    py = f * objMat[1]/objMat[2] + ppy
    pxy = np.array([px, py])

    # Calculate error
    err = np.linalg.norm(pxy-pt)
    if err > CNN_OBJ_MAXINPUT: return np.zeros(3)
    err += np.finfo(float).eps

    # derivative in x direction
    pxdx = -1 * f * rot[0][0]/objMat[2] + f * objMat[0] * rot[2][0] / (objMat[2]**2)
    pydx = f * rot[1][0]/objMat[2] - f * objMat[1] * rot[2][0] / (objMat[2]**2)
    dx = (pxy - pt)[0] * pxdx / err + (pxy - pt)[1] *pydx / err

    # derivative in y direction
    pxdy = -1 * f * rot[0][1] / objMat[2] + f * objMat[0] * rot[2][1] / (objMat[2] ** 2)
    pydy = f * rot[1][1] / objMat[2] - f * objMat[1] * rot[2][1] / (objMat[2] ** 2)
    dy = (pxy - pt)[0] * pxdy / err + (pxy - pt)[1] * pydy / err

    # derivative in z direction
    pxdz = -1 * f * rot[0][2] / objMat[2] + f * objMat[0] * rot[2][2] / (objMat[2] ** 2)
    pydz = f * rot[1][2] / objMat[2] - f * objMat[1] * rot[2][2] / (objMat[2] ** 2)
    dz = (pxy - pt)[0] * pxdz / err + (pxy - pt)[1] * pydz / err

    return np.array([dx, dy, dz])


def softMax(scores):

    maxScore = 0
    for i in range(0, len(scores)):
        if i==0 or scores[i]>maxScore:
            maxScore = scores[i]

    sf = np.zeros(len(scores))
    sum = 0.0

    for i in range(0, len(scores)):
        sf[i] = np.exp(scores[i]-maxScore)
        sum += sf[i]

    sf /= sum

    return sf


def dProjectdHyp(pt, obj, rot, t, camMat):
    f = camMat[0, 0]
    ppx = camMat[0, 2]
    ppy = camMat[1, 2]

    eyeMat = np.dot(rot, obj) + t

    if abs(eyeMat[2]) < EPS:
        return np.zeros(6)

    px = -1*f*eyeMat[0]/eyeMat[2]+ppx
    py = f*eyeMat[1]/eyeMat[2]+ppy

    err = np.sqrt((pt[0]-px)**2+(pt[1]-py)**2)

    if err > CNN_OBJ_MAXINPUT:
        return np.zeros(6)

    err += EPS

    dNdP = np.zeros((1, 2))
    dNdP[0, 0] = -1/err*(pt[0]-px)
    dNdP[0, 1] = -1/err*(pt[1]-py)

    dPdR = np.zeros((2, 9))
    dPdR[0, 0:3] = -f * np.transpose(obj) / eyeMat[2]
    dPdR[1, 3:6] = f * np.transpose(obj) / eyeMat[2]
    dPdR[0, 6:9] = f * eyeMat[0] / eyeMat[2] / eyeMat[2] * np.transpose(obj)
    dPdR[1, 6:9] = -f * eyeMat[1] / eyeMat[2] / eyeMat[2] * np.transpose(obj)

    dRdH1 = np.zeros((3, 9))
    rod = np.zeros(3)
    cv2.Rodrigues(rot, rod)
    cv2.Rodrigues(rod, rot, dRdH1)
    dRdH = np.transpose(dRdH1)

    dNdH = np.dot(np.dot(dNdP, dPdR), dRdH).reshape([-1])

    dPdT = np.zeros((2, 3))
    dPdT[0, 0] = -1*f/eyeMat[2]
    dPdT[1, 1] = f/eyeMat[2]
    dPdT[0, 2] = f*eyeMat[0]/eyeMat[2]/eyeMat[2]
    dPdT[1, 2] = -1*f*eyeMat[1]/eyeMat[2]/eyeMat[2]

    dNdT = np.dot(dNdP, dPdT).reshape([-1])

    jacobean = np.zeros(6)
    jacobean[:3] = dNdH
    jacobean[3:] = dNdT

    return jacobean


def dScore(estObj,
           sampling,
           points,
           model,    # jacobeans is output param, so not in here
           scoreOutputGradients):

    gp = properties.GlobalProperties()
    camMat = gp.getCamMat()

    hypCount = points.shape[0]
    imgPts = []
    objPts = []
    hyps = []
    diffMaps = []
    dscore_dDiffmaps = []
    dDiffMaps = []

    for h in range(hypCount):
        for i in range(4):
            x = points[h, i, 0]
            y = points[h, i, 1]

            imgPts[h].append(sampling[y, x, :])
            objPts[h].append(estObj[y, x, :])

        # 没写solvepnp输入输出格式这里按照后面一致的格式
        cvHyp = safeSolvePnP(objPts=objPts[h], imgPts=imgPts[h], camMat=camMat, disCoeffs=None,
                     methodFlag=cv2.SOLVEPNP_P3P, rot=np.zeros([3]), trans=np.zeros([3]))

        hyps[h] = TYPE.cv2our(cvHyp)
        diffMaps[h] = getDiffMap(cvHyp, estObj, sampling, camMat)
        trasnform = Transform_SCORE()
        diffMaps[h] = trasnform(diffMaps)
        dscore_dDiffmaps.append(Model_score.backward(model=model, data=diffMaps[h], device=DEVICE))
        dDiffMaps.append(dscore_dDiffmaps[h] * scoreOutputGradients)

    jacobeans = []

    # jacobeans = np.array()
    for h in range(hypCount):
        jacobean = np.zeros(estObj.shape[0] * estObj.shape[1] * 3)

        supportPointGradients = np.zeros(1, 12)
        dHdO = dPNP(imgPts[h], objPts[h])  # 6*12 dimension

        for x in range(CNN_OBJ_PATCHSIZE):
            for y in range(CNN_OBJ_PATCHSIZE):
                pt = sampling[y, x, :]
                obj = estObj[y, x, :]
                dPdO = dProjectObj(pt=pt, obj=obj, rot=hyps[h][0], t=hyps[h][1], camMat=camMat)
                dPdO = dPdO * dDiffMaps[h][y, x]
                jacobean[1, x * CNN_OBJ_PATCHSIZE * 3 + y * 3 : x * CNN_OBJ_PATCHSIZE * 3 + y * 3 + 3] = copy.copy(dPdO)

                dPdH = dProjectdHyp(sampling[y, x, :], estObj[y, x, :], hyps[h][0], hyps[h][1], camMat)
                supportPointGradients += dDiffMaps[h][y, x] * dPdH * dHdO

        for i in range(4):
            x = points[h, i, 0]
            y = points[h, i, 1]

            jacobean[1, x * CNN_OBJ_PATCHSIZE * 3 + y * 3: x * CNN_OBJ_PATCHSIZE * 3 + y * 3 + 3] += \
            supportPointGradients[1, i * 3 : i * 3 + 3]

        jacobeans.append(jacobean)

    return jacobeans


def dSMScore(estObj,
             sampling,
             points,
             losses,
             sfScores,
             model):

    temp = losses * sfScores
    scoreOutputGradients = temp - sfScores * (np.sum(temp))

    jacobeans = dScore(estObj, sampling, points, model, scoreOutputGradients)

    for i in range(0, len(jacobeans)):
        reformat = np.zeros([CNN_OBJ_PATCHSIZE * CNN_OBJ_PATCHSIZE, 3])

        for x in range(CNN_OBJ_PATCHSIZE):
            for y in range(CNN_OBJ_PATCHSIZE):
                patchGrad = jacobeans[i][1,
                            x * CNN_OBJ_PATCHSIZE * 3 + y * 3: x * CNN_OBJ_PATCHSIZE * 3 + y * 3 + 3]

                reformat[y * CNN_OBJ_PATCHSIZE + x, :] = copy.deepcopy(patchGrad)

        jacobeans[i] = reformat

    return jacobeans


def refine(inlierCount,
           refSteps,
           inlierThreshold2D,
           pixelIdxs,
           estObj,
           sampling,
           camMat,
           imgPts,
           objPts):

    hyp = safeSolvePnP(objPts=objPts, imgPts=imgPts, camMat=camMat, disCoeffs=None,
                 methodFlag=cv2.SOLVEPNP_P3P, rot=np.zeros([3]), trans=np.zeros([3]))

    diffMap = getDiffMap(hyp, estObj, sampling, camMat)

    for rStep in range(refSteps):
        localImgPts = []
        localObjPts = []
        for idx in range(pixelIdxs[rStep].shape):

            x = pixelIdxs[rStep][idx] % CNN_OBJ_PATCHSIZE
            y = pixelIdxs[rStep][idx] // CNN_OBJ_PATCHSIZE

            if diffMap[y, x] < inlierThreshold2D:
                localImgPts.append(sampling[y, x, :])
                localObjPts.append(estObj[y, x, :])

            if len(localImgPts) >= inlierCount:
                break

        if len(localImgPts) < 50:
            break

        if len(localObjPts) > 4:
            methodflag = cv2.SOLVEPNP_ITERATIVE
        else:
            methodflag = cv2.SOLVEPNP_P3P
        localObjPts, localImgPts = np.array([localObjPts]).reshape([-1, 3, 1]), np.array([localImgPts]).reshape(
            [-1, 2, 1])
        hypUpdate = copy.deepcopy(hyp)

        if not (safeSolvePnP(objPts=localObjPts, imgPts=localImgPts, camMat=camMat, disCoeffs=None,
                             methodFlag=methodflag, rot=hypUpdate[0], trans=hypUpdate[1])[0]).all():
            break

        hyp = copy.deepcopy(hypUpdate)

        # recalculate pose errors
        diffMap = getDiffMap(hyp, estObj, sampling, camMat)

    jpHyp = TYPE.cv2our(hyp)
    hy = Hypothesis.Hypothesis()
    hy.RotandTrans(jpHyp[0], jpHyp[1])
    return hy.getRodVecAndTrans()


def dRefine(inlierCount,
            refSteps,
            subSampleFactor,
            inlierThreshold2D,
            pixelIdxs,
            estObj,
            sampling,
            camMat,
            imgPts,
            objPts,
            sampledPoints,
            inlierMap,
            eps=2):

    localEstObj = copy.deepcopy(estObj)
    [num_x, num_y, num_z] = localEstObj.shape
    jacobean = np.zeros(6, num_x * num_y * num_z)

    for pt in range(sampledPoints.shape[0] - 1):
        for c in range(3):
            localEstObj[sampledPoints[pt, 1], sampledPoints[pt, 0], c] += eps
            if not c:
                objPts[pt, 0] += eps
            elif c == 1:
                objPts[pt, 1] += eps
            else:
                objPts[pt, 2] += eps

            # forward step
            fStep = refine(inlierCount,
                           refSteps,
                           inlierThreshold2D,
                           pixelIdxs,
                           localEstObj,
                           sampling,
                           camMat,
                           imgPts,
                           objPts)

            localEstObj[sampledPoints[pt, 1], sampledPoints[pt, 0], c] -= 2 * eps
            if not c:
                objPts[pt, 0] -= 2 * eps
            elif c == 1:
                objPts[pt, 1] -= 2 * eps
            else:
                objPts[pt, 2] -= 2 * eps

            # backward step
            bStep = refine(inlierCount,
                           refSteps,
                           inlierThreshold2D,
                           pixelIdxs,
                           localEstObj,
                           sampling,
                           camMat,
                           imgPts,
                           objPts)

            localEstObj[sampledPoints[pt, 1], sampledPoints[pt, 0], c] += eps
            if not c:
                objPts[pt, 0] += eps
            elif c == 1:
                objPts[pt, 1] += eps
            else:
                objPts[pt, 2] += eps

            for k in range(len(fStep)):
                jacobean[k, sampledPoints[pt, 1] * CNN_OBJ_PATCHSIZE * 3 + sampledPoints[pt, 0] * 3 + c] \
                    = (fStep[k] - bStep[k]) / (2 * eps)

        inCount = 0
        skip = 1 / subSampleFactor

        for x in range(0, inlierMap.shape[1]):
            for y in range(0, inlierMap.shape[0]):
                if not inlierMap[y, x]:
                    continue
                inCount += 1

                if inCount % skip != 0:
                    continue
                for c in range(3):
                    # forward step
                    localEstObj[y, x, c] += eps

                    fStep = refine(inlierCount,
                                   refSteps,
                                   inlierThreshold2D,
                                   pixelIdxs,
                                   localEstObj,
                                   sampling,
                                   camMat,
                                   imgPts,
                                   objPts)

                    # backward step
                    localEstObj[y, x, c] -= 2 * eps
                    bStep = refine(inlierCount,
                                   refSteps,
                                   inlierThreshold2D,
                                   pixelIdxs,
                                   localEstObj,
                                   sampling,
                                   camMat,
                                   imgPts,
                                   objPts)

                    localEstObj[y, x, c] += eps

                    for k in range(len(fStep)):
                        jacobean[k, y * CNN_OBJ_PATCHSIZE * 3 + x * 3 + c] = (fStep[k] - bStep[k]) / (2 * eps) * skip

    return jacobean


def processImage(imgBGR,
                 poseGT,
                 model_obj,
                 model_score,
                 objHyps,
                 ptCount,
                 camMat,
                 inlierThreshold2D,
                 inlierCount,
                 refSteps,
                 hyps,
                 refHyps,
                 imgPts,
                 objPts,
                 imgIdx,
                 sfScores,
                 estObj,
                 sampling,
                 sampledPoints,
                 inlierMaps,
                 pixelIdxs):
    time_start = time.time()
    keymap, _ = CornerDetector(imgBGR)
    sampling = stochasticSubSample(keymap, CNN_OBJ_PATCHSIZE, CNN_RGB_PATCHSIZE)
    # sampling = stochasticSubSample(imgBGR, CNN_OBJ_PATCHSIZE, CNN_RGB_PATCHSIZE)
    # patches = [] # here define the patch a [y,x,3], and patches is a list of them, or
    estObj = getCoordImg(imgBGR, sampling, CNN_RGB_PATCHSIZE, model_obj)

    # hyps = []

    # imgPts = [] #这个是一系列（第一维）的点集，每个点集（第二维）存了这个点在原始rgb照片中的位置（第3，4维）
    # objPts = [] #与上一个类似，最后每个点存了对应物体上点的3d坐标（第3，4，5维）
    # sampledPoints = [] #与上一个类似，每个点在subsampled图片中的位置
    # imgIdx = [] #配合着上一个使用，表示的是每一个subsampled图片中的点的一维编号（size*y+x)（第二维）
    # 他们的第一维的大小都是和假设的数量是一样的

    for h in range(0, objHyps):
        np.random.seed()
        while True:
            projections = []
            alreadyChosen = np.zeros((estObj.shape[0], estObj.shape[1]))

            imgPts.append([])
            objPts.append([])
            imgIdx.append([])  # the usage of .clear()
            sampledPoints.append([])
            hyps.append([])

            j = 0

            while j < ptCount:

                x = np.random.randint(0, estObj.shape[1])
                y = np.random.randint(0, estObj.shape[0])
                j = j + 1
                if (alreadyChosen[y, x] > 0):
                    j = j - 1
                    continue

                alreadyChosen[y, x] = 1

                imgPts[h].append(sampling[y, x, :])
                objPts[h].append(estObj[y, x, :])
                imgIdx[h].append(y * CNN_OBJ_PATCHSIZE + x)
                sampledPoints[h].append(np.array([x, y]))

            objPts_3D = np.array(objPts[h])
            imgPts_3D = np.array(imgPts[h])
            hyps[h].append(safeSolvePnP(objPts=objPts_3D, imgPts=imgPts_3D, camMat=camMat, disCoeffs=None,
                                        methodFlag=cv2.SOLVEPNP_P3P, rot=np.zeros([3]), trans=np.zeros([3]))[0])
            hyps[h].append(safeSolvePnP(objPts=objPts_3D, imgPts=imgPts_3D, camMat=camMat, disCoeffs=None,
                                        methodFlag=cv2.SOLVEPNP_P3P, rot=np.zeros([3]), trans=np.zeros([3]))[1])

            if not (safeSolvePnP(objPts=objPts_3D, imgPts=imgPts_3D, camMat=camMat, disCoeffs=None,
                                  methodFlag=cv2.SOLVEPNP_P3P, rot=np.zeros([3]), trans=np.zeros([3]))[0]).all():
                del imgPts[h]
                del objPts[h]
                del imgIdx[h]
                del sampledPoints[h]
                del hyps[h]
                continue
            #else: break


        # to project a 3d point into the image(do not know whether there is a function in cv2 can fulfill the task)
            for points in range(0, len(objPts[h])):
                objMat = objPts[h][points]
                projection, _ = cv2.projectPoints(objectPoints=objMat, rvec=hyps[h][0], tvec=hyps[h][1],
                                                  cameraMatrix=camMat, distCoeffs=None)
                projections.append(projection)

            foundOutlier = False

            for j in range(0, len(imgPts[h])):
                # print("diff is:",np.linalg.norm(imgPts[h][j] - projections[j]))
                if (np.linalg.norm(imgPts[h][j] - projections[j]) < inlierThreshold2D):
                    continue
                foundOutlier = True
                break

            if (foundOutlier):
                del imgPts[h]
                del objPts[h]
                del imgIdx[h]
                del sampledPoints[h]
                del hyps[h]
                continue
            else:
                break

        transform = Transform_SCORE()
        if not h:
            FullDiffMap = getDiffMap(hyps[h], estObj, sampling, camMat)
            FullDiffMap = transform(FullDiffMap).unsqueeze(0)
        else:
            diffmap = getDiffMap(hyps[h], estObj, sampling, camMat)
            diffmap = transform(diffmap).unsqueeze(0)
            FullDiffMap = torch.cat((FullDiffMap, diffmap), 0)
    scores = Model_score.forward(model_score, FullDiffMap, DEVICE)
    sfScores = softMax(scores)
    sfEntropy = entropy(sfScores)
    hypIdx = draw(sfScores)

    # Refine
    (height, width) = np.shape(sampling[:, :, 0])
    inlierMaps = []

    for h in range(len(hyps)):
        refHyps.append(hyps[h])
        localDiffMap = FullDiffMap[h, 0, :, :].numpy()
        inlierMaps.append(np.zeros([height, width]))
        pixelIdxs.append([])

        for rStep in range(refSteps):
            pixelIdxs[h].append([])
            for idx in range(height * width):
                pixelIdxs[h][rStep].append(idx)
            random.shuffle(pixelIdxs[h][rStep])
            localImgPts = []
            localObjPts = []
            for idx in range(height * width):
                x = pixelIdxs[h][rStep][idx] % width
                y = pixelIdxs[h][rStep][idx] // width
                if localDiffMap[y, x] < inlierThreshold2D:
                    localImgPts.append(sampling[y, x, :])
                    localObjPts.append(estObj[y, x, :])
                elif len(localObjPts) >= inlierCount:
                    break
            if len(localObjPts) < 50:
                break
            if len(localObjPts) > 4:
                methodflag = cv2.SOLVEPNP_ITERATIVE
            else:
                methodflag = cv2.SOLVEPNP_AP3P
            localObjPts, localImgPts = np.array([localObjPts]).reshape([-1, 3, 1]), np.array([localImgPts]).reshape([-1, 2, 1])
            hypUpdate = copy.deepcopy(refHyps[h])

            if not (safeSolvePnP(objPts=localObjPts, imgPts=localImgPts, camMat=camMat, disCoeffs=None,
                                 methodFlag=methodflag, rot=hypUpdate[0], trans=hypUpdate[1])[0]).all():
                break

            refHyps[h] = copy.deepcopy(hypUpdate)
        for pt in range(0, len(sampledPoints[h])):
            x = sampledPoints[h][pt][0]
            y = sampledPoints[h][pt][1]
            inlierMaps[h][y, x] = 0

    jpHyp = TYPE.cv2our(refHyps[hypIdx])

    poseEst = Hypothesis.Hypothesis()
    poseEst.RotandTrans(jpHyp[0], jpHyp[1])
    print('beforehy:', poseEst.getRotation(), poseEst.getTranslation())
    poseGT_ashyps = Hypothesis.Hypothesis()
    poseGT_ashyps.Info(poseGT)
    print('beforeGT:', poseGT_ashyps.getRotation(), poseGT_ashyps.getTranslation())

    expectedLoss, losses = expectedMaxLoss(poseGT_ashyps, refHyps, sfScores)

    print('Loss of winning hyp:', ml.maxLoss(poseGT_ashyps, poseEst), ',prob:', sfScores[hypIdx], 'expected loss:',
          expectedLoss, '.')

    invPoseEst = Hypothesis.Hypothesis()
    invPoseGT = Hypothesis.Hypothesis()
    invPoseGT = ml.getInvHyp(poseGT_ashyps)

    invPoseEst = ml.getInvHyp(poseEst)
    print('GT:', invPoseGT.getRotation(), invPoseGT.getTranslation())
    print('Hy:', invPoseEst.getRotation(), invPoseEst.getTranslation())
    rotErr = invPoseGT.calcAngularDistance(invPoseEst)

    tErr = np.linalg.norm(invPoseEst.getTranslation() - invPoseGT.getTranslation())

    correct = False
    if rotErr < 5 and tErr < 50:
        print('Bounded Rotation Err:', rotErr, 'and Bounded Translation Err:', tErr)
        correct = True
    else:
        print('Unbounded Rotation Err:', rotErr, 'and Unbounded Translation Err:', tErr)

    return expectedLoss, sfEntropy, correct, losses, tErr, rotErr, hypIdx
#, sfScores, imgPts, objPts, pixelIdxs, sampledPoints

