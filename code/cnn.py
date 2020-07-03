import torch
import cv2
import random
import copy
import math
import time
import os
import kornia

import cnn
import Customized_Datasets
import Model_obj
import Model_score
import Hypothesis
import properties
import BPnP
import TYPE
import maxloss
import cnn_Sam

import numpy as np
import torch.nn.functional as F

from torchvision import transforms
from numpy.linalg import norm
from torch.autograd import gradcheck


CNN_RGB_PATCHSIZE = 42
CNN_OBJ_PATCHSIZE = 40
CNN_OBJ_MAXINPUT = 100
BATCHSIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRANSFORM = transforms.Compose([
    transforms.Normalize([0.5], [0.5])
])
storeIntervalE2E = 100

# For recording
E2E_FILE = './end2end/'
if not os.path.exists(E2E_FILE):
    os.mkdir(E2E_FILE)
OBJ_E2E = E2E_FILE + 'model_obj_e2e.pkl'
SCORE_E2E = E2E_FILE + 'model_score_e2e.pkl'


def safesolvepnp(objPts, imgPts, camMat, disCoeffs, methodFlag, rot, trans):
    """
    we should transfer the form of objPts and imgPts from list to np.array before using it
    :param objPts:
    :param imgPts:
    :param camMat:
    :param disCoeffs:
    :param methodFlag:
    :return:
    """
    if not torch.is_tensor(rot):
        retval, rvec, tvec = cv2.solvePnP(objPts, imgPts, camMat, disCoeffs, None, None, 0, methodFlag)
        if not retval:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
        return [rvec.reshape([-1]), tvec.reshape([-1])]
    else:
        rot, trans = rot.cpu().detach().numpy(), trans.cpu().detach().numpy()
        retval, rvec, tvec = cv2.solvePnP(objPts, imgPts, camMat, disCoeffs, rot, trans, 0, methodFlag)
        if not retval:
            rvec = np.zeros(3)
            tvec = np.zeros(3)
        return [rvec.reshape([-1]), tvec.reshape([-1])]


def getCoordImg_tensor(colorData, sampling, patchsize, model):
    transform = cnn.Transform_OBJ()
    DATA = Customized_Datasets.GetCoorData(sampling=sampling, patchsize=patchsize,
                                              colorData=colorData, transform=transform)
    data_loader = torch.utils.data.DataLoader(DATA, batch_size=BATCHSIZE, shuffle=False, num_workers=8)
    for idx, (BatchData) in enumerate(data_loader):
        BatchData = BatchData.to(DEVICE).float()
        with torch.no_grad():
            pred = model(BatchData)
        # pred = Model_obj.forward_tensor(model, BatchData, DEVICE)
        if not idx:
            patch = pred
        else:
            patch = torch.cat((patch, pred), dim=0)
    (height, width) = np.shape(sampling[:, :, 0])
    modeImg = patch.view([height, width, 3]) * 1000.0
    return modeImg


# def rotvec2mrx(hyps):
#     theta = torch.norm(hyps[:3]).to(DEVICE)
#     vector = (hyps[:3] / theta).view([-1, 1]).to(DEVICE)
#     vec_mrx = torch.tensor([[0, -1 * vector[2, 0], vector[1, 0]],
#                             [vector[2, 0], 0, -1 * vector[0, 0]],
#                             [-1 * vector[1, 0], vector[0, 0], 0]]).to(DEVICE)
#     RotMrx = torch.cos(theta) * torch.eye(3).to(DEVICE) + (1 - torch.cos(theta)).to(DEVICE) \
#              * torch.mm(vector, torch.transpose(vector, 0, 1)) \
#              + torch.sin(theta).to(DEVICE) * vec_mrx
#     return RotMrx


def getdiff_tensor(objectPoints, hyps, pixelx, pixely, RotMrx, f, ppx, ppy):
    coor = objectPoints.view([-1, 1]).double().to(DEVICE)
    obj = torch.mm(RotMrx, coor) + hyps[3:].view([-1, 1]).to(DEVICE)
    # px = f * obj[0, :] / obj[2, :] + ppx
    # py = f * obj[1, :] / obj[2, :] + ppy
    # repr_err = math.sqrt((px - pixelx) * (px - pixelx) + (py - pixely) * (py - pixely))
    # min_err = torch.tensor(min(repr_err, 100.0), requires_grad=True)
    return obj


def obj2cam(objectPoints, trans, RotMrx):
    coor = objectPoints.view([-1, 1]).double().to(DEVICE)
    obj = torch.mm(RotMrx, coor) + trans
    return obj.view([-1])


def getDiffMap_tensor(hyp, objectCoordinates, sampling, camMat):

    # obj_cam = torch.zeros([height, width, 3]).to(DEVICE).double()
    RotMrx = kornia.angle_axis_to_rotation_matrix(hyp[0, :3].view(1, -1)).view([3, 3]).float()
    f = camMat[0, 0]
    ppx = camMat[0, 2]
    ppy = camMat[1, 2]
    # trans = hyp[3:].view([-1, 1]).to(DEVICE)
    # object_tensor = objectCoordinates.view([1600, 3, 1]).to(DEVICE).double()
    # Rotmrx_tensor = RotMrx.unsqueeze(0).repeat(1600, 1, 1).to(DEVICE).double()
    # print('hyp:', hyp[0, 3] * torch.ones([40, 40, 1]))
    # trans_mrx = torch.cat((hyp[0, 3] * torch.ones([40, 40, 1]),
    #                        hyp[0, 4] * torch.ones([40, 40, 1]),
    #                        hyp[0, 5] * torch.ones([40, 40, 1])), 2)
    # print('trans:', trans_mrx)
    # obj_cam_tensor = torch.bmm(Rotmrx_tensor, object_tensor).view([40, 40, 3]).to(DEVICE) + trans_mrx
    obj_cam_tensor = torch.matmul(objectCoordinates, torch.transpose(RotMrx, 0, 1))
    obj_cam_tensor[:, :, 0] += hyp[0, 3]
    obj_cam_tensor[:, :, 1] += hyp[0, 4]
    obj_cam_tensor[:, :, 2] += hyp[0, 5]

    # print('trans_mrx:', trans_mrx.shape)
    # print('rot:', RotMrx.shape)
    # obj_cam_tensor = torch.matmul(RotMrx, object_tensor) + trans_mrx

    # for i in range(width * height):
    #     x = i % width
    #     y = i // width
    #
    #     obj_cam[y, x, :] = obj2cam(objectPoints=objectCoordinates[y, x, :],
    #                                       trans=trans,
    #                                       RotMrx=RotMrx)
    #     print('obj_cam:', obj_cam[y, x, :])
    #     print('obj_cam_tensor:', obj_cam_tensor[y, x, :])

    project_x = f * obj_cam_tensor[:, :, 0] / obj_cam_tensor[:, :, 2] + ppx
    project_y = f * obj_cam_tensor[:, :, 1] / obj_cam_tensor[:, :, 2] + ppy
    reproject_x = project_x - sampling[:, :, 0]
    reproject_y = project_y - sampling[:, :, 1]
    diff = torch.sqrt(reproject_x ** 2 + reproject_y ** 2)
    diffMap = torch.clamp(diff, 0., 100.)
    diffMap = torch.unsqueeze(diffMap, 0)
    # diff = torch.zeros([1, height, width]).to(DEVICE)
    # diff[0, :, :] = torch.sqrt(reproject_x ** 2 + reproject_y ** 2).to(DEVICE)
    # diffMap = torch.where(diff > 100., torch.full_like(diff, 100.), diff).requires_grad_(True)
        # diffMap[0, y, x] = getdiff_tensor(objectPoints=objectCoordinates[y, x, :],
        #                                   hyps=hyp,
        #                                   pixelx=sampling[y, x][0],
        #                                   pixely=sampling[y, x][1],
        #                                   RotMrx=RotMrx,
        #                                   f=f, ppx=ppx, ppy=ppy).to(DEVICE)
    return diffMap


class PNP(torch.autograd.Function):
    @staticmethod
    # forward should be from objPts to [rvec, tvec], namely safesolvePnP
    # except for objPts, all other input variable should be non-tensor,
    # so their gradient should return None.
    def forward(ctx, objPts, imgPts, camMat, disCoeffs, methodFlag, rot, trans):
        ctx.imgPts = imgPts
        ctx.camMat = camMat
        ctx.disCoeffs = disCoeffs
        objPts_np = objPts.cpu().detach().numpy()
        """
            we should transfer the form of objPts and imgPts from list to np.array before using it
            :param objPts:
            :param imgPts:
            :param camMat:
            :param disCoeffs:
            :param methodFlag:
            :return:
            """
        if torch.is_tensor(rot):
            retval, rvec, tvec = cv2.solvePnP(objPts_np, imgPts, camMat, disCoeffs, None, None, 0, methodFlag)
            if not retval:
                rvec = np.zeros(3)
                tvec = np.zeros(3)

            rvec_tensor = torch.tensor(rvec.reshape([-1]))
            tvec_tensor = torch.tensor(tvec.reshape([-1]))
            ctx.save_for_backward(rvec_tensor, tvec_tensor, objPts)
            rt_vec = torch.cat((rvec_tensor, tvec_tensor))
            return rt_vec
        else:
            # rot, trans = rot.cpu().detach().numpy(), trans.cpu().detach().numpy()
            retval, rvec, tvec = cv2.solvePnP(objPts_np, imgPts, camMat, disCoeffs, rot, trans, 0, methodFlag)
            if not retval:
                rvec = np.zeros(3)
                tvec = np.zeros(3)

            # save tensor for backward use
            rvec_tensor = torch.tensor(rvec.reshape([-1]))
            tvec_tensor = torch.tensor(tvec.reshape([-1]))
            ctx.save_for_backward(rvec_tensor, tvec_tensor, objPts)
            rt_vec = torch.cat((rvec_tensor, tvec_tensor))
            return rt_vec

    @staticmethod
    def backward(ctx, grad_output):
        rvec, tvec, objPts = ctx.saved_tensors
        objPts_grad = None
        rvec_np = rvec.cpu().detach().numpy()
        tvec_np = tvec.cpu().detach().numpy()
        objPts_np = objPts.cpu().detach().numpy()
        eps = 10
        # if np.size(ctx.imgPts, 0) == 4:
        #     pnpMethod = cv2.SOLVEPNP_P3P
        # else:
        #     pnpMethod = cv2.SOLVEPNP_ITERATIVE
        pnpMethod = cv2.SOLVEPNP_P3P
        gp = properties.GlobalProperties()
        camMat = gp.getCamMat()
        imgPts = np.array(ctx.imgPts, np.float64)
        jacobean = np.zeros([6, np.size(objPts_np, 0) * 3])
        for i in range(np.size(objPts_np, 0)):
            for j in range(3):
                # Forward step
                if j == 0:
                    objPts_np[i][0] += eps
                elif j == 1:
                    objPts_np[i][1] += eps
                elif j == 2:
                    objPts_np[i][2] += eps
                objPts_np = np.array(objPts_np, np.float64)
                # cnn.safeSolvePnP(objPts=objPts_3D, imgPts=imgPts_3D, camMat=camMat, disCoeffs=None,
                                 #                             methodFlag=cv2.SOLVEPNP_P3P, rot=np.zeros([3]), trans=np.zeros([3])
                rot_f, tvec_f = cnn.safeSolvePnP(objPts=objPts_np, imgPts=imgPts, camMat=camMat, disCoeffs=None,
                                        methodFlag=pnpMethod, rot=np.zeros([3]), trans=np.zeros([3]))
                # rotf_tensor = torch.tensor(rot_f).to(DEVICE)
                # tvecf_tensor = torch.tensor(tvec_f).to(DEVICE)

                # Trans_f = TYPE.cv2our([rot_f, tvec_f])
                #
                # h_f = Hypothesis.Hypothesis()
                # h_f.RotandTrans(Trans_f[0], Trans_f[1])
                # fstep = h_f.getRodVecAndTrans()
                fstep = np.concatenate((rot_f, tvec_f))

                # Backward step
                if j == 0:
                    objPts_np[i][0] -= 2 * eps
                elif j == 1:
                    objPts_np[i][1] -= 2 * eps
                elif j == 2:
                    objPts_np[i][2] -= 2 * eps
                objPts_np = np.array(objPts_np, np.float64)

                rot_b, tvec_b = cnn.safeSolvePnP(objPts=objPts_np, imgPts=imgPts, camMat=camMat, disCoeffs=None,
                                        methodFlag=pnpMethod, rot=np.zeros([3]), trans=np.zeros([3]))

                # Trans_b = TYPE.cv2our([rot_b, tvec_b])
                # h_b = Hypothesis.Hypothesis()
                # h_b.RotandTrans(Trans_b[0], Trans_b[1])
                # bstep = h_b.getRodVecAndTrans()
                bstep = np.concatenate((rot_b, tvec_b))

                # Back to normal state
                if j == 0:
                    objPts_np[i][0] += eps
                elif j == 1:
                    objPts_np[i][1] += eps
                elif j == 2:
                    objPts_np[i][2] += eps

                # Gradient calculation
                for k in range(len(fstep)):
                    jacobean[k][3 * i + j] = (fstep[k] - bstep[k]) / (2 * eps)
                if cnn.containsNaNs(jacobean[:, 3 * i + j]):
                    return np.zeros([6, 3 * np.size(objPts_np, 0)])

        jacobean_tensor = torch.tensor(jacobean, requires_grad=True).double().to(DEVICE)
        objPts_grad = torch.mm(grad_output.view([1, -1]).to(DEVICE), jacobean_tensor).view([-1, 3]).double().requires_grad_(True).to(DEVICE)
        return objPts_grad, None, None, None, None, None, None


def processImage(
                 imgBGR,
                 poseGT,
                 model_obj,
                 model_score,
                 objHyps,
                 ptCount,
                 camMat,
                 inlierThreshold2D,
                 inlierCount,
                 refSteps,
                 refHyps,
                sampledPoints,
                 estObj,
                 sampling,
                 inlierMaps,
                 pixelIdxs,
                 optimizer,
                 round):
    time_start = time.time()
    camMat_tensor = torch.tensor(camMat).to(DEVICE).float()
    sampling = cnn_Sam.stochasticSubSamplewithoutC(imgBGR, CNN_OBJ_PATCHSIZE, CNN_RGB_PATCHSIZE)
    sampling_tensor = torch.tensor(sampling).to(DEVICE)
    estObj_tensor = getCoordImg_tensor(imgBGR, sampling, CNN_RGB_PATCHSIZE, model_obj).float()
    estobj_np = estObj_tensor.cpu().detach().numpy()
    solvePNP = PNP.apply
    bpnp = BPnP.BPnP.apply
    time_point1 = time.time()
    print('cost time in obj net:', time_point1 - time_start)

    for h in range(0, objHyps):
        if not h % 500:
            print('h:', h)
        p = 0
        while True:

            # projections = []
            alreadyChosen = np.zeros((40, 40))

            # imgPts.append([])
            # objPts.append([])
            # imgIdx.append([])  # the usage of .clear()
            sampledPoints.append([])
            # hyps.append([])

            j = 0
            # imgPts = []
            # objPts = []
            imgIdx = []
            while j < ptCount:
                np.random.seed()
                x = np.random.randint(0, 40)
                y = np.random.randint(0, 40)
                j = j + 1
                if alreadyChosen[y, x] > 0:
                    j = j - 1
                    continue

                alreadyChosen[y, x] = 1
                # imgPts.append(sampling[y, x, :])
                if j == 1:
                    imgPts = sampling_tensor[y, x, :].unsqueeze(0)
                    objPts = estObj_tensor[y, x, :].unsqueeze(0)
                else:
                    imgPts = torch.cat((imgPts,  sampling_tensor[y, x, :].unsqueeze(0)), 0)
                    objPts = torch.cat((objPts, estObj_tensor[y, x, :].unsqueeze(0)), 0)


                # objPts.append(estObj_tensor[y, x, :])
                imgIdx.append(y * CNN_OBJ_PATCHSIZE + x)
                sampledPoints[h].append(np.array([x, y]))
            # imgPts_3D = np.array(imgPts).reshape([1, -1, 2])
            imgPts = imgPts.unsqueeze(0).float()
            # if p:
            #     rtvec = solvePNP(objPts,
            #                     imgPts_3D,
            #                     camMat,
            #                     None,
            #                     cv2.SOLVEPNP_P3P,
            #                     hyps[:3],
            #                     hyps[3:]).reshape([-1]).to(DEVICE)
                # rtvec = np.array(cnn.safeSolvePnP(objPts_3D,
                #                  imgPts_3D,
                #                  camMat,
                #                  None,
                #                  cv2.SOLVEPNP_P3P,
                #                  hyps[h][:3],
                #                  hyps[h][3:])).reshape([-1])
            # else:
            rtvec = bpnp(imgPts,
                         objPts,
                         camMat_tensor)
            # rtvec = solvePNP(objPts,
            #                  imgPts_3D,
            #                  camMat,
            #                  None,
            #                  cv2.SOLVEPNP_P3P,
            #                  np.zeros([3]),
            #                  np.zeros([3])).reshape([-1]).to(DEVICE)
                # rtvec = np.array(cnn.safeSolvePnP(objPts_3D,
                #                          imgPts_3D,
                #                          camMat,
                #                          None,
                #                          cv2.SOLVEPNP_P3P,
                #                          np.zeros([3]),
                #                          np.zeros([3]))).reshape([-1])
            if not np.any(np.isfinite(rtvec.cpu().detach().numpy())):
                del imgPts
                del objPts
                del imgIdx
                del sampledPoints[h]
                continue
            else:
                hyps = rtvec
                # if not p:
                #     hyps = rtvec
                #     p += 1
                # else:
                #     hyps = rtvec
            # hyps.append(solvePNP(objPts_3D_tensor,
            #                         imgPts_3D,
            #                         camMat,
            #                         None,
            #                         cv2.SOLVEPNP_P3P,
            #                         np.zeros([3]),
            #                         np.zeros([3])).cpu().detach().numpy().reshape([-1]))
            # hyps[h].append(solvePNP(objPts_3D_tensor,
            #                         imgPts_3D,
            #                         camMat,
            #                         None,
            #                         cv2.SOLVEPNP_P3P,
            #                         np.zeros([3]),
            #                         np.zeros([3]))[3:].cpu().detach().numpy())
            # input = (objPts,
            #                         imgPts_3D,
            #                         camMat,
            #                         None,
            #                         cv2.SOLVEPNP_P3P,
            #                         np.zeros([3]),
            #                         np.zeros([3]))
            # test = gradcheck(solvePNP, input, eps=1e-4)
            # print('test_grad:', test)

            # hyps[h].append(cnn.safeSolvePnP(objPts=objPts_3D, imgPts=imgPts_3D, camMat=camMat, disCoeffs=None,
            #                             methodFlag=cv2.SOLVEPNP_P3P, rot=np.zeros([3]), trans=np.zeros([3]))[0])
            # hyps[h].append(cnn.safeSolvePnP(objPts=objPts_3D, imgPts=imgPts_3D, camMat=camMat, disCoeffs=None,
            #                             methodFlag=cv2.SOLVEPNP_P3P, rot=np.zeros([3]), trans=np.zeros([3]))[1])

            # if not (solvePNP(objPts_3D_tensor,
            #                         imgPts_3D,
            #                         camMat,
            #                         None,
            #                         cv2.SOLVEPNP_P3P,
            #                         np.zeros([3]),
            #                         np.zeros([3]))[:3].cpu().detach().numpy()).all():
            #     del imgPts[h]
            #     del objPts[h]
            #     del imgIdx[h]
            #     del sampledPoints[h]
            #     del hyps[h]
            #     continue


        # to project a 3d point into the image(do not know whether there is a function in cv2 can fulfill the task)
        #     print('objpts:', np.array(objPts[h]).reshape([-1, 3]))
            projections = cv2.projectPoints(objectPoints=objPts.cpu().detach().numpy(),
                                                 rvec=hyps[0, :3].cpu().detach().numpy(),
                                                  tvec=hyps[0, 3:].cpu().detach().numpy(),
                                                  cameraMatrix=camMat, distCoeffs=None)[0].reshape([-1, 2])

            # for points in range(4):
            #     objMat = objPts[h][points]
            #     projection, _ = cv2.projectPoints(objectPoints=objMat, rvec=hyps[h][:3],
            #                                       tvec=hyps[h][3:],
            #                                       cameraMatrix=camMat, distCoeffs=None)
            #     projections.append(projection)
            foundOutlier = False
            if np.any(norm((projections - imgPts.cpu().detach().numpy().reshape([-1, 2])), axis=1) > inlierThreshold2D):
                foundOutlier = True
            # for j in range(4):
            #     # print("diff is:",np.linalg.norm(imgPts[h][j] - projections[j]))
            #     if np.linalg.norm(imgPts[j] - projections[j]) < inlierThreshold2D:
            #         continue
            #     foundOutlier = True
            #     break

            if foundOutlier:
                del imgPts
                del objPts
                del imgIdx
                del sampledPoints[h]
                # del hyps
                p += 1
                continue
            else:
                break

        if not h:
            hyps_tensor = hyps.unsqueeze(0)
        else:
            hyps_tensor = torch.cat((hyps_tensor, hyps.unsqueeze(0)), 0)
    time_point2 = time.time()
    print('cost time in PNP:', time_point2 - time_point1)

    meet_demand = 0
    index = 0
    for h in range(objHyps):
        diffmap = getDiffMap_tensor(hyps_tensor[h, :], estObj_tensor, sampling_tensor, camMat)

        # Calculate inliers
        mask = diffmap[0, :, :].lt(inlierThreshold2D)
        num = torch.nonzero(mask, as_tuple=False).size(0)
        if num > meet_demand:
            meet_demand = num
            index = h

        # Transform for later use
        diffmap_norm = TRANSFORM(diffmap).unsqueeze(0)
        if not h:
            FullDiffMap_norm = diffmap_norm
            FullDiffMap = diffmap.unsqueeze(0)
        else:
            FullDiffMap_norm = torch.cat((FullDiffMap_norm, diffmap_norm), 0)
            FullDiffMap = torch.cat((FullDiffMap, diffmap.unsqueeze(0)), 0)

    time_point3 = time.time()
    print('cost time in get diffmap:', time_point3 - time_point2)

    with torch.no_grad():
        score = model_score(FullDiffMap_norm.float())
    # max_score = torch.max(score).to(DEVICE)
    # score_norm = score - max_score
    sfScore = F.softmax(score, dim=0)
    sfScore_np = sfScore.cpu().detach().numpy().reshape([-1])
    hypIdx = cnn_Sam.draw(sfScore_np, randomdraw=True)
    hypIdx_ransac = cnn_Sam.draw(sfScore_np, randomdraw=False)

    time_point4 = time.time()
    print('cost time in score net:', time_point4 - time_point3)


    # Refine
    (height, width) = np.shape(sampling[:, :, 0])
    inlierMaps = []
    hyps_np = hyps_tensor.cpu().detach().numpy()
    for h in range(objHyps):
        refHyps.append(hyps_np[h, :, :].reshape([-1]))
        localDiffMap = FullDiffMap[h, 0, :, :].cpu().detach().numpy()
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
                    localObjPts.append(estobj_np[y, x, :])
                    # print('sampling:', sampling[y, x, :])
                    # print('est', estobj_np[y, x, :])
                elif len(localObjPts) >= inlierCount:
                    break
            if len(localObjPts) < 50:
                break
            if len(localObjPts) > 4:
                methodflag = cv2.SOLVEPNP_ITERATIVE
            else:
                methodflag = cv2.SOLVEPNP_P3P
            localObjPts, localImgPts = np.array([localObjPts]).reshape([-1, 3]), np.array([localImgPts]).reshape([-1, 2])
            # localObjPts_tensor = torch.tensor(localObjPts, requires_grad=False)
            # print('localobj:', localObjPts)
            # print('localimg:', localImgPts)
            hypUpdate = copy.deepcopy(refHyps[h])
            # print('h:', h, 'refstep:', rStep, 'b4', hypUpdate)
            # print('h:', h, 'refstep:', rStep, 'b4', hypUpdate[0], 'trans:', hypUpdate[1])

            if not (cnn.safeSolvePnP(objPts=localObjPts,
                                      imgPts=localImgPts,
                                      camMat=camMat,
                                      disCoeffs=None,
                                      methodFlag=methodflag,
                                      rot=hypUpdate[:3],
                                      trans=hypUpdate[3:])[0]).all:
            # rvec_tensor = rt_vec[:3]
            # tvec_tensor = rt_vec[3:]
            # rvec = rvec_tensor.cpu().detach().numpy()
            # tvec = tvec_tensor.cpu().detach().numpy()
                break
            # print('h:', h, 'refstep:', rStep, 'after:', hypUpdate)
            refHyps[h] = copy.deepcopy(hypUpdate)

        if not h:
            REF_HYP = refHyps[h].reshape([1, -1])
        else:
            REF_HYP = np.concatenate((REF_HYP, refHyps[h].reshape([1, -1])), axis=0)

        for pt in range(0, len(sampledPoints[h])):
            x = sampledPoints[h][pt][0]
            y = sampledPoints[h][pt][1]
            inlierMaps[h][y, x] = 0


    time_point5 = time.time()
    print('cost time in refine:', time_point5 - time_point4)

    # For GT
    poseGT_ashyps = Hypothesis.Hypothesis()
    poseGT_ashyps.Info(poseGT)
    invPoseGT = Hypothesis.Hypothesis()
    invPoseGT = maxloss.getInvHyp(poseGT_ashyps)

    '''
    Note: hypIdx means sample/get argmax of score
          index means get highest inlier
    '''

    # For hypothesis
    jpHyp = TYPE.cv2our([REF_HYP[index, :3], REF_HYP[index, 3:]])
    poseEst = Hypothesis.Hypothesis()
    poseEst.RotandTrans(jpHyp[0], jpHyp[1])
    invPoseEst = Hypothesis.Hypothesis()
    invPoseEst = maxloss.getInvHyp(poseEst)

    jpHyp_score = TYPE.cv2our([REF_HYP[hypIdx, :3], REF_HYP[hypIdx, 3:]])
    poseEst_score = Hypothesis.Hypothesis()
    poseEst_score.RotandTrans(jpHyp_score[0], jpHyp_score[1])
    invPoseEst_score = Hypothesis.Hypothesis()
    invPoseEst_score = maxloss.getInvHyp(poseEst_score)

    jpHyp_ransac = TYPE.cv2our([REF_HYP[hypIdx_ransac, :3], REF_HYP[hypIdx_ransac, 3:]])
    poseEst_ransac = Hypothesis.Hypothesis()
    poseEst_ransac.RotandTrans(jpHyp_ransac[0], jpHyp_ransac[1])
    invPoseEst_ransac = Hypothesis.Hypothesis()
    invPoseEst_ransac = maxloss.getInvHyp(poseEst_ransac)

    print('GT:', invPoseGT.getRotation(), invPoseGT.getTranslation())
    print('Hy:', invPoseEst.getRotation(), invPoseEst.getTranslation())
    rotErr = invPoseGT.calcAngularDistance(invPoseEst)
    tErr = np.linalg.norm(invPoseEst.getTranslation() - invPoseGT.getTranslation())


    rotErr_score = invPoseGT.calcAngularDistance(invPoseEst_score)
    tErr_score = np.linalg.norm(invPoseEst_score.getTranslation() - invPoseGT.getTranslation())

    rotErr_ransac = invPoseGT.calcAngularDistance(invPoseEst_ransac)
    tErr_ransac = np.linalg.norm(invPoseEst_ransac.getTranslation() - invPoseGT.getTranslation())

    correct = False
    if rotErr < 5 and tErr < 50:
        print('Bounded Rotation Err:', rotErr, 'and Bounded Translation Err:', tErr)
        correct = True
    else:
        print('Unbounded Rotation Err:', rotErr, 'and Unbounded Translation Err:', tErr)

    correct_socre = False
    if rotErr_score < 5 and tErr_score < 50:
        print('Bounded Rotation Err_score:', rotErr_score, 'and Bounded Translation Err_score:', tErr_score)
        correct_socre = True
    else:
        print('Unbounded Rotation Err_score:', rotErr_score, 'and Unbounded Translation Err_ransac:', tErr_score)

    correct_ransac = False
    if rotErr_ransac < 5 and tErr_ransac < 50:
        print('Bounded Rotation Err_ransac:', rotErr_ransac, 'and Bounded Translation Err_ransac:', tErr_ransac)
        correct_ransac = True
    else:
        print('Unbounded Rotation Err_ransac:', rotErr_ransac, 'and Unbounded Translation Err_ransac:', tErr_ransac)



    return correct, correct_socre, correct_ransac, rotErr, tErr, rotErr_score, tErr_score, rotErr_ransac, tErr_ransac

    # GT_rot_our = poseGT_ashyps.getRotation()
    # GT_trans_our = poseGT_ashyps.getTranslation()
    #
    # GT_rot_cv = np.array([GT_rot_our[0, :],
    #                       -1 * GT_rot_our[1, :],
    #                       -1 * GT_rot_our[2, :]])
    # GT_trans_cv = np.array([GT_trans_our[0], -1 * GT_trans_our[1], -1 * GT_trans_our[2]])
    # GT_rot_cv = torch.tensor(GT_rot_cv).to(DEVICE)
    # GT_trans_cv = torch.tensor(GT_trans_cv).to(DEVICE)
    # print('GT_rot:', GT_rot_cv)
    # print('GT_trans:', GT_trans_cv)
    #
    # print('hyps_rot:', cv2.Rodrigues(REF_HYP[hypIdx, :3])[0])
    # print('hyps_trans:', REF_HYP[hypIdx, 3:])

    # for i in range(objHyps):
    #     RotMrx = kornia.angle_axis_to_rotation_matrix(hyps_tensor[i, 0, :3].view([1, -1])).view([3, 3]).float()
    #     Trans = hyps_tensor[i, 0, 3:]
    #     trace = torch.trace(torch.mm(RotMrx, torch.inverse(GT_rot_cv).float()))
    #     trace = torch.clamp(trace, -1., 3.)
    #     dist_rot = (180 * torch.acos((trace - 1.) / 2.) / math.pi)
    #
    #     dist_trans = torch.norm(GT_trans_cv - Trans).float()
    #
    #     dist = torch.max(dist_rot, dist_trans / 10.)
    #     if not i:
    #         expectedLoss = dist * sfScore[i, :]
    #     else:
    #         expectedLoss += dist * sfScore[i, :]
        # loss_vec[i] = dist * sfScore[i, :]

    # expectedLoss = torch.sum(loss_vec).to(DEVICE)



    # See intermediate grad
    # grads = {}
    # def save_grad(name):
    #     def hook(grad):
    #         grads[name] = grad
    #     return hook
    # pred.register_hook
    #
    # diffmap.register_hook(save_grad('diffmap'))
    # hyps_tensor.register_hook(save_grad('hyps'))

    # See intermediate grad
    # grads = {}

    # def save_grad(name):
    #     def hook(grad):
    #         grads[name] = grad
    #
    #     return hook
    # # estObj_tensor.register_hook(save_grad('est'))
    # estObj_tensor.register_hook(save_grad('obj'))
    # rtvec.register_hook(save_grad('rtvec'))
    #
    # estObj_tensor.register_hook(save_grad('est'))
    # refHyps_tensor.register_hook(save_grad('ref'))

    # optimizer.zero_grad()
    #
    # expectedLoss.backward()
    # optimizer.step()
    #
    # time_point7 = time.time()
    # print('cost time in backward:', time_point7 - time_point6)
    # print('expected loss:', expectedLoss.item())
    #
    # if not round % storeIntervalE2E:
    #     torch.save(model_obj.state_dict(), OBJ_E2E)
    #     torch.save(model_score.state_dict(), SCORE_E2E)


    # diffmap.register_hook(save_grad('diffmap'))
    # #
    # hyps_tensor.register_hook(save_grad('hyps'))
    # print('hyps', grads['hyps'])
    # refHyps_tensor.register_hook(save_grad('refhyps'))

    # diffmap.register_hook(save_grad('diff_map'))
    # print(grads['diff_map'])
    # estObj_tensor.register_hook(save_grad('est_obj'))
    # expectedLoss, losses = cnn.expectedMaxLoss(poseGT_ashyps, refHyps_tensor, sfScore)
    # print('diffmap:', grads['diffmap'])
    # print('hyps:', grads['hyps'])






# dataDir = './'
# trainingDir = dataDir + 'training/'
# trainingSets = util.getSubPaths(trainingDir)
# trainingDataset = dataset.Dataset()
# trainingDataset.readFileNames(trainingSets[0])
# trainingDataset.SetObjID(1)
#
# RGB_NET = OBJ_CNN().to(DEVICE)
# RGB_NET.load_state_dict(torch.load('./Model parameter/obj_model_init.pkl', map_location=DEVICE))
#
# SCORE_NET = SCORE_CNN().to(DEVICE)
# SCORE_NET.load_state_dict(torch.load('./Model parameter/score_model_init_03.05.2020(0.5).pkl', map_location=DEVICE))
#
# for i in range(2):
#     imgBGR = trainingDataset.getBGR(i)
#     info = trainingDataset.getInfo(i)
#     sampling = cnn.stochasticSubSample(imgBGR, CNN_OBJ_PATCHSIZE, CNN_RGB_PATCHSIZE)
#     estObj = getCoordImg_tensor(imgBGR, sampling, CNN_RGB_PATCHSIZE, RGB_NET)
#     cam = properties.GlobalProperties().getCamMat()
#
#     poseGT = Hypothesis.Hypothesis()
#     poseGT.Info(info)
#     hyp = TYPE.our2cv([poseGT.getRotation(), poseGT.getTranslation()])
#     hyp_tensor = torch.tensor(hyp, requires_grad=True)
#     diff = getDiffMap_tensor(hyp_tensor, estObj, sampling, cam)
#     diff_transform = TRANSFORM(diff).unsqueeze(0)
#     if not i:
#         FullDiff = diff_transform
#     else:
#         FullDiff = torch.cat((FullDiff, diff_transform), 0)
#
# score = Model_score.forward_tensor(SCORE_NET, data=FullDiff, device=DEVICE)
# max_score = torch.max(score)
# score_norm = score - max_score
# sfScore = F.softmax(score_norm, dim=0)
# sfScore_np = sfScore.cpu().detach().numpy()
# hypIdx = cnn.draw(sfScore_np)
# print('max:', sfScore[hypIdx, :])
# print('socre:', score)
# print('socre_size:', score.shape)
# print('score_norm:', score_norm)
# print('score_softmax:', sfScore)


# rot = np.array([[0.43159801],
#        [0.02226284],
#        [0.04372219]]).reshape([-1])
# trans = np.array([[536.67967025],
#        [-70.1690938],
#        [3304.77974144]]).reshape([-1])
# hyps = [rot, trans]
# hyps_tensor = torch.tensor(hyps, requires_grad=True)
#
# theta = torch.norm(hyps_tensor[0, :])
# vector = (hyps_tensor[0, :] / theta).view([-1, 1])
# vec_mrx = torch.tensor([[0, -1 * vector[2, 0], vector[1, 0]],
#                         [vector[2, 0], 0, -1 * vector[0, 0]],
#                         [-1 * vector[1, 0], vector[0, 0], 0]])
#
# RotMrx = torch.cos(theta) * torch.eye(3) + (1 - torch.cos(theta)) * torch.mm(vector, torch.transpose(vector, 0, 1)) \
#         + torch.sin(theta) * vec_mrx
# f = cam[0, 0]
# ppx = cam[0, 2]
# ppy = cam[1, 2]
# coor = torch.tensor([0.1818, -0.0475, -0.1695]).view([-1, 1]).double()
# obj = torch.mm(RotMrx, coor) + hyps_tensor[1, :].view([-1, 1])
#
# px = f * obj[0, :] / obj[2, :] + ppx
# py = f * obj[1, :] / obj[2, :] + ppy
#
# repr_err = (px - sampling[2, 2][0]) * (px - sampling[2, 2][0]) + (py -sampling[2, 2][1]) * (py -sampling[2, 2][1])
# min_err = torch.min(torch.tensor([repr_err, 100]))


# cam = properties.GlobalProperties().getCamMat()
# print(cam)



