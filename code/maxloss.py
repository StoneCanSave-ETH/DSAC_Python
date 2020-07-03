import numpy as np
import Hypothesis as hp
import cv2

MAXLOSS = 1000000.0
EPS = 0.00000001
CV_PI = 3.1415926


def getInvHyp(hyp):
    trans = np.eye(4)
    trans[:3, :3] = hyp.getRotation()
    trans[:3, 3] = hyp.getTranslation().reshape([-1])

    trans_inv = np.linalg.inv(trans)
    invRot = trans_inv[:3, :3]
    invtrans = trans_inv[:3, 3]

    invHyp = hp.Hypothesis()
    invHyp.setRotation(invRot)
    invHyp.setTranslation(invtrans)

    return invHyp


def maxLoss(h1, h2):
    invH1 = hp.Hypothesis()
    invH2 = hp.Hypothesis()
    invH1 = getInvHyp(h1)
    invH2 = getInvHyp(h2)

    rotErr = invH1.calcAngularDistance(invH2)
    tErr = np.linalg.norm(invH1.getTranslation()-invH2.getTranslation())

    return min(max(rotErr, tErr/10), MAXLOSS)


def dLossMax(est, gt):
    rod1 = np.zeros(3)
    rod2 = np.zeros(3)
    rod1 = est[0:3] #angle
    rod2 = gt[0:3]

    rot1 = np.zeros((3,3))
    rot2 = np.zeros((3,3))
    dRod = np.zeros((3,9))
    cv2.Rodrigues(np.mat(rod1),rot1,dRod)
    cv2.Rodrigues(np.mat(rod2),rot2)

    invRot1 = np.transpose(rot1)
    invRot2 = np.transpose(rot2)

    diffRot = np.dot(rot1,invRot2)

    trace = diffRot.trace()
    trace = min(3.0, max(-1.0, trace))
    rotErr = 180*np.arccos((trace-1.0)/2.0)/CV_PI # do nt know where CV_PI is defined
    
    invT1 = np.zeros(3)
    invT1[0] = -1*est[3] / 10
    invT1[1] = -1*est[4] / 10
    invT1[2] = -1*est[5] / 10
    invT1 = np.dot(invRot1, invT1)

    invT2 = np.zeros(3)
    invT2[0] = -1*gt[3] / 10
    invT2[1] = -1*gt[4] / 10
    invT2[2] = -1*gt[5] / 10
    invT2 = np.dot(invRot2, invT2)

    tErr = np.linalg.norm(invT1 - invT2)

    jacobian = np.zeros(len(est))

    if(max(rotErr,tErr)>MAXLOSS):
        return jacobian

    if ((tErr+rotErr)< EPS):
        return jacobian
    
    if(tErr>rotErr):

        dDist_dInvT1 = np.zeros(3)
        for i in range(0,3):
            dDist_dInvT1[i] = (invT1[i] - invT2[i]) / tErr

        dInvT1_dEstT = -1*invRot1

        dDist_dEstT = np.dot(dDist_dInvT1, dInvT1_dEstT)
        jacobian[3:6] = dDist_dEstT

        dInvT1_dInvRot1 = np.zeros((3,9))
        dInvT1_dInvRot1[0,0] = -1*est[3] /10
        dInvT1_dInvRot1[0,3] = -1*est[4] /10
        dInvT1_dInvRot1[0,6] = -1*est[5] /10

        dInvT1_dInvRot1[1,1] = -1*est[3] /10
        dInvT1_dInvRot1[1,4] = -1*est[4] /10
        dInvT1_dInvRot1[1,7] = -1*est[5] /10

        dInvT1_dInvRot1[2,2] = -1*est[3] /10
        dInvT1_dInvRot1[2,5] = -1*est[4] /10
        dInvT1_dInvRot1[2,8] = -1*est[5] /10

        dRod = np.transpose(dRod)
        dDist_dRod = np.dot(np.dot(dDist_dInvT1,dInvT1_dInvRot1), dRod)
        jacobian[0:3] = dDist_dRod

    else:
        
        dRod = np.transpose(dRod)

        dRotDiff = np.zeros((9,9))
        dRotDiff[0,0:3] = invRot2[0,:]
        dRotDiff[1,0:3] = invRot2[1,:]
        dRotDiff[2,0:3] = invRot2[2,:]

        dRotDiff[3,3:6] = invRot2[0,:]
        dRotDiff[4,3:6] = invRot2[1,:]
        dRotDiff[5,3:6] = invRot2[2,:]

        dRotDiff[6,6:9] = invRot2[0,:]
        dRotDiff[7,6:9] = invRot2[0,:]
        dRotDiff[8,6:9] = invRot2[0,:]

        dRotDiff = np.transpose(dRotDiff)

        dTrace = np.zeros(9)
        dTrace[0] = 1
        dTrace[4] = 1
        dTrace[8] = 1

        dAngle = (180 / CV_PI*(-1)/np.sqrt(3 - trace*trace+2*trace))*np.dot(np.dot(dTrace,dRotDiff),dRod)
        jacobian[0:3] = dAngle
        return jacobian