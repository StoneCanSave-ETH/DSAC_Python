import numpy as np
import cv2
import copy


class info_t:
    def __init__(self):
        self.name = ""
        self.rotation = np.identity(3)
        self.center = np.array([0,0,-1])
        self.extent = np.array([1,1,1])
        self.visible = True
        self.occlusion = 0


class imag_brgd_t:
    def __init__(self):
        self.bgr = np.zeros([100,100,3])
        self.depth = np.zeros((3,3))

# Convert a pose in our format to OpenCV format


def our2cv(trans): #trans is a list

    [Rows1, Cols1] = trans[0].shape
    rmat = trans[0]
    rvec = np.array([0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 0.0])

    rmat_change = np.zeros([3, 3])
    rmat_change[0, :] = rmat[0, :]
    rmat_change[1, :] = -1*rmat[1, :]
    rmat_change[2, :] = -1*rmat[2, :]
    cv2.Rodrigues(rmat_change, rvec)

    tvec[0] = trans[1][0]
    tvec[1] = -1*trans[1][1]
    tvec[2] = -1*trans[1][2]

    return rvec, tvec

# Convert a ground truth to OpenCV format
def GT2cv(trans):

    rmat = trans.rotation
    rvec = np.array([0.0,0.0,0.0])
    tvec = np.array([0.0,0.0,0.0])

    rmat_change = np.zeros([3, 3])
    rmat_change[1,:] = -1*rmat[1,:]
    rmat_change[2,:] = -1*rmat[2,:]
    cv2.Rodrigues(rmat_change, rvec);

    tvec[0] = trans.center[0]*1000.0;
    tvec[1] = -1*trans.center[1]*1000.0;
    tvec[2] = -1*trans.center[2]*1000.0;

    return rvec, tvec


def cv2our(trans):
    # rmat = np.zeros((3,3))
    rmat = cv2.Rodrigues(trans[0])[0]

    tpt = trans[1]
    tpt_change = copy.deepcopy(tpt)

    rmat[1,:] = -1*rmat[1,:]
    rmat[2,:] = -1*rmat[2,:]
    tpt_change[1] = -1*tpt[1]
    tpt_change[2] = -1*tpt[2]

    if(np.linalg.det(rmat)<0):
        tpt_change = -1*tpt
        rmat = -1*rmat

    return rmat, tpt_change