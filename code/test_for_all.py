import os
import time

import dataset
import torch
import properties
import util
import cnn

import numpy as np

from Model_obj import OBJ_CNN
from Model_score import SCORE_CNN

lrInitE2E = 0.00001
momentumE2E = 0.9
storeIntervalE2E = 1000


def main():
    GlobalProp = properties.GlobalProperties()
    objHyps = GlobalProp.ransacIterations
    refSteps = GlobalProp.ransacRefinementIterations
    inlierThreshold2D = GlobalProp.ransacInlierThreshold2D
    refInlierCount = GlobalProp.ransacBatchSize
    ptCount = 4
    CNN_OBJ_PATCHSIZE = 40
    camMat = GlobalProp.getCamMat()
    CamMat = camMat.astype(np.float)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Traning path setting
    dataDir = './'
    trainingDir = dataDir + 'training/'
    trainingSets = util.getSubPaths(trainingDir)
    print("Loading train set...\n")

    # Initialize training datasets
    training = dataset.Dataset()
    training.readFileNames(trainingSets[0])
    training.SetObjID(1)

    # For loss recording
    testfile = open('./Model parameter/chess/scene/ransac_test_loss.txt', 'a')

    # set for coordinate net
    model_obj = OBJ_CNN().to(DEVICE)
    model_obj.load_state_dict(torch.load('./Model parameter/chess/scene/obj_model_init.pkl'))

    # set for score net
    model_score = SCORE_CNN().to(DEVICE)
    model_score.load_state_dict(torch.load('./Model parameter/chess/scene/score_model_init.pkl'))

    model_obj.eval()
    model_score.eval()

    avgCorrect = 0
    expLosses = []
    sfEntropies = []
    rotErrs = []
    tErrs = []

    # For errors recording
    testErrfile = open('./Model parameter/chess/scene/ransac_test_errors.txt', 'a')

    # Training set
    TrainingRound = 5000
    round = 0

    # For recording
    # E2E_FILE = './end2end/'
    # if not os.path.exists(E2E_FILE):
    #     os.mkdir(E2E_FILE)
    # OBJ_E2E = E2E_FILE + 'model_obj_e2e.pkl'
    # SCORE_E2E = E2E_FILE + 'model_score_e2e.pkl'
    PARAMETER = list(model_obj.parameters()) + list(model_score.parameters())
    OPTIMIZER = torch.optim.SGD(PARAMETER, lr=lrInitE2E, momentum=momentumE2E)
    # OPTIMZER_OBJ = torch.optim.SGD(model_obj.parameters(), lr=lrInitE2E, momentum=momentumE2E)
    # OPTIMZER_SCORE = torch.optim.SGD(model_score.parameters(), lr=lrInitE2E, momentum=momentumE2E)

    time_start = time.time()

    argcorrect = 0
    argcorrect_score = 0
    argcorrect_ransac = 0

    dsac_errfile = open('./Model parameter/chess/scene/dsac_err.txt', 'a')
    ransac_errfile = open('./Model parameter/chess/scene/ransac_err.txt', 'a')
    inlier_errfile = open('./Model parameter/chess/scene/inlier_err.txt', 'a')

    # Beginning
    # len(testDataset)
    for i in range(0, 2000):
        print("Processing test image no.", i + 1)
        testBGR = training.getBGR(i)
        testInfo = training.getInfo(i)

        # process frame (same function used in training, hence most of the variables below are not used here), see method documentation for parameter explanation
        hyps = []
        refHyps = []
        # imgPts = []
        # objPts = []
        imgIdx = []
        estObj = np.zeros((CNN_OBJ_PATCHSIZE, CNN_OBJ_PATCHSIZE, 3))
        sampling = np.zeros((CNN_OBJ_PATCHSIZE, CNN_OBJ_PATCHSIZE, 3))
        sampledPoints = []
        inlierMaps = []
        pixelIdxs = []

        correct, correct_score, correct_ransac,\
            rotErr, tErr, \
            rotErr_score, tErr_score,\
            rotErr_ransac, tErr_ransac = cnn.processImage(
                          testBGR,
                          testInfo,
                          model_obj,
                          model_score,
                          objHyps,
                          ptCount,
                          CamMat,
                          inlierThreshold2D,
                          refInlierCount,
                          refSteps,
                          refHyps,
                          sampledPoints,
                          estObj,
                          sampling,
                          inlierMaps,
                          pixelIdxs,
                          OPTIMIZER,
                          round)
        time_end = time.time()
        print('total cost time:', time_end - time_start)
        if correct:
            argcorrect += 1
            print('argcorrect:', argcorrect)
        if correct_score:
            argcorrect_score += 1
            print('argcorrect_score:', argcorrect_score)
        if correct_ransac:
            argcorrect_ransac += 1
            print('argcorrect_ransac:', argcorrect_ransac)

        np.savetxt(dsac_errfile, np.array([rotErr_score, tErr_score]))
        np.savetxt(ransac_errfile, np.array([rotErr_ransac, tErr_ransac]))
        np.savetxt(inlier_errfile, np.array([rotErr, tErr]))

    dsac_errfile.close()
    ransac_errfile.close()
    inlier_errfile.close()



if __name__ == '__main__':
    main()

