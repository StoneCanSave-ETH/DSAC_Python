import torch
import cv2

import numpy as np
import skimage.color as color

import cnn
import dataset
import util

from Model_obj import OBJ_CNN
from skimage import io
from sklearn import preprocessing
from skimage.color import xyz2rgb
import matplotlib.pyplot as plt
import cv2

def main():

    # Basic set
    patchsize = 40

    # Load networks
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_obj = OBJ_CNN().to(DEVICE)
    model_obj.load_state_dict(torch.load('./Model parameter/fire/obj_model_init.pkl'))

    # Traning path setting
    dataDir = './'
    trainingDir = dataDir + 'training/'  # If u want to do experiment on test dataset, change it
    trainingSets = util.getSubPaths(trainingDir)

    # Initialize training datasets
    trainingDataset = dataset.Dataset()
    trainingDataset.readFileNames(trainingSets[0])
    trainingDataset.SetObjID(1)

    i = 523  # If u want to do experiment on other RGB image, change it
    imgBGR = trainingDataset.getBGR(i)

    # Do every pixel sample
    sample = [[idx % 640, idx // 640] for idx in range(480 * 640)]
    sample_with_patch = [idx for idx in sample if patchsize / 2 <= idx[0] < 640 - patchsize / 2 and \
                                                  patchsize / 2 <= idx[1] < 480 - patchsize / 2]
    sample_with_patch = np.array(sample_with_patch).reshape([440, 600, 2])
    pred_coord = cnn.getCoordImg(colorData=imgBGR,
                                 sampling=sample_with_patch,
                                 patchsize=patchsize,
                                 model=model_obj) / 1000.0
    return pred_coord


if __name__ == '__main__':
    pred_coord = main()

    x = pred_coord.shape[0]
    y = pred_coord.shape[1]
    z = pred_coord.shape[2]
    predObj_norm = np.zeros([x, y, z])
    predObj_normalized = np.zeros([x, y, z])
    mm = preprocessing.MinMaxScaler(feature_range=(0, 1))

    imgObj_x = pred_coord[:, :, 0]
    imgObj_y = pred_coord[:, :, 1]
    imgObj_z = pred_coord[:, :, 2]

    imgObj_x = imgObj_x.reshape(-1, 1)
    imgObj_y = imgObj_y.reshape(-1, 1)
    imgObj_z = imgObj_z.reshape(-1, 1)

    # idxs_x = np.argwhere(imgObj_x > 0.454)[:, 0]
    # idxs_x_min = np.argwhere(imgObj_x < -2.679)[:, 0]
    # idxs_y = np.argwhere(imgObj_y > 0.390)[:, 0]
    # idxs_y_min = np.argwhere(imgObj_y < -0.652)[:, 0]
    # idxs_z = np.argwhere(imgObj_z > 0.087)[:, 0]
    # idxs_z_min = np.argwhere(imgObj_z < -1.485)[:, 0]
    #
    # for idx in idxs_x:
    #     imgObj_x[idx] = 0.454
    #
    # for idx in idxs_x_min:
    #     imgObj_x[idx] = -2.679
    #
    # for idx in idxs_y:
    #     imgObj_y[idx] = 0.390
    #
    # for idx in idxs_y_min:
    #     imgObj_y[idx] = -0.652
    #
    # for idx in idxs_z:
    #     imgObj_z[idx] = 0.087
    #
    # for idx in idxs_z_min:
    #     imgObj_z[idx] = -1.485

    # idxs_x = np.argwhere(imgObj_x > 0.)[:, 0]
    # idxs_x_min = np.argwhere(imgObj_x < -2.712)[:, 0]
    # idxs_y = np.argwhere(imgObj_y > 0.356)[:, 0]
    # idxs_y_min = np.argwhere(imgObj_y < -0.876)[:, 0]
    # idxs_z = np.argwhere(imgObj_z > 0.041)[:, 0]
    # idxs_z_min = np.argwhere(imgObj_z < -1.219)[:, 0]
    #
    # for idx in idxs_x:
    #     imgObj_x[idx] = 0.
    #
    # for idx in idxs_x_min:
    #     imgObj_x[idx] = -2.712
    #
    # for idx in idxs_y:
    #     imgObj_y[idx] = 0.356
    #
    # for idx in idxs_y_min:
    #     imgObj_y[idx] = -0.876
    #
    # for idx in idxs_z:
    #     imgObj_z[idx] = 0.041
    #
    # for idx in idxs_z_min:
    #     imgObj_z[idx] = -1.219

    # for 523
    idxs_x = np.argwhere(imgObj_x > 0.)[:, 0]
    idxs_x_min = np.argwhere(imgObj_x < -2.924)[:, 0]
    idxs_y = np.argwhere(imgObj_y > 0.)[:, 0]
    idxs_y_min = np.argwhere(imgObj_y < -1.457)[:, 0]
    idxs_z = np.argwhere(imgObj_z > 0.)[:, 0]
    idxs_z_min = np.argwhere(imgObj_z < -1.479)[:, 0]

    for idx in idxs_x:
        imgObj_x[idx] = 0.

    for idx in idxs_x_min:
        imgObj_x[idx] = -2.924

    for idx in idxs_y:
        imgObj_y[idx] = 0.

    for idx in idxs_y_min:
        imgObj_y[idx] = -1.457

    for idx in idxs_z:
        imgObj_z[idx] = 0.

    for idx in idxs_z_min:
        imgObj_z[idx] = -1.479

    # for 1980
    # idxs_x = np.argwhere(imgObj_x > 0.)[:, 0]
    # idxs_x_min = np.argwhere(imgObj_x < -2.889)[:, 0]
    # idxs_y = np.argwhere(imgObj_y > 0.)[:, 0]
    # idxs_y_min = np.argwhere(imgObj_y < -1.477)[:, 0]
    # idxs_z = np.argwhere(imgObj_z > 0.)[:, 0]
    # idxs_z_min = np.argwhere(imgObj_z < -2.049)[:, 0]
    #
    # for idx in idxs_x:
    #     imgObj_x[idx] = 0.
    #
    # for idx in idxs_x_min:
    #     imgObj_x[idx] = -2.889
    #
    # for idx in idxs_y:
    #     imgObj_y[idx] = 0.
    #
    # for idx in idxs_y_min:
    #     imgObj_y[idx] = -1.477
    #
    # for idx in idxs_z:
    #     imgObj_z[idx] = 0.
    #
    # for idx in idxs_z_min:
    #     imgObj_z[idx] = -2.049


    pred_coord[:, :, 0] = imgObj_x.reshape(x, y)
    pred_coord[:, :, 1] = imgObj_y.reshape(x, y)
    pred_coord[:, :, 2] = imgObj_z.reshape(x, y)

    for i in range(pred_coord.shape[2]):
        imgObj_xyz = pred_coord[:, :, i]
        imgObj_xyz = imgObj_xyz.reshape(-1, 1)
        # imgObj_xyz_scale = preprocessing.scale(imgObj_xyz)
        # imgObj_xyz_normalized = preprocessing.normalize(imgObj_xyz, norm='l2')
        imgObj_xyz_norm = mm.fit_transform(imgObj_xyz)
        predObj_norm[:, :, i] = imgObj_xyz_norm.reshape(x, y)
        # predObj_normalized[:, :, i] = imgObj_xyz_normalized.reshape(x, y)
        # origin_data = mm.inverse_transform(imgObj_xyz_norm)

    imgRGB = xyz2rgb(predObj_norm)
    # show RGB array as a picture
    plt.imshow(imgRGB)
    plt.savefig('./Figure/523_predict.jpg')
    plt.show()


    # imgObj_z_norm = imgObj_z_norm.astype(np.uint8)
    # imgObj_z = imgObj_z.astype(np.uint8)
    # imgObj_z_normalized = imgObj_z_normalized.astype(np.uint8)
    # img = cv2.applyColorMap(imgObj_z_normalized, colormap=4)
    # cv2.imshow('a', img)
    # cv2.waitKey(0)

    # pred_coordz = predObj_norm[:, :, 2]
    # imgRGB = cv2.cvtColor(pred_coord, cv2.COLOR_XYZ2RGB)
    # plt.imshow(imgRGB)
    # plt.show()

    print('okkkkk')
    print('okkkkk')
