import numpy as np
import torch
import time
from torchvision import transforms

import dataset
import util
from Model_obj import OBJ_CNN
import Model_obj
from Customized_Datasets import RGB_DATASET


def assembleData(dataset, inputSize, transform, trainingPatches):

    length = len(dataset.rand_bgrFiles)
    for i in range(length):
        if not i % trainingPatches:
            selected_imgObj = dataset.rand_getObj(i)
            selected_imgBGR = dataset.rand_getBGR(i)
            data, label = dataset.rand_subsample(imgBGR=selected_imgBGR, imgObj=selected_imgObj, inputSize=inputSize)
            # For data
            img = transform(data).unsqueeze(0)

            # For label
            label_tensor = torch.from_numpy(label)
            label_tensor = label_tensor.unsqueeze(0)
        else:
            imgObj = selected_imgObj
            imgBGR = selected_imgBGR
            data, label = dataset.rand_subsample(imgBGR=imgBGR, imgObj=imgObj, inputSize=inputSize)

            # For data
            img = transform(data).unsqueeze(0)

            # For label
            label_tensor = torch.from_numpy(label)
            label_tensor = label_tensor.unsqueeze(0)
        if not i:
            FullData = img
            FullLabel = label_tensor
        else:
            FullData = torch.cat((FullData, img), 0)
            FullLabel = torch.cat((FullLabel, label_tensor), 0)

    return FullData, FullLabel


def assembleBatch(permutation, data, label):
    batchData = data[permutation, :, :, :]
    batchLabels = label[permutation, :]
    return batchData, batchLabels


if __name__ == '__main__':

    # Parameters setting
    inputSize = 42
    channels = 3
    trainingLimit = 320000
    trainingImages = 100
    trainingPatches = 512
    BATCHSIZE = 64
    lrInitPre = 0.0001
    storeIntervalPre = 1000
    lrInterval = 50000
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training path setting
    dataDir = './'
    trainingDir = dataDir + 'training/'
    trainingSets = util.getSubPaths(trainingDir)

    # Test path setting
    dataDir = './'
    TestDir = dataDir + 'test/'
    TestSets = util.getSubPaths(TestDir)

    # transform for data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    '''
    If we wanna use fire scene, change the following:
    training.readFileNames(trainingSets[0]) -> training.readFileNames(trainingSets[1])
    and comment "OBJ_NET.load_state_dict(torch.load('./Model parameter/obj_model_init.pkl'))"
    '''
    # Initialize training dataset
    training = dataset.Dataset()
    training.readFileNames(trainingSets[0])
    training.SetObjID(1)

    # Initialize test dataset
    testing = dataset.Dataset()
    testing.readFileNames(TestSets[0])
    testing.SetObjID(1)

    # Construction Model
    OBJ_NET = OBJ_CNN().to(DEVICE)
    OBJ_NET.load_state_dict(torch.load('./Model parameter/chess/scene/obj_model_init.pkl'))
    OBJ_NET.apply(Model_obj.weight_init())

    # Training
    trainCounter = 0
    round = 0
    StoreCounter = 0
    optimizer = torch.optim.Adam(OBJ_NET.parameters(), lr=lrInitPre)
    len_list = [i for i in range(trainingImages * trainingPatches)]
    lengths = trainingImages * trainingPatches

    # For recording
    trainfile = open('./Model parameter/chess/scene/training_loss_obj.txt', 'a')
    time_start = time.time()

    # Validation parameter
    ValiCounter = 0
    ValiLimit = 40000

    # Iteration for training
    while trainCounter <= trainingLimit:
        round += 1
        print('Starting Round:', round)

        training.randFiles(trainingImages=trainingImages, trainingPatches=trainingPatches)
        traindata = RGB_DATASET(dataset=training, inputSize=inputSize, transform=transform,
                                trainingImages=trainingImages, trainingPatches=trainingPatches)
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=BATCHSIZE, shuffle=True, num_workers=8)
        Loss, trainCounter = Model_obj.train(model=OBJ_NET, trainloader=train_loader, optimizer=optimizer,
                                             num=trainCounter, device=DEVICE)
        # datas, labels = assembleData(dataset=training, inputSize=inputSize, transform=transform,
        #                                    trainingPatches=trainingPatches)
        # random.shuffle(len_list)
        # for i in range(int(lengths / BATCHSIZE)):
        #     trainCounter += 1
        #     Permutation = len_list[i * BATCHSIZE:(i + 1) * BATCHSIZE]
        #     BatchData, BatchLabel = assembleBatch(Permutation, datas, labels)
        #     loss = Model_obj.train(model=OBJ_NET, batchdata=BatchData, batchlabel=BatchLabel,
        #                               optimizer=optimizer, device=DEVICE, num=trainCounter)
        # testing.randFiles(trainingImages=trainingImages, trainingPatches=trainingPatches)
        # testdata = RGB_DATASET(dataset=testing, inputSize=inputSize, transform=transform,
        #                         trainingImages=trainingImages, trainingPatches=trainingPatches)
        # test_loader = torch.utils.data.DataLoader(testdata, batch_size=BATCHSIZE, shuffle=True, num_workers=8)
        # Loss, trainCounter = Model_obj.test(model=OBJ_NET, test_loader=test_loader,
        #                                      num=trainCounter, device=DEVICE)

            # For testing
            # testing.randFiles(trainingImages=trainingImages, trainingPatches=trainingPatches)
            # TestData = RGB_DATASET(testing, inputSize, transform,
            #                                            trainingImages=trainingImages, trainingPatches=trainingPatches)
            # test_loader = torch.utils.data.DataLoader(TestData, batch_size=BATCHSIZE, shuffle=True, num_workers=8)
            # loss, ValiCounter = Model_obj.test(OBJ_NET, test_loader, DEVICE, ValiCounter)

        # For recording
        np.savetxt(trainfile, np.array([Loss]))
        time_end = time.time()
        print('Time Cost:', time_end - time_start)
    trainfile.close()
















