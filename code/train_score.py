import torch
from torchvision import transforms
import time

import Model_score
from Model_score import SCORE_CNN
from Model_obj import OBJ_CNN     # obj CNN construction
from Customized_Datasets import SCORE_DATASET
import util
import dataset


if __name__ == '__main__':

    # Parameter setting
    trainingImages = 100
    trainingHyps = 16
    trainingRounds = 120
    objTemperature = 10
    objBatchSize = 32
    lrInitPre = 0.0001
    CNN_OBJ_PATCHSIZE = 40
    CNN_RGB_PATCHSIZE = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training path setting
    dataDir = './'
    trainingDir = dataDir + 'training/'
    trainingSets = util.getSubPaths(trainingDir)

    '''
        If we wanna use fire scene, change the following:
        training.readFileNames(trainingSets[0]) -> training.readFileNames(trainingSets[1])
        and change line "RGB_NET.load_state_dict(torch.load('./Model parameter/obj_model_init.pkl', map_location=DEVICE))"
        -> RGB_NET.load_state_dict(torch.load('./Model parameter/fire/obj_model_init.pkl', map_location=DEVICE))
    '''


    # Load RGB CNN's parameters
    RGB_NET = OBJ_CNN().to(DEVICE)
    RGB_NET.load_state_dict(torch.load('./Model parameter/chess/scene/obj_model_init.pkl', map_location=DEVICE))

    # Dataset process
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Initialize training dataset
    trainingDataset = dataset.Dataset()
    trainingDataset.readFileNames(trainingSets[0])
    trainingDataset.SetObjID(1)

    # Construct SCORE CNN network
    SCORE_NET = SCORE_CNN().to(DEVICE)
    SCORE_NET.apply(Model_score.weight_init)
    # SCORE_NET.load_state_dict(torch.load('./Model parameter/fire/score_model_init.pkl', map_location=DEVICE))

    # Training parameter
    trainCounter = 0
    round = 0
    optimizer = torch.optim.Adam(SCORE_NET.parameters(), lrInitPre)
    lossfunction = torch.nn.PairwiseDistance(p=1)

    # For recording
    loss_list = []
    time_start = time.time()

    # Iteration for training
    while round <= trainingRounds:

        # Print round info
        round += 1
        print('Starting Round:', round)

        # Load datasets and train
        trainingDataset.randFiles(trainingImages=trainingImages, trainingPatches=trainingHyps)
        TrainData = SCORE_DATASET(trainingDataset, objInputSize=CNN_OBJ_PATCHSIZE, rgbInputSize=CNN_RGB_PATCHSIZE,
                                  model=RGB_NET, temperature=objTemperature, transform=transform,
                                  trainingImages=trainingImages, trainingPatches=trainingHyps)
        train_loader = torch.utils.data.DataLoader(TrainData, batch_size=objBatchSize, shuffle=True)

        # Return loss
        loss, trainCounter = Model_score.train(model=SCORE_NET, train_loader=train_loader,
                                               lossfunc=lossfunction, optimizer=optimizer, device=DEVICE, num=trainCounter)

        # Recording
        loss_list.append(loss)
        time_end = time.time()
        print(loss)
        print('Time Cost:', time_end - time_start)





