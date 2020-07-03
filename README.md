# DSAC Documentation (Python)

The document is complied for file structure, and some brief introduction and setup to DSAC code in Python version, which is revised and improved from the original  paper.

## Brief Introduction

Basically, the file `./code` includes all programs and there are some introduction to them partially.

* **train_obj.py:** With RGB images as the input, and the depth information as the ground truth, it trains a 3D coordinate regression CNN to predict each pixel's real scene coordinate in the image.

* **train_score.py:** Based on a trained 3D coordinate regression CNN, it continues to train a score regression CNN to evaluate the score for each hypothesis given their ground true camera poses.
* **test_for_all.py:** It can test component-wisely RANSAC, Vanilla RANSAC, and DSAC pipeline and outputs rotation and translation errors of the final hypothesis.
* **File_operation_chess.py:** It splits the Chess image sets into the training part and the test part.
* **File_operation_fire.py:** It splits the Fire image sets into the training part and the test part.

## Files Structure

The structure of our whole project is arranged as follows.

| Files Structure                                              |
| ------------------------------------------------------------ |
| **code**    *# it includes all codes needed* <br /> **Model parameters**   *# it includes all trained model parameters and the generated outputs*<br />          chess<br />                  *corner*<br />                  *scene*<br />          fire<br />                  *... ...*<br /> **training**   *# it includes all the information of the training set*<br />          chess <br />                  *depth_noseg*<br />                  *poses*<br />                  *rgb_noseg*<br />          fire<br />                  *... ...*<br /> **test**   *# it includes all the information of the test set*<br />          ... ...<br /> |

 

## Setup

Here are some guidance for code usage and revision to different training and test requirements.

* **Obtain the data sets**

  (1) Download the 7-scenes data set, namely `7scenes.zip`, and unpack it to get file `chess` and `fire`.

  (2) Run `./code/File_operation_chess.py` and `./code/File_operation_fire.py` to split the data into training set and test set. All required information will be transfered into file `training` and `test`.

* **Train 3D coordinate regression CNN**

  (1) The default is to train on `chess` without corner point selection.

  (2) Run `./code/train_obj.py`, and we can get the training loss information written in `training_loss_obj.txt` in the `./Model parameters/chess/`.

  (3) For switching to train on `fire`, we need to do some revision as follows.

  In the file`./code/train_obj.py`,

      Line 89	# Initialize training dataset
      		training = dataset.Dataset()
      		training.readFileNames(trainingSets[0]) # trainingSets[0] --> trainingSets[1]
      		training.SetObjID(1)
      
      Line 95	# Initialize test dataset
      		testing = dataset.Dataset()
      		testing.readFileNames(TestSets[0]) # TestSets[0] --> TestSets[1]
      		testing.SetObjID(1)
      
      ... ... 
      
      Line 99 # Construction Model
          	OBJ_NET = OBJ_CNN().to(DEVICE)
              OBJ_NET.load_state_dict(torch.load('./Model parameter/chess/scene/obj_model_init.pkl')) # ./Model parameter/chess/scene/obj_model_init.pkl --> ./Model parameter/fire/scene/obj_model_init.pkl
      
      ... ...
      
      Line 112 # For recording
          	 trainfile = open('./Model parameter/chess/scene/training_loss_obj.txt', 'a') # ./Model parameter/chess/scene/training_loss_obj.txt --> ./Model parameter/fire/scene/training_loss_obj.txt
          	 time_start = time.time()
  In file `./code/Model_obj.py`,

      Line 90 # num counter
          	num += 1
          	if not num % storeIntervalPre:
              	torch.save(model.state_dict(), './Model parameter/chess/scene/obj_model_init.pkl') # ./Model parameter/chess/scene/obj_model_init.pkl' --> ./Model parameter/fire/scene/obj_model_init.pkl'
          	if not num % lrInterval:
              	for param in optimizer.param_groups:
                  	param['lr'] *= 0.5
  (4) For switching to train with corner points selection, we need to find `./code/dataset.py` and use the function *rand_getRGBpix* in **Line 201** instead of the version in Line 235. Besides, any change involving `/chess/scene/`, we can replace it with`/chess/corner/`.

* **Train score regression CNN**

  (1) The default is to train on `chess` without corner point selection.

  (2)  Run`./code/train_score.py`.

  (3) Any similar switches to train on `fire` or with corner points selection, Please refer to the revision above.

* **Component-wise test** 

  (1) The default is to train on `chess` without corner point selection.

  (2) Run`./code/test_for_all.py`, and we can get outputs of  rotation and translation errors on three pipelines in `./Model parameter/chess/scene/`.

  (3)  For switching to test on `fire`, we need to do some revision as follows.

  In the file`./code/test_for_all.py`, we need to replace `trainingSets[0]` with `trainingSets[1]`  in **Line 41** and replace all `/chess/scene/` with `/fire/scene/` in **Line 45, 49, 53, 65, 88, 89, and 90**.

  In the file `./code/properties.py`, we need to replace this code in **Line 57**,

          file = np.fromfile('./7scenes/7scenes_chess/sensorTrans.dat',
                             dtype=float)
  with the code below.

          file = np.fromfile('./7scenes/7scenes_fire/sensorTrans.dat',
                             dtype=float)
  (4) For switching to test with the corner points selection, we need to replace *cnn_Sam.stochasticSubSamplewithoutC* with *cnn_Sam.stochasticSubSample* in **Line 316**.

  (5) For changing the size of hypotheses pool, we only need to change the parameter *ransacIterations* in `./code/properties.py` **Line 10**.

