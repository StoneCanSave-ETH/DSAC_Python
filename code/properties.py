import cv2
import numpy as np
import generic_io


class GlobalProperties:
    # pose parameters
    def __init__(self,
                 randomDraw=1,
                 ransacIterations=256,
                 ransacRefinementIterations=8,
                 ransacBatchSize=100,
                 ransacSubSample=0.01,
                 ransacInlierThreshold2D=5,
                 ransacInlierThreshold3D=100,

                 # dataset parameters
                 rawData=True,
                 focalLength=525,
                 xShift=0,
                 yShift=0,
                 secondaryFocalLength=585,
                 rawXShift=0,
                 rawYShift=0,
                 imageWidth=640,
                 imageHeight=480,
                 objScript="train_obj.lua",
                 scoreScript="train_score.lua",
                 objModel="obj_model_init.net",
                 scoreModel="score_model_init.net",
                 config="default",
                 sensorTrans=np.eye(4)):
        self.randomDraw = randomDraw
        self.ransacIterations = ransacIterations
        self.ransacRefinementIterations = ransacRefinementIterations
        self.ransacBatchSize = ransacBatchSize
        self.ransacSubSample = ransacSubSample
        self.ransacInlierThreshold2D = ransacInlierThreshold2D
        self.ransacInlierThreshold3D = ransacInlierThreshold3D
        self.rawData = rawData
        self.focalLength = focalLength
        self.xShift = xShift
        self.yShift = yShift
        self.secondaryFocalLength = secondaryFocalLength
        self.rawXShift = rawXShift
        self.rawYShift = rawYShift
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.objScript = objScript
        self.scoreScript = scoreScript
        self.objModel = objModel
        self.scoreModel = scoreModel
        self.config = config
        '''
        Note: If we train on fire, change "7scenes_chess" to "7scenes_fire"
        '''
        file = np.fromfile('./7scenes/7scenes_chess/sensorTrans.dat',
                           dtype=float)
        self.sensorTrans = np.delete(file, 0).reshape(4, 4)

    def readArgument(self, argv):
        argc = argv.size()
        i = 0
        while i < argc:
            s = argv[i]

            if s == '-iw':
                i = i + 1
                self.imageWidth = argv[i]
                print("image width: ", self.imageWidth)
                continue

            if s == "-ih":
                i = i + 1
                self.imageHeight = argv[i]
                print("image height: ", self.imageHeight)
                continue

            if s == "-fl":
                i = i + 1
                self.focalLength = argv[i]
                print("focal length: ", self.focalLength)
                continue

            if s == "-xs":
                i = i + 1
                self.xShift = argv[i]
                print("x shift: ", self.xShift)
                continue

            if s == "-ys":
                i = i + 1
                self.yShift = argv[i]
                print("y shift: ", self.yShift)
                continue

            if s == "-rd":
                i = i + 1
                self.rawData = argv[i]
                print("raw data (rescale rgb): ", self.rawData)
                continue

            if s == "-sfl":
                i = i + 1
                self.secondaryFocalLength = argv[i]
                print("secondary focal length: ", self.secondaryFocalLength)
                continue

            if s == "-rxs":
                i = i + 1
                self.rawXShift = argv[i]
                print("raw x shift: ", self.rawXShift)
                continue

            if s == "-rys":
                i = i + 1
                self.rawYShift = argv[i]
                print("raw y shift: ", self.rawYShift)
                continue

            if s == "-rdraw":
                i = i + 1
                self.randomDraw = argv[i]
                print("random draw: ", self.randomDraw)
                continue

            if s == "-oscript":
                i = i + 1
                self.objScript = argv[i]
                print("object script: ", self.objScript)
                continue

            if s == "-sscript":
                i = i + 1
                self.scoreScript = argv[i]
                print("score script: ", self.scoreScript)
                continue

            if s == "-omodel":
                i = i + 1
                self.objModel = argv[i]
                print("object model: ", self.objModel)
                continue

            if s == "-smodel":
                i = i + 1
                self.scoreModel = argv[i]
                print("score model: ", self.scoreModel)
                continue

            if s == "-rT2D":
                i = i + 1
                self.ransacInlierThreshold2D = argv[i]
                print("ransac inlier threshold: ", self.ransacInlierThreshold2D)
                continue

            if s == "-rT3D":
                i = i + 1
                self.ransacInlierThreshold3D = argv[i]
                print("ransac inlier threshold: ", self.ransacInlierThreshold3D)
                continue

            if s == "-rRI":
                i = i + 1
                self.ransacRefinementIterations = argv[i]
                print("ransac iterations (refinement): ", self.ransacRefinementIterations)
                continue

            if s == "-rI":
                i = i + 1
                self.ransacIterations = argv[i]
                print("ransac iterations: ", self.ransacIterations)
                continue

            if s == "-rB":
                i = i + 1
                self.ransacBatchSize = argv[i]
                print("ransac batch size: ", self.ransacBatchSize)
                continue

            if s == "-rSS":
                i = i + 1
                self.ransacSubSample = argv[i]
                print("ransac refinement gradient sub sampling: ", self.ransacSubSample)
                continue
            else:
                print("unkown argument: ", argv[i])
                return False

        i = i + 1

    def parseConfig(self):
        configfile = self.config + ".config"
        f = open('configfile', 'r')
        argVec = []
        while True:
            line = f.readline()
            if not line:
                break

            if line.len == 0:  # empty line
                continue

            if line[0] == '#':  # comment line
                continue

            tokens = str.split(line)
            if not tokens:
                continue
            argVec.append('-' + tokens[0])
            argVec.append(tokens[1])

        self.readArgument(argVec)

    def getCamMat(self):
        centerX = self.imageWidth/2.0 + self.xShift
        centerY = self.imageHeight/2.0 + self.yShift
        f = self.focalLength
        camMat = np.array([
            [f, 0, centerX],
            [0, f, centerY],
            [0, 0, 1]
        ])
        return camMat

