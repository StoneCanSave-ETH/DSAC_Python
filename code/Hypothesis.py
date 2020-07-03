import cv2
import numpy as np
from numpy.linalg import inv
import math

import TYPE


class Hypothesis(object):

    def __init__(self, rotation=np.eye(3), translation=np.zeros(3), invRotation=np.eye(3), points = []):
        self.rotation = rotation
        self.translation = translation
        self.invRotation = invRotation
        self.points = points

    def RotandTrans(self, rot, trans):
        self.rotation = rot
        self.translation = trans
        self.invRotation = inv(rot)
    # def RotandTrans(cls, rot = np.eye(3), trans = np.zeros(3)):
    #     return cls(rot, trans, inv(rot))

    def TransformationMatrix(self, transformation):
        self.rotation = transformation[:3, :3]
        self.translation = transformation[:3, 3]
        self.invRotation = inv(transformation[:3, :3])
    # def TransformationMatrix(cls, transformation = np.eye(4)):
    #     return cls(transformation[:3, :3], transformation[:3, 3], inv(transformation[:3, :3]))

    def RodvecandTrans(self, rodandtrans):
        rodvec = rodandtrans[:3]
        length = np.linalg.norm(rodvec)
        trans = rodandtrans[3:6]
        if length > 1e-5:
            rot = cv2.Rodrigues(rodvec)[0]
            self.rotation = rot
            self.translation = trans
            self.invRotation = inv(rot)
        else:
            self.rotation = np.eye(3)
            self.translation = trans
            self.invRotation = np.eye(3)

    def Info(self, info=TYPE.info_t()):
        self.rotation = info.rotation
        self.translation = info.center*1e3
        self.invRotation = inv(self.rotation)

    def setRotation(self, rot = np.eye(3)):
        self.rotation = rot
        self.invRotation = inv(rot)

    def setTranslation(self, trans = np.zeros(3)):
        self.translation = trans

    def transform(self, point = np.zeros(3), isNormal = False):
        tp = np.dot(self.rotation, point)
        if not isNormal:
            return tp+self.translation
        else:
            return tp

    def invTransform(self, point):
        point_tp = point - self.translation
        tp = np.dot(self.invRotation, point_tp)
        return tp

    def getTranslation(self):
        return self.translation

    def getRotation(self):
        return self.rotation

    def getInvRotation(self):
        return self.invRotation

    def getTransformation(self):
        result = np.zeros([4, 4])
        result[:3, :3] = self.rotation
        result[:3, 3] = self.translation
        result[3, 3] = 1
        return result

    def getInv(self):
        h = Hypothesis()
        h.TransformationMatrix(inv(self.getTransformation()))
        return h

    def __mul__(self, other):
        h = Hypothesis()
        transformation_h = np.dot(self.getTransformation(), other.getTransformation())
        h.TransformationMatrix(transformation_h)
        return h

    def __truediv__(self, other):
        h = Hypothesis()
        transformation_h = np.dot(self.getTransformation(), inv(other.getTransformation()))
        h.TransformationMatrix(transformation_h)
        return h

    def getRodriguesVector(self):
        result = cv2.Rodrigues(self.rotation)[0].reshape([-1])
        return result

    def getRodVecAndTrans(self):
        result = np.zeros(6)
        result[3:6] = self.getTranslation().reshape([-1])
        result[:3] = self.getRodriguesVector()
        return result

    def calcRigidBodyTransform(self, *argv):
        if len(argv) == 3:
            coV = argv[0]
            pointsA = argv[1]
            pointsB = argv[2]
            u, s, vh = cv2.SVDecomp(coV)
            sign = np.linalg.det(np.dot(np.transpose(vh), s))
            dm = np.eye(3)
            dm[2, 2] = sign
            resultRot = np.dot(np.dot(np.transpose(vh), dm), np.transpose(s))
            temp = pointsB-np.dot(resultRot, pointsA)
            return [resultRot, temp]
        else:
            points = argv[0]
            length = len(points)
            cA = np.zeros(3)
            cB = np.zeros(3)
            pointsA = np.zeros([3, length])
            pointsB = np.zeros([3, length])
            for i in range(length):
                cA += points[i, 0]
                cB += points[i, 1]
            cA = cA/length
            cB = cB/length
            for i in range(length):
                pointsA[0, i] = points[i, 0, 0] - cA[0]
                pointsA[1, i] = points[i, 0, 1] - cA[1]
                pointsA[2, i] = points[i, 0, 2] - cA[2]
                pointsB[0, i] = points[i, 1, 0] - cB[0]
                pointsB[1, i] = points[i, 1, 1] - cB[1]
                pointsB[2, i] = points[i, 1, 2] - cB[2]
            a = np.dot(pointsA, np.transpose(pointsB))
            return self.calcRigidBodyTransform(a, cA, cB)

    def refine(self, *argv):
        if len(argv) == 3:
            coV = argv[0]
            pointsA = argv[1]
            pointsB = argv[2]
            rotation = self.calcRigidBodyTransform(coV, pointsA, pointsB)[0]
            translation = self.calcRigidBodyTransform(coV, pointsA, pointsB)[1]
            invRotation = inv(self.rotation)
            return [rotation, translation, invRotation]
        else:
            point = argv[0]
            self.points.append(point)
            [rot, trans] = self.calcRigidBodyTransform(self.points)
            invrot = inv(rot)
            return [rot, trans, invrot]

    def Points(self, points):
        [rot, trans, invrot] = self.refine(points)
        self.rotation = rot
        self.translation = trans
        self.invRotation = invrot

    def load(self):
        print(self.rotation)
        print(self.points)

    def calcAngularDistance(self, h):
        rotDiff = np.dot(self.getRotation(), h.getInvRotation())
        trace = np.trace(rotDiff)
        trace = min(3.0, max(-1.0, trace))
        return 180*math.acos((trace - 1.0)/2.0)/math.pi


# c = np.eye(3)
# h1 = Hypothesis()
# h1.RotandTrans(a, b)
# h2 = Hypothesis()
# h2.RotandTrans(c, b)
# h3 = h2 / h1
# print(type(h3))
# print(h3.rotation)


