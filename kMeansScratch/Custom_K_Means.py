"""
Implementation of K-Means algorithm from scratch

Author: Anshuman Gaharwar
E-mail: agaharwar@knights.ucf.edu
"""
import copy
import math
import random
import numpy as np
import pandas as pd
from numpy.linalg import norm
from collections import Counter
from sklearn.metrics import accuracy_score


class DistanceFactory(object):
    def __init__(self, distanceType):
        assert distanceType in ["EUC", "COS", "JAC"]
        self.calculateObj = getattr(self, "_calculate{}".format(distanceType))

    def calculate(self, X, Y):
        return self.calculateObj(X, Y)

    def _calculateEUC(self, X, Y):
        return np.sum((X - Y) ** 2)

    def _calculateCOS(self, X, Y):
        return 1 - np.dot(X, Y) / (norm(X) * norm(Y))

    def _calculateJAC(self, X, Y):
        container = np.array([[X], [Y]])
        return 1 - np.sum(np.min(container, axis=0)) / np.sum(np.max(container, axis=0))


class CustomKMeans(object):
    def __init__(self, k=1, maxIter=300, centroids=[], distanceMetric="COS", stopCriteria="maxIter", threshold=0.001):
        self.iterNum = 0
        self.stopCriteria = self.setStopCriteria(stopCriteria)
        self.distanceObj = DistanceFactory(distanceMetric)
        self.k = k
        self.maxIter = maxIter
        self.clusterIds = np.array([i for i in range(k)])
        self.centroids = centroids
        self.inertia = 0
        self.prevInertia = math.inf
        self.c2cMap = {}
        self.prevCentroids = None
        self.centroidChangeThresh = threshold

    def setStopCriteria(self, stopCriteria):
        stopMap = {
            "maxIter": self.checkIter,
            "centroid": self.checkCentroid,
            "sse": self.checkSSE
        }
        method = stopMap.get(stopCriteria)
        assert method, "Invalid stop criteria passed: {}".format(stopCriteria)
        return method

    def checkIter(self):
        if self.iterNum < self.maxIter:
            return True
        print("Stopping after iteration {} as maxIter limit reached".format(self.iterNum))
        return False

    def checkCentroid(self):
        if self.iterNum == 0: return True
        change = False
        # print("ITER: ", self.iterNum)
        for cIdx in self.clusterIds:
            val = self.distanceObj.calculate(self.centroids[cIdx], self.prevCentroids[cIdx])
            change = val > self.centroidChangeThresh
        if not change:
            print("Stopping after iteration {} as centroids did not change".format(self.iterNum))
            return False
        return True

    def checkSSE(self):
        if self.iterNum == 0: return True
        if self.inertia >= self.prevInertia:
            print("Stopping after iteration {} as Inertia/SSE increased across iteration".format(self.iterNum))
            return False
        return True

    def initCentroids(self, nRow):
        centroidIdxs = random.sample(range(nRow), self.k)
        centContainer = []
        for centIdx in centroidIdxs:
            centContainer.append(self.data[centIdx])
        self.centroids = np.asarray(centContainer)

    def assignCluster(self, dataPoint):
        minIdx = 0
        minVal = math.inf
        for cIdx in self.clusterIds:
            distVal = self.distanceObj.calculate(dataPoint, self.centroids[cIdx])
            if distVal < minVal:
                minIdx = cIdx
                minVal = distVal
        return minIdx

    def updateCentroids(self):
        for cIdx in self.clusterIds:
            targetClusterIdxs = np.where(self.assignedClusters == cIdx)
            if targetClusterIdxs[0].size == 0: continue
            self.centroids[cIdx] = np.average(self.data[targetClusterIdxs], axis=0)

    def generateMap(self):
        for cIdx in self.clusterIds:
            targetClusterIdxs = np.where(self.assignedClusters == cIdx)
            clustCount = Counter(self.labels[targetClusterIdxs])
            tempList = [[item, clustCount[item]] for item in clustCount]
            tempList.sort(key=lambda x: x[1], reverse=True)
            labelCluster, _ = tempList[0]
            self.c2cMap[cIdx] = labelCluster

    def mapConversion(self, pred):
        for rIdx in range(len(pred)):
            pred[rIdx][0] = self.c2cMap.get(pred[rIdx][0])
        return pred

    def getInertia(self):
        inertia = 0
        for custId in self.clusterIds:
            inertia += np.sum((self.data[np.where(self.assignedClusters == custId)] - self.centroids[custId]) ** 2)
        return inertia

    def fit(self, data, labels):
        # init variable
        self.data = data
        self.labels = labels
        nRow, nCol = self.data.shape
        self.plotData = {}
        random.seed(7)

        # Choosing random k samples from sample set for centroids
        if not self.centroids:
            self.initCentroids(nRow)

        # Assign clusters to sample set. Init to clusterId = 0
        self.assignedClusters = np.zeros(nRow)

        # Calculate Inertia/SSE before fitting
        self.inertia = self.getInertia()

        # Loop till stop criteria is not met
        while self.stopCriteria():
            self.iterNum += 1
            self.prevInertia = self.inertia

            # Assign proper cluster to each sample point
            for rIdx in range(nRow):
                self.assignedClusters[rIdx] = self.assignCluster(self.data[rIdx])

            self.prevCentroids = copy.deepcopy(self.centroids)

            # Calculate new centroids based on clusters assigned to sample points
            self.updateCentroids()
            self.inertia = self.getInertia()

        # If labels were passed for training. Create map between unsupervised clustering IDs to actual labelIds
        self.generateMap()

    def predict(self, testData):
        nRow, nCol = testData.shape
        predictions = np.zeros((nRow, 1))
        for idx in range(nRow):
            predictions[idx][0] = self.assignCluster(testData[idx])
        if self.c2cMap:
            predictions = self.mapConversion(predictions)
        return predictions


if __name__ == "__main__":
    # Sample Injector to run from command line
    data = pd.read_csv("dataset\data.csv", header=None)
    labels = pd.read_csv("dataset\label.csv", header=None)
    kLabels = len(labels[0].unique())
    data = data.to_numpy(dtype="float")
    labels = labels.to_numpy(dtype="float").flatten()
    dist = ["EUC", "COS", "JAC"]
    print("=============== SUMMARY ================\n")
    for d in dist:
        print("-" * 50)
        model = CustomKMeans(k=kLabels, maxIter=50, distanceMetric=d, stopCriteria="centroid")
        print("Fitting model for distance: {}".format(d))
        model.fit(data, labels)
        predictions = model.predict(data)
        acc = accuracy_score(labels, predictions) * 100
        print("SSE/Inertia post fitting: {}".format(model.inertia))
        print("Accuracy for distance [{}]: {}%".format(d, acc))
