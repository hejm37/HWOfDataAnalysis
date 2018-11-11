#-*- coding:utf-8 -*-
from numpy import *

def readOneDateSet(filename):
    dataSet, labels = [], []
    with open(filename) as dataFile:
        for line in dataFile.readlines():
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            dataSet.append(lineArr)
            labels.append(float(currLine[21]))
    return dataSet, labels

def loadColic():
    trainingSet, trainingLabels = readOneDateSet("horseColicTraining.txt")
    testSet, testLabels = readOneDateSet("horseColicTest.txt")
    return trainingSet, trainingLabels, testSet, testLabels

def addBiasTerm(dataSet):
    newDataSet = []
    for lineArr in dataSet:
        newDataSet.append([1] + lineArr)
    return newDataSet

# Simple data set for illustration
def loadDataSet():
    dataMat, labelMat = [], []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # bias term
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1.0+exp(-inX))

def gradAscent(dataMatIn, classLabels, maxIter=500):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    _, n = shape(dataMatrix)
    alpha = 0.001
    weights = ones((n, 1))
    for _ in range(maxIter):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights

def stocGradAscent0(dataMatrix, classLabels, maxIter=150):
    del maxIter     # delete the unused argument
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(dataMatrix[i]*weights)
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, maxIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(maxIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# Only for two dimensional data.
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=20, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest(gradFunc=stocGradAscent1):
    errorRate, _ = colicTest2(gradFunc)
    print("The error rate of this test is: %f" % errorRate)
    return errorRate

def colicTest2(gradFunc):
    trainingSet, trainingLabels, testSet, testLabels = loadColic()
    trainingSet = addBiasTerm(trainingSet); testSet = addBiasTerm(testSet)
    trainWeight = gradFunc(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = len(testSet)
    for i in range(numTestVec):
        if int(classifyVector(array(testSet[i]), trainWeight)) != testLabels[i]:
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    return errorRate, trainWeight

def multiTest(numTests=10, gradFunc=stocGradAscent1, printEach=True):
    errorSum = 0.0
    for _ in range(numTests):
        if printEach:
            error, _ = colicTest(gradFunc)
        else:
            error, _ = colicTest2(gradFunc)
        errorSum += error
    print("After %d iterations, the average error rate is: %f" % (numTests, errorSum/float(numTests)))

if __name__ == "__main__":
    multiTest()
