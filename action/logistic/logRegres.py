from numpy import *

def loadDataSet():
    dataMat = []; labelMat = [];
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(input):
    return 1.0/(1+exp(-input))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return array(weights)

def stocGradAscent0(dataMatrix,classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix,classLabels,numIter=500):
    m,n=shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error *dataMatrix[randIndex] #让randIndex位置的样本点J趋于局部最小
            del (list(dataIndex)[randIndex])
    return weights

def classifyVector(input,weights):
    prob = sigmoid(sum(input*weights))
    if prob > 0.5 : return 1.0
    else : return 0.0

def colicTest():
    trainingData = open('data/horseColicTraining.txt')
    testData = open('data/horseColicTest.txt')
    trainingSet = [];trainingLabels = []
    for line in trainingData.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr) #python矩阵（2维数组）
        trainingLabels.append(float(currentLine[21]))
    trainingWeights = stocGradAscent1(array(trainingSet),trainingLabels,100)

    #开始跑测试数据
    errorCount = 0;numTestVec = 0.0
    for line in testData.readlines():
        numTestVec += 1.0
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        if int(classifyVector(array(lineArr),trainingWeights)) != int(currentLine[21]):
            errorCount += 1

    errorRate = (float(errorCount)/numTestVec)
    print('this error rate of this test is: %f' % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum +=colicTest()
    print('after %d iterations the average error rate is: %f' %(numTests,errorSum/float(numTests)))


multiTest()