from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(flattenMatrix, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 行数
    diffMat = tile(flattenMatrix, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 每一行向量相加
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        labelIndex = labels[sortedDistIndicies[i]]
        classCount[labelIndex] = classCount.get(labelIndex, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    flattenMatrix = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 填充矩阵
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        flattenMatrix[index, :] = listFromLine[0:3]  # 第1-3列赋给flattenMatrix的index行
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return flattenMatrix, classLabelVector


group, labels = createDataSet()
classify0([0, 0], group, labels, 3)

# 特征值标准化
def autoNorm(dataSet):
    minVals = dataSet.min(0) #print(a.min(0)) # axis=0; 每列的最小值
                             #print(a.min(1)) # axis=1；每行的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #shape 是获取矩阵的行和列数
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

matrix, labels = file2matrix('D:\MyConfiguration\szj46941\PycharmProjects\machine-learning\dataSet\datingTestSet.txt')

matrix,ranges,minVal=autoNorm(matrix)

# import matplotlib
# import matplotlib.pyplot as plot
#
# fig = plot.figure()
# subplot = fig.add_subplot(111)
# subplot.scatter(matrix[:, 1], matrix[:, 2], 10.0 * array(labels), 20.0 * array(labels))  # 第一个参数是尺寸，第二个是颜色
# plot.show()

def datingClassTest():
    hoRatios=0.20
    datingDataMatrix,datingLabels = file2matrix('D:\MyConfiguration\szj46941\PycharmProjects\machine-learning\dataSet\datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMatrix)
    m=normMat.shape[0]
    numTestVectors=int(m*hoRatios)
    errorCount=0.0
    for i in range(numTestVectors):
        classifierResult = classify0(normMat[i,:],normMat[numTestVectors:m,:]
        ,datingLabels[numTestVectors:m],5) #k=3
        print('the classifier came back with: %d,the real answer is: %d'
              %(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):errorCount +=1.0
    print('the total error rate is: %f'%(errorCount/float(numTestVectors)))
datingClassTest()
