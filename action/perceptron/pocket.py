from numpy import *
import copy


def loadDataSet(path):
    dataMat = [];
    labelMat = [];
    lineArray = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        lineArray.append(lineArr)

    random.shuffle(lineArray)

    for l in lineArray:
        dataMat.append([1.0, float(l[0]), float(l[1]), float(l[2]), float(l[3])])
        labelMat.append(int(l[4]))
    return dataMat, labelMat


def pocket(dataMatIn, classLabels, cycle):
    l = len(classLabels)
    w = zeros((len(dataMatIn[0]), 1))
    error = 0
    best = w
    bestError = -1
    times = 0
    while cycle != times:
        times += 1
        # 随机找一个犯错的点
        errorIndex = -1
        for i in range(l):
            if sign(inner(w.transpose(), dataMatIn[i]))[0] != classLabels[i]:
                if errorIndex == -1: errorIndex = i
                break

        w += classLabels[errorIndex] * mat(dataMatIn[errorIndex]).transpose()

        # 计算更新后的w是否更好
        for i in range(l):
            if sign(inner(w.transpose(), dataMatIn[i]))[0] != classLabels[i]:
                error += 1

        if bestError == -1 or error <= bestError:
            bestError = error
            best = w

        error = 0

    return best


totalErrors = 0
for i in range(200):
    dataMat, labelMat = loadDataSet('pocketTraining.txt')
    best = pocket(dataMat, labelMat, 100)
    testMat, labelTestMat = loadDataSet('pocketTest.txt')
    l = len(labelTestMat)
    error = 0
    for i in range(l):
        if sign(inner(best.transpose(), testMat[i]))[0] != labelTestMat[i]:
            error+=1
    totalErrors+=error/l
print(totalErrors/200)