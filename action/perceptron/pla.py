from numpy import *


def loadDataSet():
    dataMat = [];
    labelMat = [];
    lineArray = []
    fr = open('data.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        lineArray.append(lineArr)

    random.shuffle(lineArray)

    for l in lineArray:
        dataMat.append([1.0, float(l[0]), float(l[1]), float(l[2]), float(l[3])])
        labelMat.append(int(l[4]))
    return dataMat, labelMat


def pla(dataMatIn, classLabels):
    l = len(classLabels)

    w = zeros((len(dataMatIn[0]), 1))
    count = 0
    times = 0
    i = 0
    while l != count:
        if sign(inner(w.transpose() , dataMatIn[i]))[0] == classLabels[i]:
            count += 1
        else:
            count = 0
            w += 0.5*classLabels[i] * mat(dataMatIn[i]).transpose()
            times +=1
        if i == l-1:
            i=0
        else:
            i+=1
    return w,times

totalTimes = 0
for i in range(2000):
    dataMat, labelMat = loadDataSet()
    w,times = pla(dataMat,labelMat)
    totalTimes+=times
print(totalTimes/2000)