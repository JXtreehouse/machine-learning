from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=999999):
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    # 若rowvar = 0，说明传入的数据一行代表一个样本，若非0，说明传入的数据一列代表一个样本
    covMat = cov(meanRemoved,rowvar = 0)
    eigVals ,eigVects = linalg.eig(covMat)
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowmat = meanRemoved * redEigVects
    recmat = lowmat * redEigVects.T + meanVals
    return lowmat,recmat

mat = loadDataSet('testSet.txt')
lowmat , recmat = pca(mat,1)
print(1)