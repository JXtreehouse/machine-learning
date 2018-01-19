# 朴素贝叶斯分类器

from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    pAbusive = sum(trainCategory) / float(numTrainDocs) #计算 p(c1)
    numWords = len(trainMatrix[0])
    p0Num = ones(numWords)#好的
    p0Denom = 2.0
    p1Num = ones(numWords)#坏的
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]#获取该类别下每个单词的总数
            p1Denom += sum(trainMatrix[i])#获取该类别下总词数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p0Vect = log(p0Num / p0Denom)#计算p(X|Y=Ck) 好的分类
    p1Vect = log(p1Num / p1Denom)#不好的分类
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1) #计算好的后验 第一项内部是log
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)#计算不好的后验
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(classVec))
    testEntry = ['i', 'love', 'garbage']
    thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['love', 'love', 'garbage', 'garbage']
    thisDoc = array(bagOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))


postingList, classVec = loadDataSet()
testingNB()

mySentence = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'


def textParse(bigString):
    import re
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(bigString)
    return [token.lower() for token in listOfTokens if len(token) > 0]


def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        #垃圾文章加入list
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        #正常文章加入list
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList) #找出不重复的值
    trainingSetIndex = list(range(50));
    testSetIndex = []
    for i in range(10):
        #加入测试集
        randIndex = int(random.uniform(0, len(trainingSetIndex)))
        testSetIndex.append(trainingSetIndex[randIndex])
        del (trainingSetIndex[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSetIndex:
        #准备训练数据
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSetIndex:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:' ,float(errorCount)/len(testSetIndex))


spamTest()
