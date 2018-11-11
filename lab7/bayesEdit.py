# A copy of bayes.py, delete it when you see this

from numpy import *

# Create sample
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

# Create vocabulary of the data set
def createVocabList(dataSet):
    vocabSet = set([])
    for docment in dataSet:
        vocabSet = vocabSet | set(docment)
    return list(vocabSet)

# trainMat is the training matrix, each raw is a word vector
# trainCategory is the label of the training data, 0 for ham, 1 for spam
def train(trainMat, trainCategory):
    numTrain = len(trainMat)
    numWords = len(trainMat[0])
    pSpam = sum(trainCategory) / float(numTrain)
    p0Num = ones(numWords); p1Num = ones(numWords)  # Laplace Smoothing
    p0Denom = 2.0; p1Denom = 2.0    # Why 2.0?
    for i in range(numTrain):
        if trainCategory[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vec = log(p1Num / p1Denom)
    p0Vec = log(p0Num / p0Denom)
    return p0Vec, p1Vec, pSpam

# Return the class of vector vec2classfy
def classfy(vec2classfy, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2classfy*p1Vec) + log(pClass1)
    p0 = sum(vec2classfy*p0Vec) + log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

# Return the word vector of the input set with form ["", ""]
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("The word: %s is not in the vocabulary" % word)
    return returnVec

def createTrainingSet():
    fullTest, docList, classList = [], [], []
    for i in range(1, 26):
        with open('email/spam/%d.txt' % i, encoding="ISO-8859-1") as datafile:
            wordList = textParse(datafile.read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)
        with open('email/ham/%d.txt' % i, encoding="ISO-8859-1") as datafile:
            wordList = textParse(datafile.read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainSet = list(range(50)); testSet = []
    
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    
    trainMat, trainClass, testMat, testClass = [], [], [], []
    for docIndex in trainSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    for docIndex in testSet:
        testMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        testClass.append(classList[docIndex])
    return trainMat, trainClass, testMat, testClass, vocabList

'''
    25 ham, 25 spam. Select 10 as test-set, 40 as training set.
'''
def spamTest():
    trainMat, trainClass, testMat, testClass, vocabList = createTrainingSet()
    p0, p1, pSpam = train(array(trainMat), array(trainClass))
    errCount = 0
    for i in range(len(testMat)):
        if classfy(array(testMat[i]), p0, p1, pSpam) != testClass[i]:
            errCount += 1
            doc = [vocabList[i] for i in testMat[i] if i != 0]
            print("Classification error ", testClass[i], doc)

    print("The error rate is ", float(errCount)/len(testMat))

    return trainMat, trainClass, testMat, testClass

if __name__ == '__main__':
    spamTest()