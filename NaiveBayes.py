# coding=utf-8
# Author:Summer
# Date:2018-12-28
# Email:xyyang1994@gmail.com
# Reference: 李航《统计学习方法》Chapter4

'''
Dataset: Mnist
Training set size: 60000 observations, 1 label + 784 features
Testset size: 10000（actual: 200)
------------------------------
Results：
    Test accuracy: 84.33%
    Running time: 91s
'''

import numpy as np
import time


def loadData(fileName):
    '''
    load the file
    fileName: the path to data
    return: dataset and labels
    '''

    dataArr = []
    labelArr = []
    # read the file
    fr = open(fileName)
    for line in fr.readlines():
        # strip: Returns a copy of the string with the leading and trailing characters removed.(default: whitespace)
        curLine = line.strip().split(',')
        # Binarization: if num > 128, 1 else 0. for calculation convenience
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])
        labelArr.append(int(curLine[0]))

    return dataArr, labelArr


def NaiveBayes(Py, Px_y, x):
    '''
    Probability estimation based on naive bayes
    :param Py: prior
    :param Px_y: class-conditional probability
    :param x: the sample x
    :return: the estimated probability of all labels
    '''
    # set the number of features
    featureNum = 784
    # set the number of classes
    classNum = 10
    # the list for storing the estimated probabilities of all labels
    P = [0] * classNum
    for i in range(classNum):
        # Since we calculate the log of prob, it should be summation instead of multiplication
        sum = 0
        # obtain the conditional probability of each feature under each class
        for j in range(featureNum):
            sum += Px_y[i][j][x[j]]
        # add the prior according to formula 4.7(sum coz log transformation)
        P[i] = sum + Py[i]

    # max(P)：= find the maximum probability of all
    return P.index(max(P))


def test(Py, Px_y, testDataArr, testLabelArr):
    '''
    testing procedure
    :param Py: prior
    :param Px_y: conditional probability
    :param testDataArr: testing data
    :param testLabelArr: testing label
    :return: accuracy
    '''
    # count of the error predictions
    errorCnt = 0
    for i in range(len(testDataArr)):
        preds = NaiveBayes(Py, Px_y, testDataArr[i])
        if preds != testLabelArr[i]:
            errorCnt += 1

    return 1 - (errorCnt / len(testDataArr))


def getAllProbability(trainDataArr, trainLabelArr):
    '''
    calculate the prior and class-conditional probability based on the training set
    :param trainDataArr: training data 
    :param trainLabelArr: label of the training data
    :return: prior and class-conditional probability
    '''
    featureNum = 784
    classNum = 10

    # store the prior in Py
    Py = np.zeros((classNum, 1))

    for i in range(classNum):
        # if the count of any class = 0, the prior * conditional prob would become 0, which is not allowable.
        # So we add 1 when count the number of each label while add 10 to the total number of samples.
        # the prior probability
        Py[i] = ((np.sum(np.mat(trainLabelArr) == i)) + 1) / (len(trainLabelArr) + 10)

    # We need to transform the transform it into log form because the original multiplication of 784 numbers may lead to an overflow issue
    # since the number is between 0-1. Besides, log can simplify the calculation.
    Py = np.log(Py)

    # calculate the conditional probability Px_y=P（X=x|Y = y）according to formula 4.10
    # store the conditional probability under all circumstances
    Px_y = np.zeros((classNum, featureNum, 2))  # since we do the binarization before, there are only 0 and 1 for all the features.

    for i in range(len(trainLabelArr)):
        label = trainLabelArr[i]
        x = trainDataArr[i]
        for j in range(featureNum):
            # mark 1 at the appropriate position
            Px_y[label][j][x[j]] += 1

    for label in range(classNum):
        for j in range(featureNum):
            Px_y0 = Px_y[label][j][0]
            Px_y1 = Px_y[label][j][1]
            # according to Bayes Estimation, add 1 to the numerator and 2 to the denominator(the values that each feature can take)
            # based on formula 4.10
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

    # return the prior and conditional probability
    return Py, Px_y


if __name__ == "__main__":
    start = time.time()

    print('----Start reading trainingSet----')
    trainDataArr, trainLabelArr = loadData('mnist/mnist_train.csv')

    print('----Start reading testSet----')
    testDataArr, testLabelArr = loadData('mnist/mnist_test.csv')

    print('----Start training----')
    Py, Px_y = getAllProbability(trainDataArr, trainLabelArr)

    print('----Start testing----')
    accuracy = test(Py, Px_y, testDataArr, testLabelArr)

    print('The test accuracy is:', accuracy)
    print('Time span:', time.time() - start)
