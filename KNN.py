# coding=utf-8
# Author:Summer
# Date:2018-12-28
# Email:xyyang1994@gmail.com
# Reference: 李航《统计学习方法》Chapter3

'''
Dataset: Mnist
Training set size: 60000 observations, 1 label + 784 features
Testset size: 10000（actual: 200)
------------------------------
results：with k = 10
Distance measurement -- Euclidean distance (only test the first 200 test samples)
    Test accuracy: 97%
    Running time: 286s
Distance measurement -- Manhattan distance
    Test accuracy: 14%
    Running time: 209s
'''

import numpy as np
import time


def loadData(fileName):
    '''
    load the file
    fileName: the path to data
    return: dataset and labels
    '''
    print('------Start reading file------')
    dataArr = []
    labelArr = []
    # read the file
    fr = open(fileName)
    for line in fr.readlines():
        # strip: Returns a copy of the string with the leading and trailing characters removed.(default: whitespace)
        curLine = line.strip().split(',')
        # curLine[0] is the label
        # convert the string to integers when storing the features into array
        dataArr.append([int(num) for num in curLine[1:]])
        labelArr.append(int(curLine[0]))
    return dataArr, labelArr


def calcDist(x1, x2):
    '''
    calculate the Euclidean distance between two vectors x1 and x2
    '''
    return np.sqrt(np.sum(np.square(x1 - x2)))

    # Manhattan distance calculation
    # return np.sum(x1 - x2)


def getClosest(trainDataMat, trainLabelMat, x, topK):
    '''
    Predict the label of sample x.
    Find the topK closest points to sample x, get their labels and predict the label of sample x with the most voted one.
    :param trainDataMat:Training data
    :param trainLabelMat: Labels of training data
    :param x:sample x
    :param topK:k
    :return:the predicted label of the sample x
    '''
    # a new list to store the distance between sample x and every record of the training data
    distList = [0] * len(trainLabelMat)

    for i in range(len(trainDataMat)):
        x1 = trainDataMat[i]
        curDist = calcDist(x1, x)
        distList[i] = curDist

    # sort the distances
    # argsort: the function sort the distances in an ascending order and return their indexes
    # eg:
    #   >>> x = np.array([3, 1, 2])
    #   >>> np.argsort(x)
    #   array([1, 2, 0])
    # ----------------Optimization-------------------
    # No need to sort the entire list coz we only need the topK smallest distances.
    # Do not do the optimization here since the calculcation of the distances between high-dimensional vectors costs much more than soring.
    topKList = np.argsort(np.array(distList))[:topK]

    # create a list to store the votes of each label
    labelList = [0] * 10
    for index in topKList:
        labelList[int(trainLabelMat[index])] += 1

    return labelList.index(max(labelList))


def test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):
    '''
    test the accuracy
    :param trainDataArr: Training data features
    :param trainLabelArr: Labels of the training data
    :param testDataArr: Test data features
    :param testLabelArr: Labels of the test data
    :param topK: k
    :return: test accuracy
    '''
    print('------Start testing------')
    # convert the list/arrays to matrix for calculation convenience
    trainDataMat = np.mat(trainDataArr)  # the dimension: 60000 * 784
    trainLabelMat = np.mat(trainLabelArr).T
    testDataMat = np.mat(testDataArr)
    testLabelMat = np.mat(testLabelArr).T

    errorCnt = 0
    # Huge time cost if test the entire testset. Here only tests 200 samples.
    # for i in range(len(testDataMat)):
    for i in range(200):
        # print('test %d:%d'%(i, len(trainDataArr)))
        print('test %d:%d' % (i, 200))
        x = testDataMat[i]
        y = getClosest(trainDataMat, trainLabelMat, x, topK)
        if y != testLabelMat[i]:
            errorCnt += 1

    # return 1 - (errorCnt / len(testDataMat))
    return 1 - (errorCnt / 200)


if __name__ == "__main__":
    start = time.time()

    # load the training data
    trainDataArr, trainLabelArr = loadData('mnist/mnist_train.csv')
    # load the test data
    testDataArr, testLabelArr = loadData('mnist/mnist_test.csv')
    # calculate the test accuracy
    accur = test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 10)
    print('test accur is: %d' % (accur * 100), '%')

    end = time.time()
    print('time span:', end - start)
