# -*- coding: utf-8 -*-
"""
Input:  inX: vector to compare to existing dataset (1xN)
        dataSet: size m data set of known vectors (NxM)
        labels: data set labels (1xM vector)
        k: number of neighbors to use for comparison 
          
Output: the most popular class label
"""

from numpy import *
import operator
import os
from data_util import DataUtils
import datetime  
import time
import matplotlib.pyplot as plt
import numpy as np

trainfile_X = 'train-images.idx3-ubyte'
trainfile_y = 'train-labels.idx1-ubyte'
testfile_X = 't10k-images.idx3-ubyte'
testfile_y = 't10k-labels.idx1-ubyte'

# 定义kNN分类函数
def kNNClassify(newInput, dataSet, labels, k):
	numSamples = dataSet.shape[0] # shape[0] 代表行数

	## step 1: 计算欧式距离
	# tile(A, reps): Construct an array by repeating A reps times
	# the following copy numSamples rows for dataSet
	diff = tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise
	squaredDiff = diff ** 2 # squared for the subtract
	squaredDist = sum(squaredDiff, axis = 1) # sum is performed by row
	distance = squaredDist ** 0.5

	## step 2: 计算距离
	# argsort() returns the indices that would sort an array in a ascending order
	sortedDistIndices = argsort(distance)

	classCount = {} # 定义一个字典用于放入元素 
	for i in xrange(k):
		## step 3: 选择k值对应的距离
		voteLabel = labels[sortedDistIndices[i]]

		## step 4: 记录标签出现的次数
		# when the key voteLabel is not in dictionary classCount, get()
		# will return 0
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

	## step 5: 返回对应标签最多的类别
	maxCount = 0
	for key, value in classCount.items():
		if value > maxCount:
			maxCount = value
			maxIndex = key

	return maxIndex

# 将图片转化为向量形式
def  img2vector(filename):
 	rows = 32
 	cols = 32
 	imgVector = zeros((1, rows * cols)) 
 	fileIn = open(filename)
 	for row in xrange(rows):
 		lineStr = fileIn.readline()
 		for col in xrange(cols):
 			imgVector[0, row * 32 + col] = int(lineStr[col])

 	return imgVector


# 测试HandWriting类
def testHandWritingClass():
	## step 1: load data
	ISOTIMEFORMAT='%Y-%m-%d %X'
	load_time=time.strftime(ISOTIMEFORMAT, time.localtime())
	print "step 1: loading data...",
	print"load_time:%s"%load_time
	train_x=DataUtils(filename=trainfile_X).getImage()
	train_y=DataUtils(filename=trainfile_y).getLabel()
	test_x=DataUtils(testfile_X).getImage()
	test_y=DataUtils(testfile_y).getLabel()


	## step 2: training...
	train_time=time.strftime(ISOTIMEFORMAT, time.localtime())
	print "step 2: training...",
	print "train_time:%s"%train_time
	pass

	## step 3: testing
	test_time=time.strftime(ISOTIMEFORMAT, time.localtime())
	print "step 3: testing...",
	print "test_time:%s"%test_time
	numTestSamples = test_x.shape[0]
	matchCount = 0
	
	result=[]
	for m in range(1,121):
		for i in xrange(numTestSamples):
			predict = kNNClassify(test_x[i], train_x, train_y, m)
			if predict == test_y[i]:
				matchCount += 1
		error = 1 - float(matchCount) / numTestSamples
		matchCount=0
		result_time=time.strftime(ISOTIMEFORMAT, time.localtime())
		print m,"result_time:%s"%result_time
		print 'The classify error is: %.4f' %error
		result.append(error)

	##step4:show the error_plot
	x = range(1,121)
	y = result
	plt.plot(x, y)
	plt.show()

if __name__ == '__main__':
	testHandWritingClass()
	

