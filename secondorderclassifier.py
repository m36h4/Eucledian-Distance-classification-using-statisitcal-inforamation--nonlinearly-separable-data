import csv
import random
import math
import operator
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from pandas import *
import numpy as np
from sklearn.metrics import classification_report 
import time
start_time = time.time()

def mahalanobisDistance(instance1, instance2, length , vari):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x]))*vari[x][x], 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, vari):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = mahalanobisDistance(testInstance, trainingSet[x], length, vari)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(len(distances)):
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():

	cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
	# prepare data
	trainingSet=[]
	testSet=[]
	data=[]
	mean=[]
	# generate predictions
	predictions=[]
	y_test=[]
	y_pred=[]
	
	var1 = np.zeros((2,2), dtype=float)
	var2 = np.zeros((2,2), dtype=float)
	var3 = np.zeros((2,2), dtype=float)
	var_atr = np.zeros((2,2), dtype=float)
	var_gen = np.zeros((2,2), dtype=float)
	var_obs = np.zeros((2,2), dtype=float)

	mya = 0
	myb = 0
	myc = 0
	with open('/home/megha/Desktop/nonseparable/mean.csv', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(2):
				dataset[x][y] = float(dataset[x][y])
	        #if random.random() < split:
			mean.append(dataset[x])
	       # else:
	          #  testSet.append(dataset[x])
	with open('/home/megha/Desktop/nonseparable/train.csv', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(2):
				dataset[x][y] = dataset[x][y]
	       # if random.random() < split:
	           # trainingSet.append(dataset[x])
	        #else:
			data.append(dataset[x])
	with open('/home/megha/Desktop/nonseparable/mean.csv', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(2):
				dataset[x][y] = float(dataset[x][y])
	        #if random.random() < split:
			trainingSet.append(dataset[x])
	       # else:
	          #  testSet.append(dataset[x])
	with open('/home/megha/Desktop/nonseparable/test.csv', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(2):
				dataset[x][y] = float(dataset[x][y])
	       # if random.random() < split:
	           # trainingSet.append(dataset[x])
	        #else:
			testSet.append(dataset[x])

	for col in range(0,2):
		for row in range(len(data)):
			if data[row][-1] =='a':
				
				var1[col][col] += pow((float(data[row][col]) - float(mean[0][col])), 2)
				mya = mya +1
				#print(var1[col][col])
			elif data[row][-1] =='b':
				myb = myb + 1
				var2[col][col] += pow((float(data[row][col]) - float(mean[1][col])), 2)
			elif data[row][-1] =='c':
				myc  = myc +1
				var3[col][col] += pow((float(data[row][col]) - float(mean[2][col])), 2)

	#print('var1:')
	#print(mya/2)
	#print('var2:')
	#print(myb/2)
	#print('var3:')
	#print(myc/2)
	mya = 2/mya
	myb = 2/myb
	myc = 2/myc
	var_atr = np.multiply(var1, mya)
	var_gen = np.multiply(var2, myb)
	var_obs = np.multiply(var3, myc)
	#print("var3")	
	#print(var_obs)
	var_atr = np.linalg.inv(var_atr) 
	var_gen = np.linalg.inv(var_gen) 
	var_obs = np.linalg.inv(var_obs) 
	#print("varinverse")	
	#print(var_obs)

	#print('Train set: ' + repr(len(trainingSet)))
	#print('Test set: ' + repr(len(testSet)))

	for x in range(len(testSet)):
		if testSet[x][-1] =='a':
			var = var_atr
		elif testSet[x][-1] =='b':
			var = var_gen
		elif testSet[x][-1] =='b':
			var = var_obs
		neighbors = getNeighbors(trainingSet, testSet[x], var)
		result = getResponse(neighbors)
		predictions.append(result)
		if testSet[x][-1] =='a':
			if result == 'a':
				cm[0][0] += 1
			elif result == 'b':
				cm[0][1] += 1
			elif result == 'c':
				cm[0][2] += 1
		if testSet[x][-1] =='b':
			if result == 'a':
				cm[1][0] += 1
			elif result == 'b':
				cm[1][1] += 1
			elif result == 'c':
				cm[1][2] += 1
		if testSet[x][-1] =='c':
			if result == 'a':
				cm[2][0] += 1
			elif result == 'b':
				cm[2][1] += 1
			elif result == 'c':
				cm[2][2] += 1
		y_test.append(testSet[x][-1])
		y_pred.append(result)
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	print('Confusion Matrix:')
	print(DataFrame(cm, columns=['a', 'b', 'c'], index=['a', 'b', 'c']))
	print(classification_report(y_test, y_pred))
	
	
main()
print("Time Taken :: --- %s seconds ---" % (time.time() - start_time))  
