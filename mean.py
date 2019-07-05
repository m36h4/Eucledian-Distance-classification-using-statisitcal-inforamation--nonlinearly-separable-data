import csv
import random
import math
import operator
import csv
from pandas import *

	
def main():
	#cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.90
	#loadDataset('/home/megha/Downloads/montu lapi/MeghaInternship/Megha/group04 copy.csv', split, trainingSet, testSet)
	with open('/home/megha/Downloads/montu lapi/MeghaInternship/Megha/nonseparable.data', 'r') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(2):
				dataset[x][y] = dataset[x][y]
			if random.random() < split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])
				#print(testSet)
	#with open('/home/megha/Desktop/test.data', 'r') as csvfile:
	#	lines = csv.reader(csvfile)
	#	dataset = list(lines)
	#	for x in range(len(dataset)):
	#		for y in range(4):
	#			dataset[x][y] = float(dataset[x][y])
	       # if random.random() < split:
	           # trainingSet.append(dataset[x])
	        #else:
	#		testSet.append(dataset[x])
	#print('Train set: ' + repr(len(trainingSet)))
	#print 'Test set: ' + repr(len(testSet))
	# generate predictions
	mean = [[0,0,0],[0,0,0],[0,0,0]] 
	mean[0][2] = 'a'
	mean[1][2] = 'b'
	mean[2][2] = 'c'
	for col in range(0,2):
		a = 0
		b = 0
		c = 0
		count1 = 0
		count2 = 0
		count3 = 0
		for row in range(len(trainingSet)):
			if y == 2:
				break
			if trainingSet[row][-1] =='a':
			#if row >= 0 and row < :
				a = a + float(trainingSet[row][col])
				count1 = count1 +1
				#print(a)
			#if row >= 50 and row < 100:
			if trainingSet[row][-1] =='b':
				b = b + float(trainingSet[row][col])
				count2 = count2+1
				#print(b)
			#if row >= 100:
			if trainingSet[row][-1] =='c':
				c = c + float(trainingSet[row][col])
				count3 = count3+1
				#print(c)
		mean[0][col] = float(a)/50.0  
		mean[1][col] = float(b)/50.0 
		mean[2][col] = float(c)/50.0 
	print(count1)
	print(count2)
	print(count3)
	#for x in range(0,3):
		#for y in range(0,2):
			#print(mean[x][y])
	#for x in range(len(testSet)):
		#neighbors = getNeighbors(trainingSet, testSet[x], k)
		#result = getResponse(neighbors)
		#predictions.append(result)
		
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	#accuracy = getAccuracy(testSet, predictions)
	#print('Accuracy: ' + repr(accuracy) + '%')	
	#print 'Confusion Matrix:'
	#print DataFrame(cm, columns=['bayou', 'music_store', 'desert_vegetation'], index=['bayou', 'music_store', 'desert_vegetation'])

	with open("/home/megha/Desktop/separable/mean.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(mean)
	with open("/home/megha/Desktop/separable/test.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(testSet)
	with open("/home/megha/Desktop/separable/train.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(trainingSet)


main()
