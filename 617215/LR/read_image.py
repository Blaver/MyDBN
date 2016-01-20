from PIL import Image
import struct
import numpy
from numpy import *
#from numpy import *

class dataReader(object):
    def __init__(self, filename, labels):
        '''read train data'''
        self.trainFile = filename
        binfile1 = open(filename , 'rb')
        self.trainBuf = binfile1.read()
        self.trainIndex = 0
        magic, self.num, self.numRows, self.numColumns = struct.unpack_from('>IIII', self.trainBuf, self.trainIndex)
        #print magic, self.num, self.numRows, self.numColumns
        self.trainIndex += struct.calcsize('>IIII')
        '''read label data'''
        self.labelFile = labels
        binfile2 = open(labels , 'rb')
        self.labelBuf = binfile2.read()
        self.labelIndex = 0
        magic, numImages = struct.unpack_from('>II', self.labelBuf, self.labelIndex)
        self.labelIndex += struct.calcsize('>II')
        #print magic, numImages
             
    def readOne(self, index):
        im = struct.unpack_from('>784B' ,self.trainBuf, self.trainIndex + index*struct.calcsize('>784B'))
        return im
        
    
    def readLabel(self, index):
        im = struct.unpack_from('>B' ,self.labelBuf, self.labelIndex + index*struct.calcsize('>B'))   
        return im


class dataReader2(object):
    def __init__(self, filename):
        '''read train data'''
        frTrain = open(filename)
	self.trainingSet = []
        self.trainingLabels = []
    
	for line in frTrain.readlines():
	    currLine = line.strip().split('\t')
            lineArr = []
	    for i in range(3,len(currLine)):
		lineArr.append(float(currLine[i]))
		self.trainingSet.append(lineArr)
		self.trainingLabels.append(int(currLine[2]))       
             
    def readOne(self, index):
        return self.trainingSet[index]
        
    
    def readLabel(self, index):   
        return self.trainingLabels[index]

class dataReader_CIFAR(object):
    def __init__(self, filename):
        '''read train data'''
        self.trainFile = filename
        binfile1 = open(filename , 'rb')
        self.trainBuf = binfile1.read()
             
    def readOne(self, index):
        im = struct.unpack_from('>3073B' ,self.trainBuf, index*struct.calcsize('>3073B'))
        return im[0], mat(im[1:])
                                
'''
testDr = dataReader("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
for i in range(1):
    record = [0 for items in range(26)]
    im = testDr.readOne(0)
    imm = [0 for items in range(784)]

    for j in range(len(im)):
        record[im[j]/10] += 1
        if im[j] < 128:
            imm[j] = 0
        else:
            imm[j] = 255
    print record
    imageShow(imm, [28,28])
    imageShow(im, [28,28])

'''   

'''
size = 28,28
imageShow(im, size)
print testDr.readLabel(0)
'''

