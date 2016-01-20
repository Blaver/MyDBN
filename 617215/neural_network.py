import copy
from PIL import Image
import random
import math
import read_image
from RBM import Layer, Weights, toHiddenLayer, RBM, sample
from numpy import *

def ReLU(x):
        if x > 0:
                return x
        else:
                return 0

#ReLU = frompyfunc(reLU, 1, 1)

def toHiddenLayer_ReLU(a, M, r):
        r.nets = asarray(array((a.outs * M.arr)[0])[0])
        
        for i in range(r.num):
                r.outs[i] = ReLU(r.nets[i] + r.bias[i])
        
        #r.outs = ReLU(r.nets + r.bias)
        
def toOutputLayer(a, M, r):
        Z = 0.0
        maxValue = 0.0
        r.nets = asarray(array((a.outs * M.arr)[0])[0])
        
        for i in range(r.num):
                if r.nets[i] > maxValue:
                        maxValue = r.nets[i]
                        
        for i in range(r.num):
                r.outs[i] = math.exp(r.nets[i]- maxValue + r.bias[i])
                Z += r.outs[i]
                
        for i in range(r.num):
                r.outs[i] = r.outs[i]/Z

def findMax(r):
        maxinum = 0.0
        maxValue = 0.0;
        for i in range(10):
                if r.outs[i] > maxValue:
                        maxinum = i
                        maxValue = r.outs[i]
        return maxinum
                
class naiveNN(object):
        def __init__(self, layers, rate):
                self.rate = rate
                self.depth = len(layers)
                self.layers = [ Layer() for rows in range(self.depth)]
                self.weightLayers = [Weights() for rows in range(self.depth-1)]
                self.learning_rates = rate #[rate for items in range(self.depth-1)]
                #this line is for testing
                #self.learning_rates[0] = 100.0*rate
                for i in range(self.depth):
                        self.layers[i] = Layer(layers[i])
                for j in range(self.depth-1):
                        self.weightLayers[j] = Weights(layers[j], layers[j+1])

                self.record = [0 for i in range(10)]
                self.cv_right = 0.0
                self.last = 0
                self.decay = 0.998
                self.C = 0.1
                
        def set_decay(self, x):
                self.decay = x

        def result(self, im, label):
                self.layers[0].outs = im.copy()
                self.layers[0].nets = im.copy()
                for i in range(self.depth-2):
                        toHiddenLayer(self.layers[i], self.weightLayers[i], self.layers[i+1])
                toOutputLayer(self.layers[self.depth-2], self.weightLayers[self.depth-2], self.layers[self.depth-1])
                
                return self.layers[self.depth-1].outs[label]

        def result_ReLU(self, im, label):
                self.layers[0].outs = im.copy()
                self.layers[0].nets = im.copy()
                for i in range(self.depth-2):
                        toHiddenLayer_ReLU(self.layers[i], self.weightLayers[i], self.layers[i+1])
                toOutputLayer(self.layers[self.depth-2], self.weightLayers[self.depth-2], self.layers[self.depth-1])
                
                return self.layers[self.depth-1].outs[label]

        def result_print(self, im, label):
                self.layers[0].outs = im.copy()
                self.layers[0].nets = im.copy()
                for i in range(self.depth-2):
                        toHiddenLayer(self.layers[i], self.weightLayers[i], self.layers[i+1])
                toOutputLayer(self.layers[self.depth-2], self.weightLayers[self.depth-2], self.layers[self.depth-1])
                
                r = findMax(self.layers[self.depth-1])              
                if label == r :                 
                        return 1
                else:
                        '''
                        for i in range(10):
                                print i, "   ", self.layers[self.depth-1].outs[i]
                        print label
                        
                        print "------------------------------------------------------------------------------"
                        '''
                        return 0

        def result_print_ReLU(self, im, label):
                self.layers[0].outs = im.copy()
                self.layers[0].nets = im.copy()
                for i in range(self.depth-2):
                        toHiddenLayer_ReLU(self.layers[i], self.weightLayers[i], self.layers[i+1])
                toOutputLayer(self.layers[self.depth-2], self.weightLayers[self.depth-2], self.layers[self.depth-1])
                
                r = findMax(self.layers[self.depth-1])
                if label == r :                 
                        return 1
                else:
                        '''
                        for i in range(10):
                                print i, "   ", self.layers[self.depth-1].outs[i]
                        print label
                        print "------------------------------------------------------------------------------"
                        '''
                        return 0
        
        def backPropagation(self, im, label):
                self.result(im, label)
                '''set the sigmas of the softmax layer '''
                for i in range(self.layers[self.depth-1].num):
                        self.layers[self.depth-1].sigmas[i] = self.layers[self.depth-1].outs[i]
                self.layers[self.depth-1].sigmas[label] -= 1
                '''
                propagation of sigmas
                for i in range(1, self.depth-1):     
                        for j in range(self.layers[self.depth-i-1].num):
                                Sum = 0
                                for k in range(self.layers[self.depth-i].num):
                                        Sum += self.layers[self.depth-i].sigmas[k]*self.weightLayers[self.depth-i-1].arr[j, k]
                                coff = self.layers[self.depth-i-1].outs[j]*(1-self.layers[self.depth-i-1].outs[j])
                                self.layers[self.depth-i-1].sigmas[j] = Sum * coff
                                
                update gradients of every weights
                for i in range(self.depth-1):     
                        for j in range(self.weightLayers[i].rows):
                                for k in range(self.weightLayers[i].cols):
                                        self.weightLayers[i].gradients[j][k] -= self.layers[i].outs[j]*self.layers[i+1].sigmas[k]
                '''
                
                '''propagation of sigmas'''
                for i in range(1, self.depth):
                        t = self.layers[self.depth-i].sigmas * self.weightLayers[self.depth-i-1].arr.transpose()
                        self.layers[self.depth-i-1].sigmas = asarray(array(t[0])[0])
                        '''
                        for j in range(self.layers[self.depth-i-1].num):
                                self.layers[self.depth-i-1].sigmas[j] *= self.layers[self.depth-i-1].outs[j]*(1-self.layers[self.depth-i-1].outs[j])
                        '''
                        self.layers[self.depth-i-1].sigmas *= (self.layers[self.depth-i-1].outs - self.layers[self.depth-i-1].outs * self.layers[self.depth-i-1].outs)
                        
                '''update gradients of every weights'''
                for i in range(self.depth-1):
                        self.layers[i].outs.shape = (1, self.layers[i].num)
                        self.weightLayers[i].gradients -= self.layers[i].outs.transpose() * self.layers[i+1].sigmas
                        self.layers[i].outs.shape = (self.layers[i].num,)

                '''update bias'''
                for i in range(self.depth):
                        self.layers[i].gradients -= self.layers[i].sigmas
                
                        
        def backPropagation_ReLU(self, im, label):
                self.result_ReLU(im, label)
                '''set the sigmas of the softmax layer '''
                for i in range(self.layers[self.depth-1].num):
                        self.layers[self.depth-1].sigmas[i] = self.layers[self.depth-1].outs[i]
                self.layers[self.depth-1].sigmas[label] -= 1
                '''
                propagation of sigmas
                for i in range(1, self.depth-1):     
                        for j in range(self.layers[self.depth-i-1].num):
                                Sum = 0
                                for k in range(self.layers[self.depth-i].num):
                                        Sum += self.layers[self.depth-i].sigmas[k]*self.weightLayers[self.depth-i-1].arr[j, k]
                                if self.layers[self.depth-i-1].outs[j] > 0.0:
                                        self.layers[self.depth-i-1].sigmas[j] = Sum
                                else:
                                        self.layers[self.depth-i-1].sigmas[j] = 0.0
                                        
                update gradients of every weights
                for i in range(self.depth-1):     
                        for j in range(self.weightLayers[i].rows):
                                for k in range(self.weightLayers[i].cols):
                                        self.weightLayers[i].gradients[j][k] -= self.layers[i].outs[j]*self.layers[i+1].sigmas[k]
                '''
                
                '''propagation of sigmas'''
                for i in range(1, self.depth-1):
                        t = self.layers[self.depth-i].sigmas * self.weightLayers[self.depth-i-1].arr.transpose()
                        self.layers[self.depth-i-1].sigmas = asarray(array(t[0])[0])

                        for j in range(self.layers[self.depth-i-1].num):
                                if self.layers[self.depth-i-1].outs[j] <= 0.0:
                                        self.layers[self.depth-i-1].sigmas[j] = 0.0

                '''update gradients of every weights'''
                for i in range(self.depth-1):
                        self.layers[i].outs.shape = (1, self.layers[i].num)
                        self.weightLayers[i].gradients -= self.layers[i].outs.transpose() * self.layers[i+1].sigmas
                        self.layers[i].outs.shape = (self.layers[i].num,)
                        
        def minibatch(self, size, dr, scale):       
                '''calculate gradients'''
                for i in range(size):
                        index = random.randint(0, scale)
                        self.record[dr.readLabel(index)[0]] += 1           
                        self.backPropagation(dr.readOne(index), dr.readLabel(index)[0])
                '''
                count0 = 0
                count1 = 0
                count2 = 0
                count3 = 0
                count4 = 0
                '''
                '''update weights, reset gradients with zero'''     
                for i in range(self.depth-1):
                        '''
                        for j in range(self.weightLayers[i].rows):
                                for k in range(self.weightLayers[i].cols):                                      
                                        self.weightLayers[i].arr[j, k] += self.weightLayers[i].gradients[j][k]*self.learning_rates[i]/size
                                        
                                        if i == 0 and self.weightLayers[i].gradients[j][k] == 0:
                                                count0 += 1
                                        elif i == 1 and self.weightLayers[i].gradients[j][k] == 0:
                                                count1 += 1
                                        elif i == 2 and self.weightLayers[i].gradients[j][k] == 0:
                                                count2 += 1
                                        elif i == 3 and self.weightLayers[i].gradients[j][k] == 0:
                                                count3 += 1
                                        elif i == 4 and self.weightLayers[i].gradients[j][k] == 0:
                                                count4 += 1
                                        self.weightLayers[i].gradients[j][k] = 0
                                        '''
                        #regularization 
                        #self.weightLayers[i].arr = (1 - self.learning_rates * self.C) * self.weightLayers[i].arr
                        
                        self.weightLayers[i].arr += self.learning_rates/size * self.weightLayers[i].gradients
                        self.weightLayers[i].gradients = zeros([self.weightLayers[i].rows, self.weightLayers[i].cols])

                        #learning rate decay
                        #self.learning_rates *= self.decay

                for i in range(self.depth):
                        self.layers[i].bias += self.learning_rates/size * self.layers[i].gradients
                        self.layers[i].gradients = zeros([self.layers[i].num,])
                '''
                print "COUNT0: ", count0
                print "COUNT1: ", count1
                print "COUNT2: ", count2
                print "COUNT3: ", count3
                print "COUNT4: ", count4
                '''
        def minibatch_ReLU(self, size, dr, scale):       
                '''calculate gradients'''
                for i in range(size):
                        '''dr.num-1'''
                        index = random.randint(0, scale)
                        self.record[dr.readLabel(index)[0]] += 1
                        self.backPropagation_ReLU(dr.readOne(index), dr.readLabel(index)[0])
                '''update weights, reset gradients with zero'''
                '''
                count0 = 0
                count1 = 0
                count2 = 0
                count3 = 0
                count4 = 0
                '''
                for i in range(self.depth-1):
                        '''
                        for j in range(self.weightLayers[i].rows):
                                for k in range(self.weightLayers[i].cols):
                                        self.weightLayers[i].arr[j, k] += self.weightLayers[i].gradients[j][k]*self.rate/size
                                        
                                        if i == 0 and self.weightLayers[i].gradients[j][k] == 0:
                                                count0 += 1
                                        elif i == 1 and self.weightLayers[i].gradients[j][k] == 0:
                                                count1 += 1
                                        elif i == 2 and self.weightLayers[i].gradients[j][k] == 0:
                                                count2 += 1
                                        elif i == 3 and self.weightLayers[i].gradients[j][k] == 0:
                                                count3 += 1
                                        elif i == 4 and self.weightLayers[i].gradients[j][k] == 0:
                                                count4 += 1
                                        
                                        self.weightLayers[i].gradients[j][k] = 0
                                        
                
                print "COUNT0: ", count0
                print "COUNT1: ", count1
                print "COUNT2: ", count2
                print "COUNT3: ", count3
                print "COUNT4: ", count4
                '''
                        self.weightLayers[i].arr += self.learning_rates/size * self.weightLayers[i].gradients
                        self.weightLayers[i].gradients = zeros([self.weightLayers[i].rows, self.weightLayers[i].cols])

        def pre_train(self, train_round, size, rate, dr, scale):
                #for each pair of neighbor layers, training RBM
                for k in range(self.depth - 1):
                        rbm = RBM(self.layers[k], self.layers[k+1], self.weightLayers[k], rate)
                        # mini-batch training times 
                        for tr in range(train_round):
                                #mini-batch size
                                for i in range(size):
                                        index = random.randint(0, scale)
                                        im = dr.readOne(index)
                                        for p in range(len(im)):
                                                if im[p] >= 128:
                                                        self.layers[0].outs[p] = 1.0
                                                else:
                                                        self.layers[0].outs[p] = 0.0
                                                        
                                        for j in range(k):
                                                toHiddenLayer(self.layers[j], self.weightLayers[j], self.layers[j+1])
                                                #self.layers[j+1].outs = sample(self.layers[j+1].outs)
                                                sample(self.layers[j+1].outs)
                                        rbm.training(1)
                                rbm.update_Parameters(size)
                                #if tr%100 == 0:
                                        #print tr
                        print "Layer: ", k+1, " Completed!"
                return

        def evaluation_RBM(self, dr, scale, k):
                rbm = RBM(self.layers[k], self.layers[k+1], self.weightLayers[k], 0)
                for i in range(scale):
                        im = dr.readOne(i)
                        for p in range(len(im)):
                                if im[p] >= 128:
                                        rbm.v_layer.outs[p] = 1.0
                                else:
                                        rbm.v_layer.outs[p] = 0.0                
                        rbm.update_total_error()

                        #just for testing
                        size = [28, 28]
                        image = Image.new("L", size)
                        for p in range(len(self.layers[k].outs)):
                                if self.layers[k].outs[p] == 1.0:
                                        image.putpixel([p%28, p/28], 255)
                                else:
                                        image.putpixel([p%28, p/28], 0)
                        image.show()
                        
                rbm.print_average_error(k)

        def cross_validation(self, start, dr, rate, out_count):
                right = 0
                for i in range(start, dr.num):
                        right += self.result_print(dr.readOne(i), dr.readLabel(i)[0])

                print right

                if abs(float(right - self.cv_right)/float(dr.num - start)) < rate:
                        self.last += 1
                        if self.last == out_count:
                                return 0
                        return 1
                elif right > self.cv_right:
                        self.last = 0
                        self.cv_right = right
                        return 1
                
                return 0

        def cross_validation_ReLU(self, start, dr, rate, out_count):
                right = 0
                for i in range(start, dr.num):
                        right += self.result_print_ReLU(dr.readOne(i), dr.readLabel(i)[0])

                print right

                if abs(float(right - self.cv_right)/float(dr.num - start)) < rate:
                        self.last += 1
                        if self.last == out_count:
                                return 0
                        return 1
                elif right > self.cv_right:
                        self.last = 0
                        self.cv_right = right
                        return 1
                
                return 0


                
                        
                        
                
                        
                
                
        
        
                        
                        
                        
                
                
                                
                                
                                        
                        
                        
                        
        
                
                
                
                
                
        
