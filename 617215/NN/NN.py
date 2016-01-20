import copy
from PIL import Image
import random
import math
import read_image
from numpy import *

def Sigmoid(x):
    return 1/(1+math.exp(min(50, -x)))

def ReLU(x):
    return max(0, x)
    
sigmoid = frompyfunc(Sigmoid, 1, 1)

def Myfun(x):
    return x*(1-x)

myfun = frompyfunc(Myfun, 1, 1)

class NN(object):
    def __init__(self, layers, l_rate, mu = 0, sigma = 0.01):
        self.depth = len(layers)
        self.l_rate = l_rate
        
        self.layers  = [ mat([0 for items in range(layers[row])], dtype = float) for row in range(self.depth)]
        self.bias    = [ mat([sigma*random.randn() for items in range(layers[row])], dtype = float) for row in range(self.depth)]
        self.b_grads = [ mat([0 for items in range(layers[row])], dtype = float) for row in range(self.depth)]
        self.weights = [ mat([ [sigma*random.randn() for z in range(layers[x+1])] for y in range(layers[x])], dtype = float) for x in range(self.depth-1)]
        self.w_grads = [ mat([ [0 for z in range(layers[x])] for y in range(layers[x+1])], dtype = float) for x in range(self.depth-1)]
        self.label = 0
           
    def feed_forward(self):
        for i in range(self.depth - 2):
            #self.layers[i+1] = sigmoid(self.layers[i] * self.weights[i] + self.bias[i+1])
            
            #no frompyfunc
            self.layers[i+1] = self.layers[i] * self.weights[i] + self.bias[i+1]
            for j in range(self.layers[i+1].shape[1]):
                self.layers[i+1][0, j] = Sigmoid(self.layers[i+1][0, j])
                #self.layers[i+1][0, j] = ReLU(self.layers[i+1][0, j])
                
        #Softmax Layer
        nets = self.layers[self.depth-2] * self.weights[self.depth-2] + self.bias[self.depth-1]
        nets = exp( mat(nets - nets[0, argmax(nets)]*ones(nets.shape[1]),dtype=float) )  
        self.layers[self.depth-1] = nets/sum(nets)

    def feed_forward_testing(self, print_detail = False):   
        for i in range(self.depth - 2):
            #self.layers[i+1] = sigmoid(self.layers[i] * self.weights[i] + self.bias[i+1])
            
            #no frompyfunc
            self.layers[i+1] = self.layers[i] * self.weights[i] + self.bias[i+1]
            for j in range(self.layers[i+1].shape[1]):
                self.layers[i+1][0, j] = Sigmoid(self.layers[i+1][0, j])
                #self.layers[i+1][0, j] = ReLU(self.layers[i+1][0, j])

        #Softmax Layer
        nets = self.layers[self.depth-2] * self.weights[self.depth-2] + self.bias[self.depth-1]
        nets = exp( mat(nets - nets[0, argmax(nets)]*ones(nets.shape[1]),dtype=float) )
        self.layers[self.depth-1] = nets/sum(nets)

        if print_detail:
            print "------------------------------------"
            for i in range(self.layers[self.depth-1].shape[1]):
                print i,":", self.layers[self.depth-1][0, i],'\n'
                
        return argmax(nets)

    def back_forward(self):
        self.b_grads[self.depth-1] = self.layers[self.depth-1]
        self.b_grads[self.depth-1][0, self.label] -= 1

        for i in range(self.depth-2):
            self.w_grads[self.depth - i - 2] += self.layers[self.depth - i - 2].T * self.b_grads[self.depth - i - 1]
            
            #no frompyfunc
            buf = self.layers[self.depth - i - 2] - multiply(self.layers[self.depth - i - 2], self.layers[self.depth - i - 2]) 
            #buf = sign(self.layers[self.depth - i - 2])
            self.b_grads[self.depth - i - 2] += multiply( self.b_grads[self.depth - i - 1] * self.weights[self.depth - i - 2].T, buf)
            #self.b_grads[self.depth - i - 2] += multiply( self.b_grads[self.depth - i - 1] * self.weights[self.depth - i - 2].T, myfun(self.layers[self.depth - i - 2]) )

        self.w_grads[0] += self.layers[0].T * self.b_grads[1]

    def update_paras(self, size):
        #update parameters
        for i in range(self.depth-1):
            self.bias[i+1] -= self.l_rate/size * self.b_grads[i+1]
            self.weights[i] -= self.l_rate/size * self.w_grads[i]

    def mini_batch(self, size, dr, scale):
        #empty the buffer of gradients
        for i in range(self.depth-1):
            self.b_grads[i+1] = mat( zeros( (1, self.layers[i+1].shape[1]) ) )
            self.w_grads[i] = mat(zeros( (self.layers[i].shape[1], self.layers[i+1].shape[1]) ))

        for i in range(size):
            index = random.randint(0, scale)

            '''
            #This is CIFAR-10
            self.label, self.layers[0] = dr.readOne(index)
            '''
            
            #This is MNIST
            self.layers[0] = mat(dr.readOne(index))
            self.label = dr.readLabel(index)[0]
            
            '''
            #This is Biology Data
            self.inputs = mat(dr.readOne(index))
            #self.label = dr.readLabel(index)
            '''
            self.feed_forward()
            self.back_forward()

        self.update_paras(size)

    def training(self, mini_size, dr, rounds, dr_scale):
        for i in range(rounds):
            self.mini_batch(mini_size, dr, dr_scale)

    def testing(self, dr, scale):
        accepted = 0
         
        for i in range(scale):

            '''
            #This is CIFAR-10
            x, self.layers[0] = dr.readOne(i)
            
            
            
            #This is Biology
            self.inputs = mat(dr.readOne(i))
            x = dr.readLabel(i)
            '''
            #This is MNIST
            self.layers[0] = mat(dr.readOne(i))
            x = dr.readLabel(i)[0]
            
            y = self.feed_forward_testing(False)
            
            if x == y:
                accepted += 1


        print "Accepted rate: ", (accepted + 0.0)/scale
        return (accepted + 0.0)/scale

