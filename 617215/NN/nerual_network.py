import copy
from PIL import Image
import random
import math
import read_image
from numpy import *

def sigmoid(x):
    return 1/(1+math.exp(min(50, -x)))

def myfun(x):
    return x*(1-x)

class NN(object):
    def __init__(self, layers, l_rate, mu = 0, sigma = 0.01):
        self.depth = len(layers)
        self.l_rate = rate
        
        self.layers  = [ mat([0 for items in range(layers[row])], dtype = float) for row in range(self.depth)]
        self.bias    = [ mat([sigma*random.randn() for items in range(layers[row])], dtype = float) for row in range(self.depth)]
        self.b_grads = [ mat([0 for items in range(layers[row])], dtype = float) for row in range(self.depth)]
        self.weights = [ mat([ [sigma*random.randn() for z in range(layers[x])] for y in range(layers[x+1])], dtype = float) for x in range(self.depth-1)]
        self.w_grads = [ mat([ [0 for z in range(layers[x])] for y in range(layers[x+1])], dtype = float) for x in range(self.depth-1)]

        self.label = 0
           
    def feed_forward(self):   
        for i in range(self.depth - 2):
            self.layers[i+1] = sigmoid(self.layer[i] * self.weights[i] + self.bias[i+1])

        #Softmax Layer
        nets = self.layers[self.depth-2] * self.weights[self.depth-2] + self.bias[self.depth-1] 
        nets = math.exp( nets - nets[argmax(nets)]*ones(len(nets)) )   
        self.layers[self.depth-1] = nets/sum(nets)

    def feed_forward_testing(self, print_detail = False):   
        for i in range(self.depth - 2):
            self.layers[i+1] = sigmoid(self.layer[i] * self.weights[i] + self.bias[i+1])

        #Softmax Layer
        nets = self.layers[self.depth-2] * self.weights[self.depth-2] + self.bias[self.depth-1] 
        return argmax(nets)

    def back_forward(self)
        self.b_grads[self.depth-1][0, self.label] -= 1

        for i in range(self.depth-2):
            self.w_grads[self.depth - i - 2] += self.layers[self.depth - i - 2].T * self.b_grads[self.depth - i - 1]    
            self.b_grads[self.depth - i - 2] += multiply( self.b_grads[self.depth - i - 1] * self.weights[self.depth - i - 2].T, myfun(self.layers[self.depth - i - 2]) )

        self.w_grads[0] += self.layers[0].T * self.b_grads[1]

    def update_paras(self):
        #update parameters
        for i in range(self.depth-1):
            self.bias[i+1] -= self.l_rate * self.b_grads[i]
            self.weights[i] -= self.l_rate * self.w_grads[i]

    def mini_batch(self, size, dr, scale):
        #empty the buffer of gradients
        for i in range(self.depth-1):
            self.b_grads[i+1] = zeros((len(self.layers[i+1])))
            self.w_grads[i] = zeros( (len(self.layers[i]), len(self.layers[i+1])) )

        for i in range(size):
            index = random.randint(0, scale)
            
            #This is CIFAR-10
            self.label, self.layers[0] = dr.readOne(index)
            
            '''
            #This is MNIST
            self.inputs = mat(dr.readOne(index))
            self.label = array(dr.readLabel(index))[0]

            #This is Biology Data
            self.inputs = mat(dr.readOne(index))
            #self.label = dr.readLabel(index)
            '''
            self.feed_forward()
            self.back_forward()

        self.update_paras()

    def training(self, mini_size, dr, rounds, dr_scale):
        for i in range(rounds):
            self.mini_batch(mini_size, dr, dr_scale)

    def testing(self, dr, scale):
        accepted = 0
         
        for i in range(scale):

            #This is CIFAR-10
            x, self.layers[0] = dr.readOne(i)
            
            '''
            #This is Biology
            self.inputs = mat(dr.readOne(i))
            x = dr.readLabel(i)
            
            #This is MNIST
            self.inputs = mat(dr.readOne(i))
            x = array(dr.readLabel(i))[0]
            '''
            y = self.feed_forward_testing(False)
            
            if x == y:
                accepted += 1

        print "Accepted rate: ", (accepted + 0.0)/scale

