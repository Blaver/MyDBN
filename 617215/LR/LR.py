from PIL import Image
import random
import math
import read_image
from numpy import *

class LR(object):
    
    def __init__(self, input_size, output_size, l_rate, mu = 0, sigma = 0.01):
        #inputs and outputs
        self.inputs = mat([0 for item in range(input_size)], dtype=float)
        self.outs = mat([0 for item in range(output_size)], dtype=float)

        #parameters
        self.bias = mat([0 for item in range(output_size)], dtype=float)
        self.weights = mat([[0 for col in range(output_size)] for row in range(input_size)], dtype=float)

        #gradients of parameters
        self.b_grads = mat([0 for item in range(output_size)], dtype=float)
        self.w_grads = mat([[0 for col in range(output_size)] for row in range(input_size)], dtype=float)

        #records of vector sizes
        self.int_sz = input_size
        self.out_sz = output_size

        #record label of current input
        self.label = 0

        #record testing data
        self.accepted = 0
        self.count = 0
        self.record = range(10)
        
        #learning rate
        self.l_rate = l_rate

        #parameters for initialize biases and weights
        self.mu = mu
        self.sigma = sigma

        #initialize biases and weights
        for i in range(output_size):
            self.bias[0, i] = sigma*random.randn()

        for i in range(0, input_size):
            for j in range(0, output_size):
                self.weights[i, j] = sigma*random.randn()

    def feed_forward(self, print_detail = False):

        Z = 0.0
        maxValue = 0.0
        max_i = 0 
        
        nets = self.inputs * self.weights + self.bias 

        #find maximum value in nets
        for i in range(self.out_sz):
            if nets[0, i] > maxValue:
                maxValue = nets[0, i]
                max_i = i

        #each value in 'nets' have to divide maximum value, prevent overflow
        for i in range(self.out_sz):
            self.outs[0, i] = math.exp(nets[0, i]- maxValue)
            Z += self.outs[0, i]

        #Normalization       
        self.outs = self.outs/Z

        if print_detail:
            print "------------------------------------"
            for i in range(self.out_sz):
                print i,":", self.outs[0, i],'\n'
            

        return max_i

    def back_forward(self):
        self.outs[0, self.label] -= 1
        
        self.b_grads += self.outs  
        self.w_grads += self.inputs.transpose() * self.outs

    def update_paras(self):
        #update parameters
        self.bias -= self.l_rate * self.b_grads
        self.weights -= self.l_rate * self.w_grads

    def mini_batch(self, size, dr, scale):
        #empty the buffer of gradients
        self.b_grads = mat(zeros((1, self.out_sz)))
        self.w_grads = mat(zeros((self.int_sz, self.out_sz)))

        for i in range(size):
            index = random.randint(0, scale)
            
            #This is CIFAR-10
            self.label, self.inputs = dr.readOne(index)
            
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
        for i in range(scale):

            #This is CIFAR-10
            x, self.inputs = dr.readOne(i)
            
            '''
            #This is Biology
            self.inputs = mat(dr.readOne(i))
            x = dr.readLabel(i)
            
            #This is MNIST
            self.inputs = mat(dr.readOne(i))
            x = array(dr.readLabel(i))[0]
            '''
            y = self.feed_forward(False)

            #print "Result(anwser, estimated): ", x, y, '\n'
            
            if x == y:
                self.accepted += 1

        print "Accepted rate: ", (self.accepted + 0.0)/scale



            
        

    
            
            
        
        
        
        
    
    
    
