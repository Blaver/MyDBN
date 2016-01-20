import copy
from PIL import Image
import random
import math
import read_image
from numpy import *
from mpi4py import MPI
import time

class LR_MPI(object):
    
    def __init__(self, input_size, output_size, l_rate, myrank = 0, mu = 0, sigma = 0.01):
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

        '''----MPI----'''
        #MPI rank
        self.rank = myrank
        '''----MPI----'''
        
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

        '''----MPI-----'''
        #split the weights matrix by column
        local_nets = self.inputs * self.weights[:, (2*self.rank, 1+2*self.rank)]

        #merge local results
        nets = hstack(comm.allgather(local_nets)) + self.bias
        '''----MPI-----'''
        
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

    def feed_forward_test(self, print_detail = False):

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

        '''----MPI----'''
        #split the weights matrix by column
        local_wgrads = self.inputs.transpose() * self.outs[0, (2*self.rank, 1+2*self.rank)]

        #merge local results
        self.w_grads += hstack(comm.allgather(local_wgrads))
        
        #self.w_grads += self.inputs.transpose() * self.outs
        '''----MPI----'''

    def update_paras(self):
        #update parameters
        self.bias -= self.l_rate * self.b_grads
        self.weights -= self.l_rate * self.w_grads

    def mini_batch(self, size, dr, scale):
        #empty the buffer of gradients
        self.b_grads = mat(zeros((1, self.out_sz)))
        self.w_grads = mat(zeros((self.int_sz, self.out_sz)))
        
        for i in range(size):
            #root select training examples, them broadcast it
            index = array(0, dtype=int)
            if self.rank == 0:
                index.fill(random.randint(0, scale))
            comm.Bcast([index, MPI.INT], root = 0)

            self.inputs = mat(dr.readOne(index))
            self.label = array(dr.readLabel(index))[0]
            #self.label = dr.readLabel(index)

            self.feed_forward()
            self.back_forward()

        self.update_paras()

    def training(self, mini_size, dr, rounds, dr_scale):
        for i in range(rounds):
            self.mini_batch(mini_size, dr, dr_scale)

    def testing(self, dr, scale):
        for i in range(scale):
            self.inputs = mat(dr.readOne(i))

            x = array(dr.readLabel(i))[0]
            #x = dr.readLabel(i)
            y = self.feed_forward_test(False)

            #print "Result(anwser, estimated): ", x, y, '\n'
            
            if x == y:
                self.accepted += 1

        print "Accepted rate: ", (self.accepted + 0.0)/scale


'''---------------TEST------------------'''
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

dr = read_image.dataReader("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
testDr = read_image.dataReader("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
myLR = LR_MPI(784, 10, 0.002, myrank) 

if myLR.rank == 0:
    print time.localtime()
myLR.training(50, dr, 1000, 50000)
if myLR.rank == 0:
    print time.localtime()

if myLR.rank == 0:
    myLR.testing(testDr, 10000)
