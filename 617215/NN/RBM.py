import random
import math
import copy
from numpy import *
import time

def Sigmoid(x):
        if x < -50.0:
                return 0.0
        return 1.0/(1 + math.exp(-x))


mysigmoid = frompyfunc(Sigmoid, 1, 1)

class Layer(object):
        def __init__(self, num=1, mu=0, sigma=0.01, first=False):
                self.num = num
                self.nets = array([0 for col in range(num)], dtype=float)
                self.bias = array([0 for col in range(num)], dtype=float)
                #self.bias = [0 for col in range(num)]
                self.outs = array([0 for col in range(num)], dtype=float)
                self.sigmas = array([0 for col in range(num)], dtype=float)
                self.gradients = array([0 for col in range(num)], dtype=float)
                #self.gradients = [0 for col in range(num)]
                if not first:
                        for i in range(num):
                                self.bias[i] = sigma*random.randn()
                                
        def printValues(self):
                for i in range(0, self.num):
                        print self.nets[i]
                                
class Weights(object):
        def __init__(self, rows=1, cols=1, mu=0, sigma=0.01):
                self.rows = rows
                self.cols = cols
                self.arr = mat([[0 for col in range(cols)] for row in range(rows)], dtype=float)
                self.gradients = mat([[0 for col in range(cols)] for row in range(rows)], dtype=float)
                
                for i in range(0,rows):
                        for j in range(0,cols):
                                self.arr[i, j] = sigma*random.randn()
		
        def printValues(self):
                for i in range(0, self.rows):
                        for j in range(0, self.cols):
                                print self.arr[i, j]

def toHiddenLayer(a, M, r):
        r.nets = asarray(array((a.outs * M.arr)[0])[0])
        #t0=time.clock()
        for i in range(r.num):
                r.outs[i] = Sigmoid(r.nets[i] + r.bias[i])
        '''
        t0=time.clock()
        r.outs = mysigmoid(r.nets + r.bias)
        '''
        #t1=time.clock()
        #print":" ,t1-t0
def T_toHiddenLayer(a, M, r):
        r.nets = asarray(array((a.outs * M.arr.transpose())[0])[0])
        
        #t0=time.clock()
        for i in range(r.num):
                r.outs[i] = Sigmoid(r.nets[i] + r.bias[i])
        '''
        t0=time.clock()
        r.outs = mysigmoid(r.nets + r.bias)
        '''
        #t1=time.clock()
        
        #print t1-t0
'''
def Sample(x):
        if x > random.random():
                return 1.0
        else:
                return 0.0

sample = frompyfunc(Sample, 1, 1)
'''
      
def sample(outs):
        for i in range(len(outs)):
                if outs[i] > random.random():
                        outs[i] = 1.0
                else:
                        outs[i] = 0.0     
        return


class RBM(object):
    def __init__(self, v_layer, h_layer, weights, rate):
        self.rate = rate
        self.v_layer = v_layer
        self.h_layer = h_layer
        self.weights = weights
        self.v_num = v_layer.num
        self.h_num = h_layer.num
        self.cur_pv0 = array([0 for item in range(h_layer.num)])
        self.cur_v0 = array([0 for item in range(v_layer.num)])
        self.total_error = 0
        self.total_test = 0
        
    def CD_k(self, k):
        for i in range(k):       
            toHiddenLayer(self.v_layer, self.weights, self.h_layer)        
            if i == 0:
                self.cur_pv0 = self.h_layer.outs.copy()        
            #self.h_layer.outs = sample(self.h_layer.outs)
            sample(self.h_layer.outs)
            T_toHiddenLayer(self.h_layer, self.weights, self.v_layer)
            #self.v_layer.outs = sample(self.v_layer.outs)
            sample(self.v_layer.outs)
            
        toHiddenLayer(self.v_layer, self.weights, self.h_layer)
        '''
        Now:
        cur_v0         stores   v(0)           ;
        cur_pv0        stores   P(h_i =1|v(0)) ;
        v_layer.outs   stores   v(k)           ;
        h_layer.outs   stores   P(h_i =1|v(k)) ;
        '''
        return

    def update_gradients(self):
        #updates grad of w_ij
        '''
        for i in range(self.weights.rows):
            for j in range(self.weights.cols):
                a = self.cur_pv0[j]*self.cur_v0[i]
                b = self.h_layer.outs[j]*self.v_layer.outs[i]
                self.weights.gradients[i][j] += (a - b)
        '''
        self.cur_v0.shape = (1, self.v_num)
        self.v_layer.outs.shape = (1, self.v_num)
        self.weights.gradients += self.cur_v0.transpose()*self.cur_pv0 - self.v_layer.outs.transpose()*self.h_layer.outs
        self.cur_v0.shape = (self.v_num,)
        self.v_layer.outs.shape = (self.v_num,)
        '''
        #updates grad of a_i
        
        for i in range(self.v_num):
                self.v_layer.gradients[i] += (self.cur_v0[i] - self.v_layer.outs[i])
        #updates grad of b_i
        for i in range(self.h_num):
                self.h_layer.gradients[i] += (self.cur_pv0[i] - self.h_layer.outs[i])
        '''
        
        self.v_layer.gradients += (self.cur_v0 - self.v_layer.outs)
        self.h_layer.gradients += (self.cur_pv0 - self.h_layer.outs)
        
    def training(self, k):
        self.cur_v0 = self.v_layer.outs.copy()
        self.CD_k(k)
        self.update_gradients()
        return

    def update_Parameters(self, size):
        #updates w_ij
        '''
        for i in range(self.weights.rows):
            for j in range(self.weights.cols):
                self.weights.arr[i, j] += self.rate * self.weights.gradients[i][j]/size
                self.weights.gradients[i][j] = 0
        '''
        self.weights.arr += self.rate/size * self.weights.gradients
        self.weights.gradients = zeros([self.weights.rows, self.weights.cols])
        
        '''
        for i in range(self.v_num):
                self.v_layer.bias[i] += self.rate * self.v_layer.gradients[i]/size
                self.v_layer.gradients[i] = 0.0
        #updates b_i
        for i in range(self.h_num):
                self.h_layer.bias[i] += self.rate * self.h_layer.gradients[i]/size
                self.h_layer.gradients[i] = 0.0
        '''
        #updates a_i
        self.v_layer.bias += self.rate/size * self.v_layer.gradients
        self.v_layer.gradients = zeros([self.v_num,])
        #updates b_i
        self.h_layer.bias += self.rate/size * self.h_layer.gradients
        self.h_layer.gradients = zeros([self.h_num,])
        
    def reset_total_error(self):
        total_error = 0
        total_test = 0
            
    def update_total_error(self):
        im = self.v_layer.outs.copy()
        toHiddenLayer(self.v_layer, self.weights, self.h_layer)        
        #self.h_layer.outs = sample(self.h_layer.outs)
        sample(self.h_layer.outs)
        T_toHiddenLayer(self.h_layer, self.weights, self.v_layer)
        #self.v_layer.outs = sample(self.v_layer.outs)
        sample(self.v_layer.outs)
        
        Sum = 0  
        for i in range(self.v_num):
                Sum +=(im[i] - self.v_layer.outs[i])*(im[i] - self.v_layer.outs[i])
        self.total_error += math.sqrt(Sum)
        self.total_test += 1

    def print_average_error(self, k):
        print k+1, "'th Layer RBM Average_Error: ", float(self.total_error)/float(self.total_test)
                
        
        
            
                

        

