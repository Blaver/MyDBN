import read_image
import neural_network
import time

dr = read_image.dataReader("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
testDr = read_image.dataReader("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
myNN = neural_network.naiveNN([784, 1000, 500, 10], 0.0005)

count1 = 0
count2 = 0

testRound = 10000

pretrain =  False

isReLU = True

if pretrain:
    while (True):
        myNN.pre_train(3600, 50, 0.5, dr, dr.num - 10000)
        myNN.evaluation_RBM(dr, 20, 0)

    for i in range(testRound):  
        count1 += myNN.result_print(dr.readOne(i), dr.readLabel(i)[0])
        
    for i in range(testRound):
        count2 += myNN.result_print(testDr.readOne(i), testDr.readLabel(i)[0])

    print "Accepted rate of Training: ", count1, count1/float(testRound)

    print "Accepted rate of Testing: ", count2, count2/float(testRound)

    count1 = 0
    count2 = 0

    weightsFile = open("Weights",  'w')
    for i in range(myNN.depth-1):
        for j in range(myNN.weightLayers[i].rows):
            for k in range(myNN.weightLayers[i].cols):
                weightsFile.write(str(i)+", "+str(j)+", "+str(k)+": "+str(myNN.weightLayers[i].arr[j,k])+'\n')

    for i in range(myNN.depth):
        for j in range(len(myNN.layers[i].bias)):
            weightsFile.write(str(i)+", "+str(j)+": "+str(myNN.layers[i].bias[j])+'\n')
        
    weightsFile.write("----------------END----------------")
    
    
if isReLU:
    while True:
        print time.localtime()
        for i in range(200):
            myNN.minibatch_ReLU(50, dr, dr.num - 10000)
        print time.localtime()
        if myNN.cross_validation_ReLU(50000, dr, 0.002, 5) == 0:
            #break
            myNN.learning_rates *= 0.5

    for i in range(testRound):  
        count1 += myNN.result_print_ReLU(dr.readOne(i), dr.readLabel(i)[0])
        
    for i in range(testRound):
        count2 += myNN.result_print_ReLU(testDr.readOne(i), testDr.readLabel(i)[0])

else:
    while True:
        for i in range(200):
            myNN.minibatch(50, dr, dr.num - 10000)
        if myNN.cross_validation(50000, dr, 0.005, 5) == 0:
            break
            #myNN.

    for i in range(testRound):  
        count1 += myNN.result_print(dr.readOne(i), dr.readLabel(i)[0])
        
    for i in range(testRound):
        count2 += myNN.result_print(testDr.readOne(i), testDr.readLabel(i)[0])

print myNN.record

print "Accepted rate of Training: ", count1, count1/float(testRound)

print "Accepted rate of Testing: ", count2, count2/float(testRound)





 





    
