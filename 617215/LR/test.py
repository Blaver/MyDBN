import read_image
import LR
import time
'''
dr = read_image.dataReader("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
testDr = read_image.dataReader("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
myLR = LR.LR(784, 10, 0.002)


print time.localtime()
myLR.training(50, dr, 1000, 50000)

print time.localtime()
    
myLR.testing(testDr, 10000)

#-------------------------Biology----------------------------
dr = read_image.dataReader2("bio_train.dat")
testDr = read_image.dataReader2("bio_train.dat")
myLR = LR.LR(74, 2, 0.001) 

myLR.training(100, dr, 1500, 145751)
myLR.testing(dr, 145751)

'''

dr = read_image.dataReader_CIFAR("data_batch_1.bin")
testDr = read_image.dataReader_CIFAR("data_batch_1.bin")
myLR = LR.LR(3072, 10, 0.002) 

myLR.training(100, dr, 10000, 10000)
myLR.testing(testDr, 10000)







 





    
