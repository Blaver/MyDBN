import read_image
import NN
import time

dr = read_image.dataReader("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
testDr = read_image.dataReader("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
myNN = NN.NN([784, 1000, 500, 10], 0.0005)

f=open("Result", "w")


while True:
    print time.localtime()
    myNN.training(50, dr, 200, 40000)
    print time.localtime()
    ac = myNN.testing(testDr, 10000)
    
    f.write(str(ac)+"\n")
    myNN.l_rate *= 0.998






 





    
