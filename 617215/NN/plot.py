import numpy as np
import matplotlib.pyplot as plt

f=open('Result2', 'r')

x=[1,2,3,4,5]
y=[5407, 2946, 2201, 1681, 1823]
z=[4055, 2297, 1657, 1400, 1395]
yy=[]
zz=[]
for i in range(1,5):
    yy.append(1 - 5407.0/(i+1)/y[i])
    zz.append(1 - 4055.0/(i+1)/z[i])
'''
for i in range(145):
    line=f.readline().split('\n')
    print line
    x.append(i)
    #y.append(0.92-float(line[3].split('\n')[0]))
    y.append(1-float(line[0])/10000.0)
'''
plt.figure(1)
plt.plot(x[1:], yy[0:], label='DBN')
plt.plot(x[1:], zz[0:], label='DNN')
plt.xticks(np.linspace(2,5,4,endpoint=True))
plt.legend(loc='upper right', frameon=False)
plt.show()
    
