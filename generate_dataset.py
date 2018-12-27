import numpy as np
import matplotlib
import matplotlib.pyplot as plt

X=[]
Y=[]
with open("/home/manan/intern/datasets/Corners.txt", "r") as filestream:
	for line in filestream:
		currentline = line.split(" ")
	        x_t=[float(currentline[0]),float(currentline[1])]
		
		X.append(x_t)
		Y.append(int(currentline[2]))

print X[0],Y[0]
X=np.array(X)
Y=np.array(Y)
colors = ['red','green','blue','purple']
fig = plt.figure(figsize=(8,8))
plt.scatter(X[:,0] ,X[:,1],c=Y, cmap=matplotlib.colors.ListedColormap(colors))

cb = plt.colorbar()
loc = np.arange(0,max(Y),max(Y)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(colors)
plt.show()
