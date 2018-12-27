from som import MiniSom

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

data = np.genfromtxt('S11.csv', delimiter=',', usecols=(0, 1, 2))
# data normalization
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)

# Initialization and training
som = MiniSom(7, 7, 3, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data)
print("Training...")
som.train_random(data, 1000)  # random training
print("\n...ready!")

# Plotting the response for each pattern in the  dataset
plt.bone()
plt.pcolor(som.distance_map().T)  # plotting the distance map as background
plt.colorbar()

#target = np.genfromtxt('S1.csv', delimiter=',', usecols=(3), dtype=str)
#t = np.zeros(len(target), dtype=int)

t[target == '0'] = 0
t[target == '1'] = 1
t[target == '2'] = 2
t[target == '3'] = 3
#t[target == '4'] = 4
#t[target == '5'] = 5
#t[target == '6'] = 6
"""
# use different colors and markers for each label
markers = ['o', 's', 'D','p','8','*']
colors = ['r', 'g', 'b', 'c', 'y', 'm']
for cnt, xx in enumerate(data):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markerfacecolor='None',Untitled Folder
              markersize=12, markeredgewidth=2)
#markeredgecolor=colors[t[cnt]]
plt.axis([0,7, 0, 7])
plt.show()
quantization_error = som.quantization_error(data)
win_map=som.win_map(data)

print(quantization_error)
#print(win_map)

