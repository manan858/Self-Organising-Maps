# Self-Organising-Maps
Numpy based implementation of Self-Organising Maps(SOM) and  visualize the algorithm on different neighbourhood functions with the help of some synthetic datasets like (Elliptical, triangular, Gaussian etc).SOM is a type of Artificial Neural Network able to convert complex, nonlinear statistical relationships between high-dimensional data items into simple geometric relationships on a low-dimensional display.
## Using the trained SOM
After the training you will be able to

* Compute the coordinate assigned to an observation x on the map with the method winner(x).
* Compute the average distance map of the weights on the map with the method distance_map().
* Compute the number of times that each neuron have been considered winner for the observations of a new data set with the method activation_response(data).
* Compute the quantization error with the method quantization_error(data)
