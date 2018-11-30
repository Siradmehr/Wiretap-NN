import numpy as np
from keras import backend as K
from autoencoder_wiretap import loss_distance_cluster_mean, loss_distance_cluster

n = 3
k = 2
r = 1
x = [[1]*3, [10]*3, [2]*3, [20]*3, [3]*3, [30]*3, [4]*3, [40]*3]
x = np.array(x)
x = np.reshape(x, (2**k, 2**r, n))
c = np.mean(x, axis=0, keepdims=True)
d = x - c
d = np.linalg.norm(d, axis=2)**2
print(d)



x = K.variable(x)
#m = loss_distance_cluster_mean(0, x, k, r, n)
m = loss_distance_cluster(0, x, k, r, n)
print(K.eval(m))
