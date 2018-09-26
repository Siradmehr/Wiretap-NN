import numpy as np

from keras import models, layers
from keras import backend as K

from autoencoder_wiretap import loss_leakage_gauss_mix
from digcommpy import information_theory as it

mu1 = np.array([[1, 2, 3], [4, 5, 6], [-1, 4, 2], [-7, 6, 1]])
mu2 = np.array([[7, 6, 2], [-1, 0, 2], [1, 3, 3], [-1, 2, 1]])
mu3 = np.array([[7, 6, 1], [-2, 4, 2], [3, 1, -3], [-1, 1, 1]])
mu4 = np.array([[-7, 6, 2], [-3, 3, 3], [1, -1, -3.4], [-1.2, 1.2, -1]])
mu = (mu1, mu2, mu3, mu4)
m = np.vstack(mu)
y = np.zeros((4, 3))
power = .5

#model = models.Sequential()
#model.add(layers.Dense(5, input_shape=(3,)))
#model.add(layers.Dense(3, activation='sigmoid'))
#model.compile('adam', loss=lambda y_true, y_pred: loss_leakage_gauss_mix(y_true, y_pred, 1, 1, 3, power))
#model.fit(m, y, batch_size=len(m), verbose=2, epochs=3)

k = int(np.log2(len(mu)))
h_zm = [it.entropy_gauss_mix_upper(_mu, power) for _mu in mu]
h_z = it.entropy_gauss_mix_upper(m, power)
expected = (h_z - np.mean(h_zm))/(np.log(2)*k)
t_m = K.variable(m, dtype='float32')
loss = loss_leakage_gauss_mix(t_m, t_m, k, 2, 3, power)

print("Expected h_z:\t{}".format(h_z))
print("Expected h_zm:\t{} ({})".format(np.mean(h_zm), h_zm))
print("Expected leakage:\t{}".format(expected))
print(K.eval(loss))
