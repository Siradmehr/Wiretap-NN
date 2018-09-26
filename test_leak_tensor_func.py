import numpy as np

from keras import models, layers
from keras import backend as K

from autoencoder_wiretap import loss_leakage_gauss_mix
from digcommpy import information_theory as it

mu1 = np.array([[1, 2, 3], [4, 5, 6]])
mu2 = np.array([[7, 6, 2], [-1, 0, 2]])
m = np.vstack((mu1, mu2))
#y = np.array([[0], [0], [1], [1]])
y = np.zeros((4, 3))
power = .5

#model = models.Sequential()
#model.add(layers.Dense(5, input_shape=(3,)))
#model.add(layers.Dense(3, activation='sigmoid'))
#model.compile('adam', loss=lambda y_true, y_pred: loss_leakage_gauss_mix(y_true, y_pred, 1, 1, 3, power))
#model.fit(m, y, batch_size=len(m), verbose=2, epochs=3)

expected = it.entropy_gauss_mix_upper(m, power) - .5*(it.entropy_gauss_mix_upper(mu1, power)+it.entropy_gauss_mix_upper(mu2, power))
print(expected)

t_m = K.variable(m)
loss = loss_leakage_gauss_mix(t_m, t_m, 1, 1, 3, power)
print(K.eval(loss)*np.log(2)*1)
