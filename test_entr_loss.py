from autoencoder_wiretap import loss_gauss_mix_entropy
import numpy as np
from keras import layers
from keras import models


m = np.array([[1, 2], [-1, -2], [-4, 0], [0.5, 2.1]])
#y = np.array([[-1, -2], [-.5, 1], [-.2, 1], [1.2, 1]])
y = np.array([[-1, -2, 2], [-.5, 1, -1], [-.2, 1, .1], [1.2, 1, 0]])
batch_size = np.shape(m)[0]
dim = np.shape(y)[1]

model = models.Sequential()
model.add(layers.Dense(2, input_shape=(2,)))
model.add(layers.Dense(3))
model.compile('adam', lambda x, y: loss_gauss_mix_entropy(x, y, batch_size, dim))
#model.compile('adam', 'mse')

model.fit(m, y, epochs=2)
