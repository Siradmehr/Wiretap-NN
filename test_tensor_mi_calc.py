import numpy as np
from keras import backend as K

from autoencoder_wiretap import tensor_entropy_gauss_mix_upper, tensor_norm_pdf


x = np.array([[0, 1.2, 2.2], [-1., 2, -3]])
mu = np.array([[1.2, 2.2, 3], [-1.4, -2, -3]])
#expected: [x[0]m[0], x[1]m[0]
#expected=np.array([[0.0136, 8.429e-11], [1.914e-10, 1.966e-5]])
#x = np.array([[0, 1.2, 2.2]])
#mu = np.array([[1.2, 2.2, 3]])
sig = np.array([[1, 0, 0], [0, 1., 0], [0., 0, 1]])
batch_size = np.shape(mu)[0]
dim = np.shape(mu)[1]

x = K.variable(value=x)
mu = K.variable(value=mu)
#sig = K.variable(value=sig)

#entr = tensor_entropy_gauss_mix_upper(mu, sig, batch_size, dim)
#print(K.eval(entr))
norm = tensor_norm_pdf(x, mu, sig)
print(K.eval(norm))
