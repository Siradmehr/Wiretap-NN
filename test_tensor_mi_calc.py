import numpy as np
from keras import backend as K

from autoencoder_wiretap import tensor_entropy_gauss_mix_upper, tensor_norm_pdf


def test_pdf():
    x = np.array([[0, 1.2, 2.2], [-1., 2, -3]])
    mu = np.array([[1.2, 2.2, 3], [-1.4, -2, -3]])
    #expected: [x[0]m[0], x[0]m[1], x[1]m[0], x[1]m[1]]
    #expected=np.array([[0.0136, 1.914e-10, 8.429e-11,1.966e-5]])
    sig = np.array([[1, 0, 0], [0, 1., 0], [0., 0, 1]])

    batch_size = np.shape(mu)[0]
    x = K.variable(value=x)
    x = K.repeat_elements(x, batch_size, axis=0)
    #x = K.repeat(x, batch_size)
    mu = K.variable(value=mu)
    mu = K.tile(mu, (batch_size, 1))
    #mu = K.reshape(mu, (batch_size, batch_size, -1))
    #print(K.eval(x), K.eval(mu))

    norm = tensor_norm_pdf(x, mu, sig)
    norm = K.reshape(norm, (batch_size, batch_size, -1))
    print(K.eval(norm))

    test_sum(norm)

def test_sum(norm):
    inner_sums = K.sum(.5*norm, axis=1, keepdims=True)
    #print(K.eval(inner_sums))
    log_inner = K.log(inner_sums)
    #print(K.eval(log_inner))
    outer_sum = K.sum(.5*log_inner, axis=0)
    print(K.eval(outer_sum))

def test_entropy():
    x = np.array([[0, 1.2, 2.2], [-1., 2, -3]])
    mu = np.array([[1.2, 2.2, 3], [-1.4, -2, -3]])
    #expected: [x[0]m[0], x[1]m[0]
    #expected=np.array([[0.0136, 8.429e-11], [1.914e-10, 1.966e-5]])
    #x = np.array([[0, 1.2, 2.2]])
    #mu = np.array([[1.2, 2.2, 3]])
    sig = np.array([[1, 0, 0], [0, 1., 0], [0., 0, 1]])
    batch_size = np.shape(mu)[0]
    dim = np.shape(mu)[1]
    entr = tensor_entropy_gauss_mix_upper(mu, sig, batch_size, dim)
    print(K.eval(entr))

if __name__ == "__main__":
    #test_pdf()
    test_entropy()
