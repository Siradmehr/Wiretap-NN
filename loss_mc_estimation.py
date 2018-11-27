import numpy as np
from keras import backend as K

def loss_mutual_info_mc(y_true, y_pred, k, r, dim, noise_pow=.5, num_samples=50000):
    batch_size = int(2**(r+k))
    split_len = int(2**r)
    entr_z = _estimate_diff_entropy(y_pred, noise_pow, batch_size, dim, num_samples)
    #entr_z = K.repeat_elements(entr_z, batch_size, 0)
    entr_zm = []
    for i in range(2**k):
        _mu = y_pred[i*split_len:(i+1)*split_len, :]
        _entr_zm = _estimate_diff_entropy(_mu, noise_pow, split_len, dim, num_samples)
        entr_zm.append(_entr_zm)
    entr_zm = K.variable(entr_zm)
    entr_zm = K.mean(entr_zm)
    #print(K.eval(entr_zm))
    return (entr_z - entr_zm)/(np.log(2)*k)

def _estimate_diff_entropy(mu, noise_pow, batch_size, dim, num_samples):
    noise_std = np.sqrt(noise_pow)
    samples_z = [K.random_normal((num_samples//batch_size, dim), mean=mu[i], stddev=noise_std) for i in range(batch_size)]
    samples_z = K.concatenate(samples_z, axis=0)
    logpdf = []
    means = K.repeat_elements(mu, (num_samples//batch_size)*batch_size, 0)
    for i in range(batch_size):
        _means = means[(num_samples//batch_size)*batch_size*i:(num_samples//batch_size)*batch_size*(i+1)]
        logpdf.append(-np.log(batch_size) + _norm_logpdf(samples_z, _means, noise_pow, dim))
    logpdf = K.concatenate(logpdf)
    logpdf = K.logsumexp(logpdf, axis=-1)
    return -K.mean(logpdf, axis=-1)

def _norm_logpdf(x, mu, sigma, dim):
    #tensor_sig_inv = 1./sigma * K.constant(np.eye(dim), dtype='float32')
    #exponent = K.dot((x-mu), tensor_sig_inv)
    #exponent = K.batch_dot(exponent, (x-mu), axes=1)
    exponent = 1./sigma * K.batch_dot((x-mu), (x-mu), axes=1)
    factor = dim*np.log(2*np.pi*sigma)
    return -.5*(factor + exponent)
