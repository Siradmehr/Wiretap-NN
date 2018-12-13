import numpy as np
from keras import backend as K
from keras import models, layers
import tensorflow as tf

def loss_taylor_expansion_gm(y_true, y_pred, k, r, dim, noise_pow=.5):
    sigma = np.eye(dim)*noise_pow
    batch_size = int(2**(r+k))
    split_len = int(2**r)
    entr_z = _entropy_gm_taylor_expansion(y_pred, sigma, batch_size)
    entr_zm = []
    for message in range(2**k):
        _y_pred_message = y_pred[message*split_len:(message+1)*split_len, :]
        _entr_zm = _entropy_gm_taylor_expansion(_y_pred_message, sigma, split_len)
        entr_zm.append(_entr_zm)
    entr_zm = K.stack(entr_zm, axis=0)
    entr_zm = K.mean(entr_zm)
    return K.mean(entr_z - entr_zm)/(np.log(2)*k)

def _entropy_gm_taylor_expansion(mu, sigma, num_comp):
    noise_pow = sigma[0][0]
    entr_zero_ord = _entropy_gm_taylor_zero(mu, sigma, num_comp)
    #return entr_zero_ord
    #result = []
    x = K.repeat(mu, num_comp)
    mu = K.permute_dimensions(x, (1, 0, 2))
    cap_f = _gm_capital_f(x, mu, sigma, num_comp) # X x N x N
    cap_f_c = tf.linalg.trace(cap_f) # X
    _result = K.mean(cap_f_c)
    #for component in range(num_comp):
    #    mu_i = K.expand_dims(mu[component, :], axis=0)
    #    _cap_f = _gm_capital_f(mu_i, mu, sigma, num_comp)
    #    result.append(tf.linalg.trace(_cap_f))
    #result = K.stack(result, axis=0)
    #result = K.mean(result)
    result = entr_zero_ord - .5*noise_pow*_result
    return result

def _entropy_gm_taylor_zero(mu, sigma, num_comp):
    x = K.repeat(mu, num_comp)
    mu = K.permute_dimensions(x, (1, 0, 2))
    log_pdf = tensor_gm_logpdf(x, mu, sigma, num_comp)
    return -K.mean(log_pdf)

def _gm_capital_f(x, mu, sigma, num_comp):
    noise_pow = sigma[0][0]
    dim = np.shape(sigma)[0]
    pdf_gm = _tensor_gm_pdf(x, mu, sigma, num_comp)
    #xx = K.expand_dims(x, axis=3)
    #mm = K.expand_dims(mu, axis=3)
    _grad = tensor_gradient_gm(x, mu, sigma, num_comp) # X x N
    _grad = K.expand_dims(_grad, axis=1) # X x 1 x N
    _grad = K.tile(_grad, (1, num_comp, 1)) # X x M x N
    _grad = K.expand_dims(_grad, axis=3)
    _xm = K.expand_dims(x-mu, axis=3)
    _part1 = K.batch_dot(_xm, _grad, axes=3)  # X x M x N x N
    _pdf_gm = K.expand_dims(K.expand_dims(K.expand_dims(pdf_gm)))
    _part1 = _part1/pdf_gm  # X x M x N x N
    _part2 = K.batch_dot(_xm, _xm, axes=3)/noise_pow
    _eye = K.expand_dims(K.expand_dims(K.eye(dim), 0), 0)
    _result = _part1 + _part2 - _eye
    _norm = K.expand_dims(K.expand_dims(_tensor_norm_pdf(x, mu, sigma)))
    _result = _result * _norm # X x M x N x N
    _result = K.mean(_result, axis=1) # X x N x N
    result = _result/(K.expand_dims(K.expand_dims(pdf_gm))*noise_pow)
    return result
    #
    #results = []
    #for component in range(num_comp):
    #    mu_i = K.expand_dims(mu[component, :], axis=0)
    #    _pdf_norm = _tensor_norm_pdf(x, mu_i, sigma)
    #    _grad = tensor_gradient_gm(x, mu, sigma, num_comp)
    #    _part1 = K.dot(tf.linalg.transpose(x-mu_i), _grad)/pdf_gm
    #    _part2 = 1./noise_pow * K.dot(tf.linalg.transpose(x-mu_i), x-mu_i)
    #    _result = (_part1 + _part2 - K.eye(dim))*_pdf_norm
    #    results.append(_result)
    #results = K.stack(results, axis=2)
    #result = K.mean(results, axis=2)
    #
    #return result/(pdf_gm*noise_pow)

def tensor_gradient_gm(x, mu, sigma, num_comp):
    noise_pow = sigma[0][0]
    _grad = (x-mu)*_tensor_norm_pdf(x, mu, sigma)
    grad = K.mean(axis=1) # 
    #result = []
    #for component in range(num_comp):
    #    mu_i = K.expand_dims(mu[component, :], axis=0)
    #    _grad = (x-mu_i)*_tensor_norm_pdf(x, mu_i, sigma)
    #    result.append(_grad)
    #result = K.stack(result, axis=-1)
    ##result = K.concatenate(result, axis=0)
    #result = K.mean(result, axis=-1)
    return grad

def _tensor_gm_pdf(x, mu, sigma, num_comp):
    dim = np.shape(sigma)[0]
    #x = K.repeat_elements(x, num_comp, 0)
    _pdf_exp = _tensor_norm_pdf_exponent(x, mu, sigma)
    factor = (1./num_comp)/(np.sqrt(2*np.pi*sigma[0][0])**dim)
    pdf = factor * K.sum(K.exp(_pdf_exp), axis=1)
    return pdf

def tensor_gm_logpdf(x, mu, sigma, num_comp):
    dim = np.shape(sigma)[0]
    #x = K.repeat_elements(x, num_comp, 0)
    _pdf_exp = _tensor_norm_pdf_exponent(x, mu, sigma)
    factor = K.constant((1./num_comp)/(np.sqrt(2*np.pi*sigma[0][0])**dim), dtype='float32')
    pdf = K.log(factor) + K.logsumexp(_pdf_exp, axis=1)
    return pdf

def _tensor_norm_pdf(x, mu, sigma):
    dim = np.shape(sigma)[0]
    _pdf_exp = _tensor_norm_pdf_exponent(x, mu, sigma)
    factor = 1./(np.sqrt(2*np.pi*sigma[0][0])**dim)
    pdf = factor * K.exp(_pdf_exp)
    return pdf

def _tensor_norm_pdf_exponent(x, mu, sigma):
    #_tensor_sig_inv = K.constant(np.linalg.inv(sigma), dtype='float32')
    #_exponent = K.dot((x-mu), _tensor_sig_inv)
    #_exponent = K.batch_dot(_exponent, (x-mu), axes=1)
    noise_pow = sigma[0][0]
    _exponent = K.square(x - mu)
    _exponent = K.sum(_exponent, axis=2)
    _exponent = -.5*_exponent/noise_pow
    return _exponent

def main():
    #x = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, 1], [2, 1, 0]])
    #x = K.variable(x)
    #leak_upper = loss_taylor_expansion_gm(0, x, 1, 1, 3, .25)
    #print(K.eval(leak_upper))
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape=(3,)))
    model.compile('adam', loss=lambda a, b: loss_taylor_expansion_gm(a, b, 4, 3, 16, .25))
    print("Start training")
    model.fit(np.random.rand(128, 3), np.random.randn(128, 16), epochs=10, batch_size=128)


if __name__ == "__main__":
    main()
