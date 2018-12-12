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
    result = []
    for component in range(num_comp):
        mu_i = K.expand_dims(mu[component, :], axis=0)
        _cap_f = _gm_capital_f(mu_i, mu, sigma, num_comp)
        result.append(tf.linalg.trace(_cap_f))
    result = K.stack(result, axis=0)
    result = K.mean(result)
    result = entr_zero_ord - .5*noise_pow*result
    return result

def _entropy_gm_taylor_zero(mu, sigma, num_comp):
    log_pdfs = []
    for message in range(num_comp):
        mu_i = K.expand_dims(mu[message, :], axis=0)
        _log_pdf = tensor_gm_logpdf(mu_i, mu, sigma, num_comp)
        log_pdfs.append(_log_pdf)
    log_pdfs = K.stack(log_pdfs, axis=0)
    return -K.mean(log_pdfs)

def _gm_capital_f(x, mu, sigma, num_comp):
    noise_pow = sigma[0][0]
    dim = np.shape(sigma)[0]
    pdf_gm = tensor_gm_pdf(x, mu, sigma, num_comp)
    results = []
    for component in range(num_comp):
        mu_i = K.expand_dims(mu[component, :], axis=0)
        _pdf_norm = tensor_norm_pdf(x, mu_i, sigma)
        _grad = tensor_gradient_gm(x, mu, sigma, num_comp)
        _part1 = K.dot(tf.linalg.transpose(x-mu_i), _grad)
        _part2 = 1./noise_pow * K.dot(tf.linalg.transpose(x-mu_i), x-mu_i)
        _result = (_part1 + _part2 - K.eye(dim))*_pdf_norm
        results.append(_result)
    results = K.stack(results, axis=2)
    result = K.mean(results, axis=2)
    return result/(pdf_gm*noise_pow)

def tensor_gradient_gm(x, mu, sigma, num_comp):
    noise_pow = sigma[0][0]
    result = []
    for component in range(num_comp):
        mu_i = K.expand_dims(mu[component, :], axis=0)
        _grad = (x-mu_i)*tensor_norm_pdf(x, mu_i, sigma)
        result.append(_grad)
    result = K.stack(result, axis=-1)
    #result = K.concatenate(result, axis=0)
    result = K.mean(result, axis=-1)
    return result

def tensor_gm_pdf(x, mu, sigma, num_comp):
    dim = np.shape(sigma)[0]
    x = K.repeat_elements(x, num_comp, 0)
    _pdf_exp = tensor_norm_pdf_exponent(x, mu, sigma)
    factor = (1./num_comp)/(np.sqrt(2*np.pi*sigma[0][0])**dim)
    pdf = factor * K.sum(K.exp(_pdf_exp))
    return pdf

def tensor_gm_logpdf(x, mu, sigma, num_comp):
    dim = np.shape(sigma)[0]
    x = K.repeat_elements(x, num_comp, 0)
    _pdf_exp = tensor_norm_pdf_exponent(x, mu, sigma)
    factor = K.constant((1./num_comp)/(np.sqrt(2*np.pi*sigma[0][0])**dim), dtype='float32')
    pdf = K.log(factor) + K.logsumexp(_pdf_exp)
    return pdf

def tensor_norm_pdf(x, mu, sigma):
    dim = np.shape(sigma)[0]
    _pdf_exp = tensor_norm_pdf_exponent(x, mu, sigma)
    factor = 1./(np.sqrt(2*np.pi*sigma[0][0])**dim)
    pdf = factor * K.sum(K.exp(_pdf_exp))
    return pdf

def tensor_norm_pdf_exponent(x, mu, sigma):
    _tensor_sig_inv = K.constant(np.linalg.inv(sigma), dtype='float32')
    _exponent = K.dot((x-mu), _tensor_sig_inv)
    _exponent = K.batch_dot(_exponent, (x-mu), axes=1)
    _exponent = -.5*_exponent
    return _exponent

def main():
    x = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, 1], [2, 1, 0]])
    #x = K.variable(x)
    #leak_upper = loss_taylor_expansion_gm(0, x, 1, 1, 3, .25)
    #print(K.eval(leak_upper))
    model = models.Sequential()
    model.add(layers.Dense(5, input_shape=(3,)))
    model.compile('adam', loss=lambda a, b: loss_taylor_expansion_gm(a, b, 1, 1, 5, .25))
    model.fit(x, np.random.randn(4, 5), epochs=4)


if __name__ == "__main__":
    main()
