import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import losses

def loss_leakage_gauss_mix(y_true, y_pred, k, r, dim, noise_pow=.5):
    split_len = int(2**r)
    batch_size = int(2**(r+k))
    sigma = noise_pow * np.eye(dim)
    entr_z = tensor_entropy_gauss_mix_upper(y_pred, sigma, batch_size, dim)
    entr_z = K.repeat_elements(entr_z, batch_size, 0)
    entr_zm = []
    for i in range(2**k):
        _y_pred_message = y_pred[i*split_len:(i+1)*split_len, :]
        _entr_zm = tensor_entropy_gauss_mix_lower(_y_pred_message, sigma, split_len, dim)
        entr_zm.append(_entr_zm)
    entr_zm = K.concatenate(entr_zm, axis=0)
    entr_zm = K.mean(entr_zm, axis=0)
    return K.square(K.mean(entr_z - entr_zm, axis=-1)/(np.log(2)*k))

def loss_leakage_upper(y_true, y_pred, k, r, dim, noise_pow=.5):
    batch_size = int(2**(r+k))
    split_len = int(2**r)
    sigma = noise_pow * np.eye(dim)
    leak_upper_bound = _leak_upper_bound(y_pred, sigma, batch_size, dim, split_len, k)
    return leak_upper_bound/(np.log(2)*k)

def _leak_new_upper_bound(mu, noise_power, batch_size, r, eps):
    weight = 1./batch_size
    p_i = K.repeat_elements(mu, batch_size, axis=0)
    p_j = K.tile(mu, (batch_size, 1))
    kl = 1./(2*noise_power)*K.batch_dot(p_i-p_j, p_i-p_j, axes=1)
    kl = K.reshape(kl, (batch_size, batch_size, -1))
    log_inner = np.log(weight) + K.logsumexp(-kl, axis=1, keepdims=True)
    outer_sum = K.sum(weight*log_inner, axis=0)
    mi_upper = -outer_sum - r + eps
    return mi_upper

def loss_leakage_with_gap(y_true, y_pred, k, r, dim, noise_pow=.5):
    split_len = int(2**r)
    batch_size = int(2**(r+k))
    sigma = noise_pow * np.eye(dim)
    leak_upper_bound = _leak_upper_bound(y_pred, sigma, batch_size, dim, split_len, k)
    leak_lower_bound = _leak_lower_bound(y_pred, sigma, batch_size, dim, split_len, k)
    return .5*(K.square(leak_upper_bound) + leak_upper_bound - leak_lower_bound)

def _leak_upper_bound(y_pred, sigma, batch_size, dim, split_len, k):
    entr_z = tensor_entropy_gauss_mix_upper(y_pred, sigma, batch_size, dim)
    entr_z = K.repeat_elements(entr_z, batch_size, 0)
    entr_zm = []
    for i in range(2**k):
        _y_pred_message = y_pred[i*split_len:(i+1)*split_len, :]
        _entr_zm = tensor_entropy_gauss_mix_lower(_y_pred_message, sigma, split_len, dim)
        entr_zm.append(_entr_zm)
    entr_zm = K.concatenate(entr_zm, axis=0)
    entr_zm = K.mean(entr_zm, axis=0)
    return K.mean(entr_z - entr_zm, axis=-1)/(np.log(2)*k)

def _leak_lower_bound(y_pred, sigma, batch_size, dim, split_len, k):
    entr_z = tensor_entropy_gauss_mix_lower(y_pred, sigma, batch_size, dim)
    entr_z = K.repeat_elements(entr_z, batch_size, 0)
    entr_zm = []
    for i in range(2**k):
        _y_pred_message = y_pred[i*split_len:(i+1)*split_len, :]
        _entr_zm = tensor_entropy_gauss_mix_upper(_y_pred_message, sigma, split_len, dim)
        entr_zm.append(_entr_zm)
    entr_zm = K.concatenate(entr_zm, axis=0)
    entr_zm = K.mean(entr_zm, axis=0)
    return K.relu(K.mean(entr_z - entr_zm, axis=-1))/(np.log(2)*k)

def loss_log_mse(y_true, y_pred, weight=1.):
    return -K.log(weight - losses.mse(y_true, y_pred))

def loss_log_leak(y_true, y_pred, k, r, dim, noise_pow=.5, weight=1):
    split_len = int(2**r)
    batch_size = int(2**(k+r))
    sigma = noise_pow * np.eye(dim)
    leak_upper_bound = _leak_upper_bound(y_pred, sigma, batch_size, dim, split_len, k)
    return -K.log(weight - leak_upper_bound/(np.log(2)*k))

def loss_log_taylor(y_true, y_pred, k, r, dim, noise_pow, weight=1.):
    return -K.log(weight - loss_taylor_expansion_gm(y_true, y_pred, k, r, dim, noise_pow))

def loss_gauss_mix_entropy(y_true, y_pred, batch_size, dim, noise_pow=.5, k=1):
    sigma = noise_pow * np.eye(dim)
    entr_gauss_mix = tensor_entropy_gauss_mix_upper(y_pred, sigma, batch_size, dim)
    entr_gauss_mix = K.repeat_elements(entr_gauss_mix, batch_size, 0)
    entr_noise = .5*dim*np.log(2*np.pi*np.e*noise_pow)
    return K.mean(entr_gauss_mix - entr_noise, axis=-1)/(np.log(2)*k)

def loss_hershey_olsen_bound(y_true, y_pred, k, r, dim, noise_pow=.5):
    batch_size = int(2**(r+k))
    split_len = int(2**r)
    #sigma = noise_pow * np.eye(dim)
    results = []
    y_pred_rep = K.repeat_elements(y_pred, split_len, 0)
    for message in range(2**k):
        _y_pred_message = y_pred[message*split_len:(message+1)*split_len, :]
        _y_pred_message = K.tile(_y_pred_message, (batch_size, 1))
        _kl = 1./(2*noise_pow)*K.batch_dot(_y_pred_message-y_pred_rep, _y_pred_message-y_pred_rep, axes=1)
        results.append(_kl)
    results = K.concatenate(results, axis=0)
    result = K.mean(results, axis=0)
    result = result/(batch_size*split_len)
    return result

def _kl_gaussian(mu, noise_power, batch_size, r, eps):
    weight = 1./batch_size
    p_i = K.repeat_elements(mu, batch_size, axis=0)
    p_j = K.tile(mu, (batch_size, 1))
    kl = 1./(2*noise_power)*K.batch_dot(p_i-p_j, p_i-p_j, axes=1)
    kl = K.reshape(kl, (batch_size, batch_size, -1))
    log_inner = np.log(weight) + K.logsumexp(-kl, axis=1, keepdims=True)
    outer_sum = K.sum(weight*log_inner, axis=0)
    mi_upper = -outer_sum - r + eps
    return mi_upper

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


def tensor_entropy_gauss_mix_upper(mu, sig, batch_size, dim=None):
    """Calculate the upper bound of the gaussian mixture entropy using the 
    KL-divergence as distance (Kolchinsky et al, 2017)
    """
    weight = 1./batch_size
    x = K.repeat_elements(mu, batch_size, axis=0)
    mu = K.tile(mu, (batch_size, 1))
    dim = np.shape(sig)[0]
    _factor_log = -(dim/2)*np.log(2*np.pi*sig[0, 0])
    norm_exp = tensor_norm_pdf_exponent(x, mu, sig)
    norm_exp = K.reshape(norm_exp, (batch_size, batch_size, -1))
    log_inner = np.log(weight) + _factor_log + K.logsumexp(norm_exp, axis=1, keepdims=True)
    outer_sum = K.sum(weight*log_inner, axis=0)
    entropy_kl = dim/2 - outer_sum
    return entropy_kl

def tensor_entropy_gauss_mix_lower(mu, sig, batch_size, dim=None, alpha=.5):
    """Calculate the upper bound of the gaussian mixture entropy using the 
    KL-divergence as distance (Kolchinsky et al, 2017)
    """
    sig = sig/(alpha*(1.-alpha))
    weight = 1./batch_size
    x = K.repeat_elements(mu, batch_size, axis=0)
    mu = K.tile(mu, (batch_size, 1))
    dim = np.shape(sig)[0]
    _factor_log = -(dim/2)*np.log(2*np.pi*sig[0, 0])
    norm_exp = tensor_norm_pdf_exponent(x, mu, sig)
    norm_exp = K.reshape(norm_exp, (batch_size, batch_size, -1))
    log_inner = np.log(weight) + _factor_log + K.logsumexp(norm_exp, axis=1, keepdims=True)
    outer_sum = K.sum(weight*log_inner, axis=0)
    entropy = dim/2 + dim/2*np.log(alpha*(1.-alpha)) - outer_sum
    return entropy

def tensor_norm_pdf_exponent(x, mu, sigma):
    _tensor_sig_inv = K.constant(np.linalg.inv(sigma), dtype='float32')
    _exponent = K.dot((x-mu), _tensor_sig_inv)
    _exponent = K.batch_dot(_exponent, (x-mu), axes=1)
    _exponent = -.5*_exponent
    return _exponent

def tensor_norm_pdf(x, mu, sigma):
    dim = np.shape(sigma)[0]
    _factor = 1./(np.sqrt((2*np.pi)**dim*np.linalg.det(sigma)))
    _tensor_sig_inv = K.constant(np.linalg.inv(sigma), dtype='float32')
    _exponent = K.dot((x-mu), _tensor_sig_inv)
    _exponent = K.batch_dot(_exponent, (x-mu), axes=1)
    _exp = K.exp(-.5*_exponent)
    return _factor*_exp

def loss_distance_cluster_mean(y_true, y_pred, k, r, dim):
    codewords = K.reshape(y_pred, (2**k, 2**r, dim))
    cluster_centers = K.mean(codewords, axis=0, keepdims=True)
    distance_to_centers = codewords - cluster_centers
    distance_to_centers = K.sum(distance_to_centers*distance_to_centers, axis=-1)
    dist_of_centers = K.sum(cluster_centers*cluster_centers, axis=1)
    return K.max(distance_to_centers)

def loss_distance_cluster(y_true, y_pred, k, r, dim):
    codewords = K.reshape(y_pred, (2**k, 2**r, dim))
    cw_repeat = K.repeat_elements(codewords, 2**k, 0)
    cw_tile = K.tile(codewords, (2**k, 1, 1))
    distances = cw_repeat - cw_tile
    distances = K.sum(distances*distances, axis=-1)
    distances_clusters = K.mean(distances, axis=0)
    return K.mean(distances_clusters)

def loss_distance_leakage_combined(y_true, y_pred, k, r, dim, noise_pow=.5):
    _leak = loss_leakage_upper(y_true, y_pred, k, r, dim, noise_pow=noise_pow)
    _dist = loss_distance_cluster(y_true, y_pred, k, r, dim)
    return K.sqrt(_dist*_leak)

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
    x = K.repeat(mu, num_comp)
    mu = K.permute_dimensions(x, (1, 0, 2))
    cap_f = _gm_capital_f(x, mu, sigma, num_comp) # X x N x N
    cap_f_c = tf.linalg.trace(cap_f) # X
    _result = K.mean(cap_f_c)
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
    _grad = tensor_gradient_gm(x, mu, sigma, num_comp) # X x N
    _grad = K.expand_dims(_grad, axis=1) # X x 1 x N
    _grad = K.tile(_grad, (1, num_comp, 1)) # X x M x N
    _grad = K.expand_dims(_grad, axis=3)
    _xm = K.expand_dims(x-mu, axis=3)
    _part1 = K.batch_dot(_xm, _grad, axes=3)  # X x M x N x N
    _pdf_gm = K.expand_dims(K.expand_dims(K.expand_dims(pdf_gm)))
    _part1 = _part1/_pdf_gm  # X x M x N x N
    _part2 = K.batch_dot(_xm, _xm, axes=3)/noise_pow
    _eye = K.expand_dims(K.expand_dims(K.eye(dim), 0), 0)
    _result = _part1 + _part2 - _eye
    _norm = K.expand_dims(K.expand_dims(_tensor_norm_pdf(x, mu, sigma)))
    _result = _result * _norm # X x M x N x N
    _result = K.mean(_result, axis=1) # X x N x N
    result = _result/(K.expand_dims(K.expand_dims(pdf_gm))*noise_pow)
    return result

def tensor_gradient_gm(x, mu, sigma, num_comp):
    noise_pow = sigma[0][0]
    _grad = (x-mu)*K.expand_dims(_tensor_norm_pdf(x, mu, sigma))
    grad = K.mean(_grad, axis=1) # 
    return grad

def _tensor_gm_pdf(x, mu, sigma, num_comp):
    dim = np.shape(sigma)[0]
    _pdf_exp = _tensor_norm_pdf_exponent(x, mu, sigma)
    factor = (1./num_comp)/(np.sqrt(2*np.pi*sigma[0][0])**dim)
    pdf = factor * K.sum(K.exp(_pdf_exp), axis=1)
    return pdf

def tensor_gm_logpdf(x, mu, sigma, num_comp):
    dim = np.shape(sigma)[0]
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
    noise_pow = sigma[0][0]
    _exponent = K.square(x - mu)
    _exponent = K.sum(_exponent, axis=2)
    _exponent = -.5*_exponent/noise_pow
    return _exponent
