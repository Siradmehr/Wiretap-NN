import os.path
import pickle

from joblib import cpu_count
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import hamming_loss
import tensorflow as tf
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import callbacks
from keras import backend as K

from digcommpy import messages, metrics
from digcommpy import information_theory as it

from test_loss_taylor import loss_taylor_expansion_gm

GPU = False
num_cores = cpu_count()
if GPU:
    num_GPU = 1
    num_CPU = 1
else:
    num_GPU = 0
    num_CPU = 1
config =tf.ConfigProto(intra_op_parallelism_threads=num_cores,
    inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
    device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


DIRNAME = "n{n}k{k}r{r}-B{bob}E{eve}T{train}-{ts}"

class TestOnlyGaussianNoise(layers.GaussianNoise):
    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs), mean=0., stddev=self.stddev)
        return K.in_train_phase(inputs, noised, training=training)

class BinaryNoise(layers.Layer):
    def __init__(self, **kwargs):
        super(BinaryNoise, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        return K.in_train_phase(K.round(K.random_uniform(shape=K.shape(inputs))),
                                inputs, training=training)

class SimpleNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleNormalization, self).__init__(**kwargs)
        self.mean = None
        self.std = None

    def call(self, inputs, training=None):
        def _transform(m=None, s=None):
            if m is None:
                m = K.mean(inputs, axis=-1, keepdims=True)
            _shape = K.int_shape(inputs)
            tm = inputs -  K.repeat_elements(m, _shape[1], -1)
            if s is None:
                s = K.var(tm, axis=-1, keepdims=True)
            tmv = tm/(K.sqrt(K.repeat_elements(s, _shape[1], -1)+0.001))
            return tmv, m, s
        def training_transform():
            out, m, s = _transform()
            self.mean = m
            self.std = s
            return out
        def test_transform():
            out, m, s = _transform(self.mean, self.std)
            return out
        return K.in_train_phase(training_transform(), test_transform(), training=training)

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
    #return K.mean(entr_z - entr_zm, axis=-1)/(np.log(2)*k)
    return K.square(K.mean(entr_z - entr_zm, axis=-1)/(np.log(2)*k))

def loss_leakage_upper(y_true, y_pred, k, r, dim, noise_pow=.5):
    batch_size = int(2**(r+k))
    split_len = int(2**r)
    #leak_upper_bound = _leak_new_upper_bound(y_pred, noise_pow, batch_size, r, 1.)
    sigma = noise_pow * np.eye(dim)
    leak_upper_bound = _leak_upper_bound(y_pred, sigma, batch_size, dim, split_len, k)
    #return K.square(leak_upper_bound/(np.log(2)*k))
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
    #return K.square(K.mean(entr_z - entr_zm, axis=-1)/(np.log(2)*k))

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
    #return K.square(K.mean(entr_z - entr_zm, axis=-1)/(np.log(2)*k))

def loss_log_mse(y_true, y_pred, weight=1.):
    return -K.log(weight - losses.mse(y_true, y_pred))

def loss_log_leak(y_true, y_pred, k, r, dim, noise_pow=.5, weight=1):
    split_len = int(2**r)
    batch_size = int(2**(k+r))
    sigma = noise_pow * np.eye(dim)
    leak_upper_bound = _leak_upper_bound(y_pred, sigma, batch_size, dim, split_len, k)
    return -K.log(weight - leak_upper_bound/(np.log(2)*k))

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
    #return K.max(distance_to_centers) - K.min(dist_of_centers)
    return K.max(distance_to_centers)# - K.min(dist_of_centers)

def loss_distance_cluster(y_true, y_pred, k, r, dim):
    codewords = K.reshape(y_pred, (2**k, 2**r, dim))
    cw_repeat = K.repeat_elements(codewords, 2**k, 0)
    cw_tile = K.tile(codewords, (2**k, 1, 1))
    distances = cw_repeat - cw_tile
    distances = K.sum(distances*distances, axis=-1)
    #distances_clusters = K.max(distances, axis=0)
    distances_clusters = K.mean(distances, axis=0)
    return K.mean(distances_clusters)

def loss_distance_leakage_combined(y_true, y_pred, k, r, dim, noise_pow=.5):
    _leak = loss_leakage_upper(y_true, y_pred, k, r, dim, noise_pow=noise_pow)
    _dist = loss_distance_cluster(y_true, y_pred, k, r, dim)
    #return (_dist + _leak*np.log(2)*k)*.5
    return K.sqrt(_dist*_leak)

def create_model(code_length:int =16, info_length: int =4, activation='relu',
                 symmetrical=True, loss_weights=[.5, .5],
                 train_snr={'bob': 0., 'eve': -5.}, random_length=3,
                 nodes_enc=None, nodes_dec=None, test_snr=0., optimizer='adam'):
    rate = info_length/code_length
    if nodes_enc is None:
        nodes_enc = [code_length]
    if nodes_dec is None:
        nodes_dec = []
    train_snr_lin = {k: 10.**(v/10.) for k, v in train_snr.items()}
    input_power = 1.
    train_noise = {k: input_power/(2*v) for k, v in train_snr_lin.items()}
    noise_layers = {k: layers.GaussianNoise(np.sqrt(v), input_shape=(code_length,))
                    for k, v in train_noise.items()}
    if symmetrical:
        nodes_dec = reversed(nodes_enc)
    main_input = layers.Input(shape=(info_length+random_length,))
    #random_input = layers.Input(shape=(random_length,), name='random_input')
    #random_bits = BinaryNoise(input_shape=(random_length,))(random_input)
    #input_layer = layers.concatenate([main_input, random_bits])
    input_layer = main_input
    layer_list_enc = [input_layer]
    for idx, _nodes in enumerate(nodes_enc):
        _new_layer = layers.Dense(_nodes, activation=activation)(layer_list_enc[idx])
        layer_list_enc.append(_new_layer)
    layer_list_enc.append(SimpleNormalization(name='codewords')(layer_list_enc[-1]))
    noise_layer_bob = noise_layers['bob'](layer_list_enc[-1])
    test_noise = 1./(2*10.**(test_snr/10.))
    test_noise_layer = TestOnlyGaussianNoise(np.sqrt(test_noise), input_shape=(code_length,))(noise_layer_bob)
    layer_list_decoder = [test_noise_layer]
    for idx, _nodes in enumerate(nodes_dec):
        _new_layer = layers.Dense(_nodes, activation=activation)(layer_list_decoder[idx])
        layer_list_decoder.append(_new_layer)
    output_layer_bob = layers.Dense(info_length+random_length, activation='sigmoid',
                                    name='output_bob')(layer_list_decoder[-1])
    model = models.Model(inputs=main_input, #inputs=[main_input, random_input],
                         outputs=[output_layer_bob, layer_list_enc[-1]])
    model.compile(optimizer, loss_weights=loss_weights,
                  #loss=['mse', lambda x, y: loss_leakage_gauss_mix(x, y, info_length, random_length, code_length, noise_pow=train_noise['eve'])])
                  #loss=['mse', lambda x, y: loss_leakage_with_gap(x, y, info_length, random_length, code_length, noise_pow=train_noise['eve'])])
                  #loss=['mse', lambda x, y: K.square(loss_leakage_upper(x, y, info_length, random_length, code_length, noise_pow=train_noise['eve']))])
                  #loss=[lambda x, y: loss_log_mse(x, y, weight=1./loss_weights[0]),
                  #      lambda x, y: loss_log_leak(x, y, info_length, random_length, code_length, noise_pow=train_noise['eve'], weight=1./loss_weights[1])])
                  #loss=['mse', lambda x, y: K.square(loss_distance_cluster(x, y, info_length, random_length, code_length))])
                  #loss=['mse', lambda x, y: K.square(loss_hershey_olsen_bound(x, y, info_length, random_length, code_length, train_noise['eve']))])
                  loss=['mse', lambda x, y: K.square(loss_taylor_expansion_gm(x, y, info_length, random_length, code_length, train_noise['eve']))])
                  #loss=['mse', lambda x, y: loss_distance_leakage_combined(x, y, info_length, random_length, code_length, train_noise['eve'])])
    return model

def _ebn0_to_esn0(snr, rate=1.):
    return snr + 10*np.log10(rate)

def loss_weight_sweep(n=16, k=4, train_snr={'bob': 2., 'eve': -5.}, test_snr=0.,
                      random_length=3, loss_weights=[.5, .5], test_size=30000,
                      nodes_enc=None, nodes_dec=None, test_snr_eve=-5.,
                      optimizer_config=None, save_model=False):
    train_snr['eve'] = _ebn0_to_esn0(train_snr['eve'], (k+random_length)/n)
    train_snr['bob'] = _ebn0_to_esn0(train_snr['bob'], (k+random_length)/n)
    test_snr_eve = _ebn0_to_esn0(test_snr_eve, (k+random_length)/n)
    print("Train_SNR (Es/N0): {}".format(train_snr))
    test_snr = _ebn0_to_esn0(test_snr, (k+random_length)/n)
    dirname = DIRNAME.format(n=n, k=k, r=random_length, bob=test_snr,
                             eve=train_snr['eve'], train=train_snr['bob'],
                             ts=np.random.randint(0, 100))
    dirname = os.path.join("results", dirname)
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "config"), 'w') as infile:
        infile.write("Encoder: {}\nDecoder: {}\nOptimizer: {}\n".format(
            nodes_enc, nodes_dec, optimizer_config))
    info_book = messages.generate_data(k, binary=True)
    info_train = messages.generate_data(k+random_length, number=None, binary=True)
    #info_train = info_train[:, :k]
    #rnd_train = np.zeros((2**(random_length+k), random_length))
    #test_info = messages.generate_data(k, number=test_size, binary=True)
    #test_rnd = messages.generate_data(random_length, number=test_size,
    #                                  binary=True)
    #test_set = [test_info, test_rnd]
    test_set = messages.generate_data(k+random_length, number=test_size, binary=True)
    test_info = test_set
    target_eve = np.zeros((len(info_train), n))
    #_weights = np.linspace(0.5, .01, 3)
    #_weights = np.linspace(0.7, 0.2, 7)
    _weights = np.linspace(.8, .2, 4)
    #_weights = np.logspace(-4, -2, 5)
    #_weights = np.logspace(-6, -4, 5)
    loss_weights = [[1.-k, k] for k in reversed(_weights)]
    #loss_weights.append([1, 0])
    #loss_weights = [[0.8675, .1325]]
    results_file = 'lwc-B{bob}E{eve}-T{0}-n{1}-k{2}-r{3}.dat'.format(
                    test_snr, n, k, random_length, **train_snr)
    results_file = os.path.join(dirname, results_file)
    with open(results_file, 'w') as outf:
        outf.write("wB\twE\tBER\tBLER\tLeak\tLoss\n")
    for combination in loss_weights:
        print("Loss weight combination: {}".format(combination))
        optimizer = optimizers.Adam.from_config(optimizer_config)
        #optimizer = optimizers.Adadelta.from_config(optimizer_config)
        model = create_model(n, k, symmetrical=False, loss_weights=combination,
                             train_snr=train_snr, activation='relu',
                             random_length=random_length, nodes_enc=nodes_enc,
                             nodes_dec=nodes_dec, test_snr=test_snr,
                             optimizer=optimizer)
        #print("Start training...")
        checkpoint_path = "checkpoint-{}-{{epoch:d}}.model".format(combination)
        checkpoint_path = os.path.join(dirname, checkpoint_path)
        checkpoint = callbacks.ModelCheckpoint(checkpoint_path, 'loss', save_weights_only=True,
                                               period=5000)
        #history = model.fit([info_train, rnd_train], [info_train, target_eve],
        history = model.fit(info_train, [info_train, target_eve],
                            epochs=300000, verbose=0, shuffle=False, 
                            batch_size=len(info_train), callbacks=[checkpoint])
        print("Finished training...")
        pred = model.predict(test_set)[0] # Noise is added during testing
        pred_bit = np.round(pred)
        ber = metrics.ber(test_info[:, :k], pred_bit[:, :k])
        bler = metrics.bler(test_info[:, :k], pred_bit[:, :k])
        #leak = history.history['codewords_loss'][-1]
        total_loss = history.history['loss'][-1]
        print("BER:\t{}".format(ber))
        print("BLER:\t{}".format(bler))
        print("Loss_E:\t{}".format(history.history['codewords_loss'][-1]))
        print("Loss:\t{}".format(total_loss))
        codebook = _save_codebook(model, k, random_length, combination, dirname)
        energy_symbol = np.var(codebook[2])
        noise_var_eve = energy_symbol/(2*10.**(test_snr_eve/10.))
        leak = calc_wiretap_leakage(codebook[0], codebook[2], noise_var_eve)
        print("Leak UB:\t{}\n".format(leak))
        with open(results_file, 'a') as outf:
            outf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                *combination, ber, bler, leak, total_loss))
        if save_model:
            _model_path = "{}_{}-{}-{}".format(*combination,
                    int(leak/k*100), int(bler*100))
            _model_path = os.path.join(dirname, _model_path)
            model.save(_model_path+".model", include_optimizer=False)
            #with open(_model_path+'.hist', 'wb') as _hist_file:
            #    pickle.dump(history.history, _hist_file)
    return history, model

def _save_codebook(model, info_length, random_length, combination, dirname='.'):
    _all_cw = messages.generate_data(info_length+random_length, binary=True)
    _all_cw = np.array(_all_cw)
    info, rand = _all_cw[:, :info_length], _all_cw[:, info_length:]
    encoder_out_layer = model.get_layer("codewords")
    layer_output_func = K.function([*model.inputs, K.learning_phase()],
                                   [encoder_out_layer.output])
    #codewords = layer_output_func([info, rand, 0])[0]
    codewords = layer_output_func([_all_cw, 0])[0]
    results_file = "codewords-{}.dat".format(combination)
    results_file = os.path.join(dirname, results_file)
    with open(results_file, 'w') as outf:
        outf.write("message\trandom\tcodeword\n")
        for _info, _rand, _cw in zip(info, rand, codewords):
            outf.write("{}\t{}\t{}\n".format(list(_info), list(_rand), list(_cw)))
    return info, rand, codewords

def calc_wiretap_leakage(info, codewords, noise_var):
    entr_z = it.entropy_gauss_mix_upper(codewords, noise_var)
    messages, idx_rev = np.unique(info, axis=0, return_inverse=True)
    entr_zm = []
    for _mess_idx in np.unique(idx_rev):
        _idx = np.where(idx_rev == _mess_idx)[0]
        _relevant_codewords = codewords[_idx]
        entr_zm.append(it.entropy_gauss_mix_lower(_relevant_codewords, noise_var))
    entr_zm = np.mean(entr_zm)
    leak = (entr_z - entr_zm)/np.log(2)
    return leak

if __name__ == "__main__":
    code_length = 16
    combinations = (([code_length], []), ([8*code_length, 4*code_length, code_length], [8*code_length, 4*code_length]))
    #combinations = (([8*code_length, 4*code_length, code_length], [8*code_length, 4*code_length]),)
    #combinations = (([code_length], []),) 
    #combinations = (([16*code_length, code_length], []),)
    for comb in combinations:
            train_snr = {'bob': 0., 'eve': -5.}
            opt_conf = optimizers.Adam(amsgrad=False).get_config()
            #opt_conf = optimizers.Adadelta().get_config()
            history, model = loss_weight_sweep(n=code_length, k=4, train_snr=train_snr,
            test_snr=0., random_length=3, test_size=1e4, nodes_enc=comb[0],
            nodes_dec=comb[1], test_snr_eve=-5., optimizer_config=opt_conf,
            save_model=True)
