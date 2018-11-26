import os.path
import pickle

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import hamming_loss
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import callbacks
from keras import backend as K

from digcommpy import messages, metrics, decoders
from digcommpy import information_theory as it

DIRNAME = "code-n{n}k{k}r{r}-B{bob}E{eve}T{train}-{ts}"

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

def loss_bob_and_eve(y_true, y_pred, k, r, dim, noise_pow_eve=.5,
                     noise_pow_bob=.25, weights=[.5, .5]):
    batch_size = int(2**(r+k))
    split_len = int(2**r)
    sigma_bob = noise_pow_bob * np.eye(dim)
    sigma_eve = noise_pow_eve * np.eye(dim)
    leak_ub_eve = _leak_upper_bound(y_pred, sigma_eve, batch_size, dim, split_len, k)
    mi_lb_bob = _leak_lower_bound(y_pred, sigma_bob, batch_size, dim, split_len, k)
    loss = weights[1]*leak_ub_eve - weights[0]*mi_lb_bob
    return loss/(np.log(2)*k)

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

def create_encoder(code_length:int =16, info_length: int =4, activation='relu',
                   loss_weights=[.5, .5], train_snr={'bob': 0., 'eve': -5.},
                   random_length=3, nodes_enc=None, optimizer='adam'): 
    rate = info_length/code_length
    if nodes_enc is None:
        print("W: Using default encoder nodes")
        nodes_enc = [code_length]
    train_snr_lin = {k: 10.**(v/10.) for k, v in train_snr.items()}
    input_power = 1.
    train_noise = {k: input_power/(2*v) for k, v in train_snr_lin.items()}
    main_input = layers.Input(shape=(info_length,))
    random_input = layers.Input(shape=(random_length,), name='random_input')
    random_bits = BinaryNoise(input_shape=(random_length,))(random_input)
    input_layer = layers.concatenate([main_input, random_bits])
    layer_list_enc = [input_layer]
    for idx, _nodes in enumerate(nodes_enc):
        _new_layer = layers.Dense(_nodes, activation=activation)(layer_list_enc[idx])
        layer_list_enc.append(_new_layer)
    layer_list_enc.append(SimpleNormalization(name='codewords')(layer_list_enc[-1]))
    model = models.Model(inputs=[main_input, random_input],
                         outputs=[layer_list_enc[-1]])
    model.compile(optimizer, loss=lambda x, y: loss_bob_and_eve(x, y, info_length, random_length, code_length, noise_pow_eve=train_noise['eve'], noise_pow_bob=train_noise['bob'],
                  weights=loss_weights))
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
    info_train = info_train[:, :k]
    rnd_train = np.zeros((2**(random_length+k), random_length))
    test_info = messages.generate_data(k, number=test_size, binary=True)
    test_rnd = messages.generate_data(random_length, number=test_size,
                                      binary=True)
    test_set = [test_info, test_rnd]
    target_eve = np.zeros((len(info_train), n))
    #_weights = np.linspace(0.5, .01, 3)
    #_weights = np.linspace(0.7, 0.2, 7)
    _weights = np.linspace(1, 0, 6)
    #_weights = np.logspace(np.log10(.25), -4, 20)
    loss_weights = [[1.-k, k] for k in _weights]
    #loss_weights.append([1, 0])
    #loss_weights = [[0.8675, .1325]]
    results_file = 'codedesign-B{bob}E{eve}-T{0}-n{1}-k{2}-r{3}.dat'.format(
                    test_snr, n, k, random_length, **train_snr)
    results_file = os.path.join(dirname, results_file)
    with open(results_file, 'w') as outf:
        outf.write("wB\twE\tBER\tBLER\tLeak\tLoss\n")
    for combination in loss_weights:
        print("Loss weight combination: {}".format(combination))
        optimizer = optimizers.Adam.from_config(optimizer_config)
        #optimizer = optimizers.Adadelta.from_config(optimizer_config)
        model = create_encoder(n, k, loss_weights=combination,
                             train_snr=train_snr, activation='sigmoid',
                             random_length=random_length, nodes_enc=nodes_enc,
                             optimizer='adam')
        #print("Start training...")
        checkpoint_path = "checkpoint-{}-{{epoch:d}}.model".format(combination)
        checkpoint_path = os.path.join(dirname, checkpoint_path)
        checkpoint = callbacks.ModelCheckpoint(checkpoint_path, 'loss', save_weights_only=True,
                                               period=5000)
        history = model.fit([info_train, rnd_train], target_eve,
                            epochs=999, verbose=0, shuffle=False, 
                            batch_size=len(info_train), callbacks=[checkpoint])
        print("Finished training...")
        total_loss = history.history['loss'][-1]
        codebook = _save_codebook(model, k, random_length, combination, dirname)
        energy_symbol = np.var(codebook[2])
        noise_var_eve = energy_symbol/(2*10.**(test_snr_eve/10.))
        noise_var_bob = energy_symbol/(2*10.**(test_snr/10.))
        leak = calc_wiretap_leakage(codebook[0], codebook[2], noise_var_eve)
        mi_bob = calc_bob_mi(codebook[0], codebook[2], noise_var_bob)
        print("MI Bob LB:\t{}\n".format(mi_bob))
        print("Leak UB:\t{}\n".format(leak))
        print("Loss:\t{}".format(total_loss))
        decoder = decoders.NeuralNetDecoder(n, k, nodes_dec, train_snr=train_snr['bob'])
        decoder.train_system((codebook[2], codebook[0]), epochs=1000)
        test_code = model.predict(test_set)
        pred_info = decoder.decode_messages(test_code)
        pred_info_bit = np.round(pred_info)
        ber = metrics.ber(test_info, pred_info_bit)
        bler = metrics.bler(test_info, pred_info_bit)
        print("BLER:\t{}".format(bler))
        with open(results_file, 'a') as outf:
            outf.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                *combination, ber, bler, mi_bob, leak, total_loss))
        if save_model:
            _model_path = "{}_{}-{}-{}".format(*combination,
                    int(leak/k*100), int(bler*100))
            _model_path = os.path.join(dirname, _model_path)
            model.save(_model_path+".model", include_optimizer=False)
            with open(_model_path+'.hist', 'wb') as _hist_file:
                pickle.dump(history.history, _hist_file)
    return history, model

def _save_codebook(model, info_length, random_length, combination, dirname='.'):
    _all_cw = messages.generate_data(info_length+random_length, binary=True)
    _all_cw = np.array(_all_cw)
    info, rand = _all_cw[:, :info_length], _all_cw[:, info_length:]
    encoder_out_layer = model.get_layer("codewords")
    layer_output_func = K.function([*model.inputs, K.learning_phase()],
                                   [encoder_out_layer.output])
    codewords = layer_output_func([info, rand, 0])[0]
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

def calc_bob_mi(info, codewords, noise_var):
    entr_z = it.entropy_gauss_mix_lower(codewords, noise_var)
    messages, idx_rev = np.unique(info, axis=0, return_inverse=True)
    entr_zm = []
    for _mess_idx in np.unique(idx_rev):
        _idx = np.where(idx_rev == _mess_idx)[0]
        _relevant_codewords = codewords[_idx]
        entr_zm.append(it.entropy_gauss_mix_upper(_relevant_codewords, noise_var))
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
