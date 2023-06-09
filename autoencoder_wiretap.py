# Autoencoders for flexible wiretap code design in Python
#
# Copyright (C) 2019 Karl-Ludwig Besser
# License: GPL Version 3

import os.path
import pickle
from itertools import product

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

from custom_layers import BinaryNoise, SimpleNormalization, TestOnlyGaussianNoise
from custom_losses import loss_taylor_expansion_gm, loss_leakage_upper, loss_log_taylor, loss_log_mse

GPU = True
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
                  #loss=['mse', lambda x, y: loss_leakage_upper(x, y, info_length, random_length, code_length, noise_pow=train_noise['eve'])])
                  #loss=[lambda x, y: loss_log_mse(x, y, weight=1./loss_weights[0]),
                  #      lambda x, y: loss_log_taylor(x, y, info_length, random_length, code_length, noise_pow=train_noise['eve'], weight=1./loss_weights[1])])
                  loss=['mse', lambda x, y: loss_taylor_expansion_gm(x, y, info_length, random_length, code_length, train_noise['eve'])])
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
    test_set = messages.generate_data(k+random_length, number=test_size, binary=True)
    test_info = test_set
    target_eve = np.zeros((len(info_train), n))
    _weights = np.linspace(0, .1, 5)
    loss_weights = [[1.-k, k] for k in reversed(_weights)]
    #_wb = [.9910875]
    #_we = np.logspace(-4, -1, 5)
    #loss_weights = product(_wb, _we)
    results_file = 'lwc-B{bob}E{eve}-T{0}-n{1}-k{2}-r{3}.dat'.format(
                    test_snr, n, k, random_length, **train_snr)
    results_file = os.path.join(dirname, results_file)
    with open(results_file, 'w') as outf:
        outf.write("wB\twE\tBER\tBLER\tLeak\tLoss\n")
    for combination in loss_weights:
        combination = list(combination)
        print("Loss weight combination: {}".format(combination))
        optimizer = optimizers.Adam.from_config(optimizer_config)
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
        history = model.fit(info_train, [info_train, target_eve],
                            epochs=750, verbose=0, shuffle=False, 
                            batch_size=len(info_train), callbacks=[checkpoint])
        print("Finished training...")
        pred = model.predict(test_set)[0] # Noise is added during testing
        pred_bit = np.round(pred)
        ber = metrics.ber(test_info[:, :k], pred_bit[:, :k])
        bler = metrics.bler(test_info[:, :k], pred_bit[:, :k])
        total_loss = history.history['loss'][-1]
        print("BER:\t{}".format(ber))
        print("BLER:\t{}".format(bler))
        codebook = _save_codebook(model, k, random_length, combination, dirname)
        energy_symbol = np.var(codebook[2])
        noise_var_eve = energy_symbol/(2*10.**(test_snr_eve/10.))
        leak = calc_wiretap_leakage_ub(codebook[0], codebook[2], noise_var_eve)
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
    codewords = layer_output_func([_all_cw, 0])[0]
    results_file = "codewords-{}.dat".format(combination)
    results_file = os.path.join(dirname, results_file)
    with open(results_file, 'w') as outf:
        outf.write("message\trandom\tcodeword\n")
        for _info, _rand, _cw in zip(info, rand, codewords):
            outf.write("{}\t{}\t{}\n".format(list(_info), list(_rand), list(_cw)))
    return info, rand, codewords

def calc_wiretap_leakage_ub(info, codewords, noise_var):
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
    config1 = ([code_length], [])
    config2 = ([8*code_length, 4*code_length, code_length], [8*code_length, 4*code_length])
    config3 = ([16*code_length, code_length], [16*code_length])
    combinations = (config1, config2, config3)
    for comb in combinations:
            train_snr = {'bob': 0., 'eve': -5.}
            opt_conf = optimizers.Adam().get_config()
            try:
                history, model = loss_weight_sweep(n=code_length, k=4, train_snr=train_snr,
                test_snr=0., random_length=3, test_size=1e4, nodes_enc=comb[0],
                nodes_dec=comb[1], test_snr_eve=-5., optimizer_config=opt_conf,
                save_model=True)
            except ValueError:
                print("Error with network: {}".format(comb))
