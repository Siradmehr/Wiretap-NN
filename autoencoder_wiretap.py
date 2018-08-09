import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import hamming_loss
from keras import models
from keras import layers
from keras import backend as K

from digcommpy import messages

class AlwaysOnGaussianNoise(layers.GaussianNoise):
    def call(self, inputs, training=None):
        return inputs + K.random_normal(shape=K.shape(inputs),
                                        mean=0.,
                                        stddev=self.stddev)

def loss_gauss_mix_entropy(y_true, y_pred, batch_size, dim, noise_pow=.5):
    #return K.variable(np.array([[1]]))
    #return K.max(y_true-y_pred)
    sigma = noise_pow * np.eye(dim)
    entr_gauss_mix = tensor_entropy_gauss_mix_upper(y_pred, sigma, batch_size, dim)
    entr_gauss_mix = K.repeat_elements(entr_gauss_mix, batch_size, 0)
    entr_noise = .5*np.log((2*np.pi*np.e)**dim*np.linalg.det(sigma))
    return K.mean(entr_gauss_mix - entr_noise, axis=-1)

def tensor_entropy_gauss_mix_upper(mu, sig, batch_size, dim):
    """Calculate the upper bound of the gaussian mixture entropy using the 
    KL-divergence as distance (Kolchinsky et al, 2017)
    """
    #weights = np.ones(batch_size, dtype=float)/batch_size
    weight = 1./batch_size
    #x = K.variable(value=mu)
    x = K.repeat_elements(mu, batch_size, axis=0)
    #x = K.repeat(x, batch_size)
    #mu = K.variable(value=mu)
    mu = K.tile(mu, (batch_size, 1))
    #mu = K.reshape(mu, (batch_size, batch_size, -1))
    norm = tensor_norm_pdf(x, mu, sig)
    norm = K.reshape(norm, (batch_size, batch_size, -1))
    inner_sums = K.sum(weight*norm, axis=1, keepdims=True)
    log_inner = K.log(inner_sums)
    outer_sum = K.sum(weight*log_inner, axis=0)
    entropy_kl = dim/2 - outer_sum
    return entropy_kl

def tensor_norm_pdf(x, mu, sigma):
    dim = np.shape(sigma)[0]
    _factor = 1./(np.sqrt((2*np.pi)**dim*np.linalg.det(sigma)))
    _log_factor = -.5*dim*(np.log(2*np.pi) + np.log(sigma[0,0]))
    _tensor_sig_inv = K.constant(np.linalg.inv(sigma), dtype='float32')
    _exponent = K.dot((x-mu), _tensor_sig_inv)
    #_exponent = K.dot(_exponent, K.transpose(x-mu))
    #_exponent = K.batch_dot(_exponent, (x-mu), axes=2)
    _exponent = K.batch_dot(_exponent, (x-mu), axes=1)
    _exp = K.exp(-.5*_exponent)
    return _factor*_exp
    #return _log_factor-.5*_exponent

def create_model(code_length:int =16, info_length: int =4, activation='relu',
                 symmetrical=True):
    rate = info_length/code_length
    #nodes_enc = [2*code_length, (info_length+code_length)//2, code_length]
    #nodes_enc = [2*code_length, code_length]
    nodes_enc = [2**info_length, 4*code_length, code_length]
    nodes_dec = [4*code_length, 2**info_length]
    #train_snr = {'bob': 0., 'eve': -5.}
    train_snr = {'bob': 10., 'eve': 5.}
    train_snr_lin = {k: 10.**(v/10.) for k, v in train_snr.items()}
    input_power = 1.
    #train_noise = {k: input_power/(2.*rate*v) for k, v in train_snr_lin.items()}
    train_noise = {k: input_power/(v) for k, v in train_snr_lin.items()}
    #noise_layers = {k: layers.GaussianNoise(v, input_shape=(code_length,))
    #                for k, v in train_noise.items()}
    noise_layers = {k: AlwaysOnGaussianNoise(v, input_shape=(code_length,))
                    for k, v in train_noise.items()}
    if symmetrical:
        nodes_dec = reversed(nodes_enc)
    main_input = layers.Input(shape=(info_length,))
    layer_list_enc = [main_input]
    for idx, _nodes in enumerate(nodes_enc):
        _new_layer = layers.Dense(_nodes, activation=activation)(layer_list_enc[idx])
        layer_list_enc.append(_new_layer)
    layer_list_enc.append(layers.BatchNormalization()(layer_list_enc[-1]))
    noise_layer_bob = noise_layers['bob'](layer_list_enc[-1])
    #noise_layer_eve = noise_layers['eve'](layer_list_enc[-1])
    layer_list_decoder = [noise_layer_bob]
    for idx, _nodes in enumerate(nodes_dec):
        _new_layer = layers.Dense(_nodes, activation=activation)(layer_list_decoder[idx])
        layer_list_decoder.append(_new_layer)
    output_layer_bob = layers.Dense(info_length, activation='sigmoid',
                                    name='output_bob')(layer_list_decoder[-1])
    #Append decoder layer to Eve
    #output_layer_eve = layers.Dense(info_length, activation='sigmoid')(noise_layer_eve)
    #output_layer_eve = layers.Lambda(
    #    mutual_info_eve, arguments={'noise_pow': train_noise['eve']},
    #    output_shape=(1,), name='output_eve')(noise_layer_eve)
    
    model = models.Model(inputs=[main_input],
                         outputs=[output_layer_bob, layer_list_enc[-1]])
    model.compile('adam', loss_weights=[.2, .8],
                  loss=['binary_crossentropy', lambda x, y: loss_gauss_mix_entropy(x, y, 2**info_length, code_length, noise_pow=train_noise['eve'])])
                  #loss=['mse', 'mse'])
    #model = models.Model(inputs=[main_input], outputs=[output_layer_bob])
    #model.compile('adam', 'binary_crossentropy')
    return model

def plot_history(history):
    epochs = history.epoch
    hist_loss = history.history
    fig = plt.figure()
    axs = fig.add_subplot(111)
    for loss_name, loss_val in hist_loss.items():
        axs.plot(epochs, loss_val, label=loss_name)
    axs.legend()

def main():
    n = 16
    k = 4
    model = create_model(n, k, symmetrical=True)
    info_book = messages.generate_data(k, binary=True)
    print("Start training...")
    #target_eve = .5*np.ones(np.shape(info_book))
    target_eve = np.zeros((len(info_book), n))
    #target_eve = np.zeros((n,))
    history = model.fit([info_book], [info_book, target_eve], epochs=10000, verbose=0, batch_size=2**k)
    #history = model.fit([info_book], [info_book], epochs=400, verbose=0, batch_size=2**k)
    test_set = messages.generate_data(k, number=10000, binary=True)
    print("Start testing...")
    pred = model.predict(test_set)[0]
    pred_bit = np.round(pred)
    print("BER: {}".format(hamming_loss(np.ravel(test_set),
                                        np.ravel(pred_bit))))
    plot_history(history)
    return history, model

if __name__ == "__main__":
    history, model = main()
    plt.show()
