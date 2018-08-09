import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import hamming_loss
from keras import models
from keras import layers
from keras import backend as K

from digcommpy import messages
from gauss_mix_entropy import entropy_gauss_mix_upper

def mutual_info_eve(x, noise_pow=5.):
    #_shape = K.eval(K.shape(x))
    #_shape = x.shape
    _shape = K.int_shape(x)
    #n_comp, n_dim = _shape
    n_dim = _shape[1]
    entr_noise = .5*np.log2(2*np.pi*np.e*noise_pow*n_dim)
    #mu = K.get_value(x)
    mu = x
    print(mu)
    #sigma = [noise_pow*np.eye(n_dim)]#*n_comp
    sigma = noise_pow * np.eye(n_dim)
    entr_gauss_mix = entropy_gauss_mix_upper(mu, sigma)
    #mutual_info = entr_gauss_mix - entr_noise
    #return K.variable(np.array([[mutual_info]]))
    #return x
    return K.variable(np.array([entr_gauss_mix]))


def loss_gauss_mix_entropy(y_true, y_pred, batch_size, dim, noise_pow=.5):
    #return K.variable(np.array([[1]]))
    #return K.max(y_true-y_pred)
    sigma = noise_pow * np.eye(dim)
    entr_gauss_mix = tensor_entropy_gauss_mix_upper(y_true, sigma, batch_size, dim)
    entr_gauss_mix = K.variable(value=entr_gauss_mix)
    return K.mean(entr_gauss_mix - 0)

def tensor_entropy_gauss_mix_upper(mu, sig, batch_size, dim):
    """Calculate the upper bound of the gaussian mixture entropy using the 
    KL-divergence as distance (Kolchinsky et al, 2017)
    """
    #weights = np.ones(batch_size, dtype=float)/batch_size
    weight = 1./batch_size

    outer_sum = 0
    #for idx_i, c_i in enumerate(weights):
    for mu_i in mu:
        print(mu_i)
        inner_sum = 0
        #for idx_j, c_j in enumerate(weights):
        for mu_j in mu:
            _pdf_j = tensor_norm_pdf(mu_i, mu_j, sig)
            inner_sum = weight * K.sum(_pdf_j)
            #inner_sum += weight*_pdf_j
        outer_sum += weight*K.log(inner_sum)

    entropy_kl = dim/2 - outer_sum
    return entropy_kl

def tensor_norm_pdf(x, mu, sigma):
    dim = np.shape(sigma)[0]
    _factor = 1./(np.sqrt((2*np.pi)**dim*np.linalg.det(sigma)))
    _tensor_sig_inv = K.constant(np.linalg.inv(sigma), dtype='float32')
   #_exp = K.exp(-.5*K.transpose(x-mu)*_tensor_sig_inv*(x-mu))
    _exponent = K.dot((x-mu), _tensor_sig_inv)
    #_exponent = K.dot(_exponent, K.transpose(x-mu))
    _exponent = K.batch_dot(_exponent, (x-mu), axes=1)
    _exp = K.exp(-.5*_exponent)
    return _factor*_exp

def create_model(code_length:int =16, info_length: int =4, activation='relu',
                 symmetrical=True):
    rate = info_length/code_length
    nodes_enc = [2*code_length, (info_length+code_length)//2, code_length]
    nodes_dec = []
    train_snr = {'bob': 5., 'eve': 0.}
    train_snr_lin = {k: 10.**(v/10.) for k, v in train_snr.items()}
    input_power = 1.
    train_noise = {k: input_power/(2.*rate*v) for k, v in train_snr_lin.items()}
    noise_layers = {k: layers.GaussianNoise(v, input_shape=(code_length,))
                    for k, v in train_noise.items()}
    if symmetrical:
        nodes_dec = reversed(nodes_enc)
    main_input = layers.Input(shape=(info_length,))
    layer_list_enc = [main_input]
    for idx, _nodes in enumerate(nodes_enc):
        _new_layer = layers.Dense(_nodes, activation=activation)(layer_list_enc[idx])
        layer_list_enc.append(_new_layer)
    noise_layer_bob = noise_layers['bob'](layer_list_enc[-1])
    noise_layer_eve = noise_layers['eve'](layer_list_enc[-1])
    layer_list_decoder = [noise_layer_bob]
    for idx, _nodes in enumerate(nodes_dec):
        _new_layer = layers.Dense(_nodes, activation=activation)(layer_list_decoder[idx])
        layer_list_decoder.append(_new_layer)
    output_layer_bob = layers.Dense(info_length, activation='sigmoid',
                                    name='output_bob')(layer_list_decoder[-1])
    #Append decoder layer to Eve
    #output_layer_eve = layers.Dense(info_length, activation='sigmoid')(noise_layer_eve)
    output_layer_eve = layers.Lambda(
        mutual_info_eve, arguments={'noise_pow': train_noise['eve']},
        output_shape=(1,), name='output_eve')(noise_layer_eve)
    
    model = models.Model(inputs=[main_input],
                         outputs=[output_layer_bob, output_layer_eve])
    model.compile('adam', loss_weights=[1., 1.],
                  loss={'output_bob': 'binary_crossentropy', 'output_eve': 'mse'})
    return model


def main():
    n = 16
    k = 10  # 4
    model = create_model(n, k)
    info_book = messages.generate_data(k, binary=True)
    print("Start training...")
    #target_eve = .5*np.ones(np.shape(info_book))
    target_eve = np.zeros((len(info_book), 1))
    model.fit([info_book], [info_book, target_eve], epochs=1000, verbose=0)
    test_set = messages.generate_data(k, number=100000, binary=True)
    print("Start testing...")
    pred = model.predict(test_set)
    for user_pred in pred:
        pred_bit = np.round(user_pred)
        print("BER: {}".format(hamming_loss(np.ravel(test_set),
                                            np.ravel(pred_bit))))

if __name__ == "__main__":
    main()
