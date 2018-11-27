from time import time

from joblib import cpu_count
import numpy as np
import tensorflow as tf
from keras import backend as K
from digcommpy import information_theory as it
from scipy.stats import multivariate_normal

from loss_mc_estimation import loss_mutual_info_mc

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

def estimate_differential_entropy(means, noise_var, samples):
    gauss_mix = it.GaussianMixtureRv(means, noise_var)
    _samples = gauss_mix.rvs(samples)
    entropy = gauss_mix.logpdf(_samples)
    entropy = -np.mean(entropy)
    return entropy

def estimate_leakage_from_codebook(means, noise_var, samples):
    h_z = estimate_differential_entropy(means, noise_var, samples)
    h_zm = []
    info_length = 1
    r = 1
    for message in range(2**info_length):
        _means = means[message*(2**r):(message+1)*(2**r), :]
        h_zm.append(estimate_differential_entropy(_means, noise_var, samples))
    h_zm = np.mean(h_zm)
    return (h_z - h_zm)/np.log(2)

num_samples = int(1e4)
means = np.array([[1, 2, 3], [-1, -2, -3], [4, 5, 6], [-7, -8, -9]])
noise_pow = 2.6

start1 = time()
est_leak_numpy = estimate_leakage_from_codebook(means, noise_pow, num_samples)
end1 = time()
print(est_leak_numpy)
print("Numpy:\t{}".format(end1-start1))

means = K.variable(means, dtype='float32')
start2 = time()
est_leak = loss_mutual_info_mc(0, means, 1, 1, 3, noise_pow, num_samples=num_samples)
end2 = time()
print(K.eval(est_leak))
print("Keras:\t{}".format(end2-start2))

#samples = K.eval(samples)
#means = K.eval(means)
#gm = it.GaussianMixtureRv(means, sigma=noise_pow)
#print(gm.logpdf(samples))
#print(np.vstack([multivariate_normal.logpdf(samples, mean=_m, cov=noise_pow*np.eye(3)) for _m in means]).T)
