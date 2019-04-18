import numpy as np
from keras import layers, models
from keras import backend as K

from custom_losses import loss_taylor_expansion_gm


n = 64
k = 4
r = 3
noise_pow = 1./(2*10**(-10/10.))

x = np.random.randint(0, 2, (2**(k+r), k))
y = np.random.randn(2**(k+r), n)
yt = K.variable(y)
print(K.eval(loss_taylor_expansion_gm(yt, yt, k, r, n, noise_pow)))


#m = models.Sequential()
#m.add(layers.Dense(n, activation='relu', input_shape=(k,)))
#m.add(layers.Dense(n, activation='relu'))
#m.compile('adam', loss=lambda x, y: loss_taylor_expansion_gm(x, y, k, r, n, noise_pow))
#m.fit(x, y, batch_size=len(x), epochs=5, verbose=1)
