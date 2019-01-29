from keras import layers
from keras import backend as K

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
