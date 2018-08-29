import numpy as np

from digcommpy import decoders
from digcommpy.messages import unpack_to_bits
from map_classifier import MAPClassifier

class MapDecoder(decoders.MachineLearningDecoder):
    def _create_decoder(self, **kwargs):
        return MAPClassifier()

    def _train_system(self, training_code, training_info, training_info_bit,
                      **kwargs):
        _decoder = self.decoder
        _decoder.fit(training_code, np.ravel(training_info))
        return _decoder

    def decode_messages(self, messages, channel=None):
        _pred_info = self.decoder.predict(messages)
        _pred_info_bit = unpack_to_bits(_pred_info, self.info_length)
        return _pred_info_bit
