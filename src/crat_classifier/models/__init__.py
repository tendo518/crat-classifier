from .crat import CratTrajClassifier
from .rnn import RNNClassifier
from .seq2seq import EncoderDecoderClassifier

model_dict = {
    "rnn": RNNClassifier,
    "crat": CratTrajClassifier,
    "seq2seq": EncoderDecoderClassifier,
}


def get_model(config):
    model_cls = model_dict[config.model_name]
    return model_cls(config=config.__getattribute__(config.model_name))
