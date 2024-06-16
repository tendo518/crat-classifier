from .crat import CratTrajClassifier
from .rnn import RNNClassifier
from .seq2seq import EncoderDecoderClassifier

model_dict = {
    "rnn": RNNClassifier,
    "crat": CratTrajClassifier,
    "seq2seq": EncoderDecoderClassifier,
}


def build_model(config):
    return model_dict[config.model_name](
        config=config.__getattribute__(config.model_name),
    )
