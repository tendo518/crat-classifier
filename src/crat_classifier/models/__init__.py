from .rnn import RNNClassifier
from .st_gnn import STGraphyClassifier

model_dict = {
    "rnn": RNNClassifier,
    "st_gnn": STGraphyClassifier,
}


def build_model(config):
    return model_dict[config.model_name](
        config=config.__getattribute__(config.model_name),
    )
