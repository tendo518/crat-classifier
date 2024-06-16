from dataclasses import dataclass

from crat_classifier.dataset.suscape_csv_dataset import (
    suscape_csv_length,
    suscape_num_valid_classes,
)

# class BaseModel(pl.LightningModule):
#     pass


@dataclass
class BaseModelConfig:
    num_classes: int = suscape_num_valid_classes
    num_preds: int = suscape_csv_length
