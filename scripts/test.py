from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from crat_classifier.dataset.dataset_utils import collate_fn_dict
from crat_classifier.dataset.suscape_csv_dataset import (
    CSVDataset,
    num_classes,
    suscape_id2class,
)
from crat_classifier.crat import ModelConfig, OptimizerConfig, TrajClassifier
from crat_classifier.utils import MetricsAccumulator
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TestConfig:
    # model configuration
    crat: ModelConfig
    # optimizer configuration
    optimizer: OptimizerConfig
    # val split
    test_split: Path = Path("data/suscape/val")
    # ckeckpoint path
    ckpt_path: Path = Path(
        "output/lightning_logs/version_17/checkpoints/epoch=19-step=180.ckpt"
    )
    # output root dir
    output_root: Path = Path("output")
    # training epochs
    num_epochs: int = 1500
    # random seed, None for not set
    seed: int | None = None
    # learning rate
    learning_rate: float = 1e-3
    # training/valication batch size
    batch_size: int = 64
    # workders for dataloder
    num_workers: int = 8
    # gpu to use
    gpu: bool = True


def main(configs: TestConfig):
    dataset = CSVDataset(configs.test_split, configs)
    data_loader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        collate_fn=collate_fn_dict,
        shuffle=False,
        pin_memory=True,
    )

    # Load model with weights
    model = TrajClassifier.load_from_checkpoint(
        checkpoint_path=configs.ckpt_path,
        # model_config=configs.crat,
        # optimizer_config=configs.optimizer,
    )
    print(model)
    if configs.gpu:
        model.cuda()
    model.eval()

    metric_accul = MetricsAccumulator(num_classes)

    with torch.no_grad():
        for batch_index, batch in tqdm(enumerate(data_loader)):
            batched_out = model.forward(batch)

            batched_classes = F.softmax(batched_out, dim=-1).argmax(dim=-1).cpu()
            batched_gts = torch.stack(batch["gt"], dim=0).cpu()

            metric_accul.update(batched_classes, batched_gts)

    metric_accul.calculate_metrics(suscape_id2class)


if __name__ == "__main__":
    main(tyro.cli(TestConfig))
