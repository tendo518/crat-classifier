from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from crat_classifier.dataset.dataset_utils import collate_fn_dict
from crat_classifier.dataset.suscape_csv_dataset import (
    CSVDataset,
    suscape_id2class,
)
from crat_classifier.trainer import Classifier
from crat_classifier.utils import MetricsAccumulator
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TestConfig:
    """test configuration"""

    test_split: Path = Path("data/suscape/val")
    # val split
    ckpt_path: Path = Path(
        "output/lightning_logs/version_17/checkpoints/epoch=19-step=180.ckpt"
    )
    # ckeckpoint path
    output_root: Path = Path("output")
    # output root dir
    seed: int | None = None
    # random seed, None for not set
    batch_size: int = 64
    # training/valication batch size
    num_workers: int = 8
    # workders for dataloder
    gpu: bool = True
    # gpu to use


def main(configs: TestConfig):
    dataset = CSVDataset(configs.test_split)
    data_loader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        collate_fn=collate_fn_dict,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # Load model with weights
    model = Classifier.load_from_checkpoint(
        checkpoint_path=configs.ckpt_path,
        map_location="cpu",
        # model_config=configs.crat,
        # optimizer_config=configs.optimizer,
    )
    print(model)
    if configs.gpu:
        model.cuda()
    model.eval()

    metric_accul = MetricsAccumulator(
        num_classes=len(suscape_id2class),
        class_mapping=suscape_id2class,
    )

    with torch.no_grad():
        for _, batch in tqdm(enumerate(data_loader)):
            batched_out = model.forward(batch)

            batched_classes = F.softmax(batched_out, dim=-1).argmax(dim=-1).cpu()
            batched_gts = torch.stack(batch["gt"], dim=0).cpu()
            batched_valid_mask = torch.stack(batch["valid_mask"], dim=0).cpu()

            metric_accul.update(
                predicted=batched_classes,
                targets=batched_gts,
                valid_mask=batched_valid_mask,
            )
            print(batched_classes[0], batched_gts[0])

    metrics = metric_accul.calculate_metrics()

    for cls, metric in metrics.items():
        metric_str = [
            f"{metric_name}: {value:.4f}" for metric_name, value in metric.items()
        ]
        print(f"{cls}\n\t{" ".join(metric_str)}")
    metric_accul.visualize_confusion_matrix(output_path="vis.png")


if __name__ == "__main__":
    main(tyro.cli(TestConfig))
