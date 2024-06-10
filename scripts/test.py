import argparse
from dataclasses import dataclass
import os
from tqdm import tqdm
import re
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tyro
from crat_classifier.dataset.suscape_csv_dataset import (
    CSVDataset,
    suscape_id2class,
    num_classes,
)
from crat_classifier.dataset.dataset_utils import collate_fn_dict
from crat_classifier.model import CratClassifier
from crat_classifier.model import CratClassifier, ModelConfig, OptimizerConfig


class ClassificationMetricsAccumulator:
    def __init__(self, num_classes, ignore_class=19):
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.true_positives = torch.zeros(num_classes)
        self.false_positives = torch.zeros(num_classes)
        self.true_counts = torch.zeros(num_classes)

    def update(self, predicted: torch.Tensor, targets: torch.Tensor):
        for i in range(targets.size(0)):
            true_label = targets[i]
            predicted_label = predicted[i]
            mask = true_label != self.ignore_class
            true_label, predicted_label = true_label[mask], predicted_label[mask]

            self.true_positives[predicted_label] += predicted_label == true_label
            self.false_positives[predicted_label] += predicted_label != true_label
            self.true_counts[true_label] += 1

    def calculate_metrics(self, class_dict):
        overall_accuracy = torch.sum(self.true_positives) / torch.sum(self.true_counts)
        overall_recall = torch.sum(self.true_positives) / (
            torch.sum(self.true_counts) + 1e-8
        )
        recall = self.true_positives / (self.true_counts + 1e-8)

        for class_idx in range(self.num_classes):
            if class_idx != self.ignore_class:
                class_name = class_dict.get(class_idx, f"class{class_idx}")
                accuracy = self.true_positives[class_idx] / (
                    self.true_counts[class_idx] + 1e-8
                )
                print(
                    f"{class_name}: Accuracy: {accuracy.item():.4f}, Recall: {recall[class_idx].item():.4f}"
                )

        print(
            f"Overall Accuracy: {overall_accuracy.item():.4f}, Overall Recall: {overall_recall.item():.4f}"
        )
    

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
        "output/lightning_logs/version_3/checkpoints/epoch=6074-step=230850.ckpt"
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
    model = CratClassifier.load_from_checkpoint(
        checkpoint_path=configs.ckpt_path, model_config=configs.crat, optimizer_config=configs.optimizer
    )
    print(model)
    if configs.gpu:
        model.cuda()
    model.eval()

    metric_accul = ClassificationMetricsAccumulator(num_classes)

    with torch.no_grad():
        for batch_index, batch in tqdm(enumerate(data_loader)):
            batched_out = model.forward(batch)

            batched_classes = F.softmax(batched_out, dim=-1).argmax(dim=-1).cpu()
            batched_gts = torch.stack(batch["gt"], dim=0).cpu()

            metric_accul.update(batched_classes, batched_gts)

    metric_accul.calculate_metrics(suscape_id2class)


if __name__ == "__main__":
    main(tyro.cli(TestConfig))
