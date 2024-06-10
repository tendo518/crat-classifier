from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
import tyro
from crat_classifier.dataset.dataset_utils import collate_fn_dict
from crat_classifier.dataset.suscape_csv_dataset import CSVDataset
from crat_classifier.model import CratClassifier, ModelConfig, OptimizerConfig
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    # model configuration
    model: ModelConfig
    # optimizer configuration
    optimizer: OptimizerConfig
    # training split
    train_split: Path = Path("data/suscape/train")
    # val split
    val_split: Path = Path("data/suscape/val")
    # output root dir
    output_root: Path = Path("output")
    # training epochs
    num_epochs: int = 1500
    # random seed, None for not set
    seed: int | None = None
    # learning rate
    batch_size: int = 64
    # workders for dataloder
    num_workers: int = 8
    # gpu to use
    gpus: int = 1


def main(configs: TrainConfig):
    if configs.seed is not None:
        seed_everything(configs.seed)

    train_dataset = CSVDataset(configs.train_split, configs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        collate_fn=collate_fn_dict,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    val_dataset = CSVDataset(configs.val_split, configs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        collate_fn=collate_fn_dict,
        drop_last=True,
        pin_memory=True,
    )

    checkpoint_callback = ModelCheckpoint(monitor="val/acc", save_top_k=5, mode="max")
    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    model = CratClassifier(
        model_config=configs.model, optimizer_config=configs.optimizer
    )
    # load_checkpoint_path = "ckpts/v2e35.ckpt"
    # model = CratPred.load_from_checkpoint(load_checkpoint_path, args=args, strict=False)

    trainer = pl.Trainer(
        default_root_dir=configs.output_root,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        accelerator="gpu",
        devices=configs.gpus,
        max_epochs=configs.num_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=5,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
