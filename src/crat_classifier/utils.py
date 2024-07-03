from os import PathLike
from typing import Optional

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike


def norm_stack(arr1, arr2):
    return np.linalg.norm(np.vstack((arr1, arr2)), ord=2, axis=0)


def gradient_for_angle(y: ArrayLike, x: ArrayLike):
    """calculate gradient for periodic angle values

    Args:
        y (ArrayLike): N d
        x (ArrayLike): N d

    Returns:
        NDArray: gradient for given y
    """
    y, x = np.array(y), np.array(x)
    grad1 = np.gradient(y, x)
    y = (y + 90) % 360
    grad2 = np.gradient(y, x)

    return np.where(np.abs(grad1) > np.abs(grad2), grad2, grad1)


class MetricsAccumulator:
    def __init__(
        self,
        num_classes: int,
        class_mapping: Optional[dict] = None,
    ):
        self.num_classes = num_classes
        self.class_mapping = (
            class_mapping if class_mapping is not None else dict(range(num_classes))
        )
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ):
        if valid_mask is not None:
            predicted = predicted[valid_mask]
            targets = targets[valid_mask]
        self.confusion_matrix += np.bincount(
            (targets.numpy() * self.num_classes + predicted.numpy()).flatten(),
            minlength=self.confusion_matrix.size,
        ).reshape(self.num_classes, -1)

    def calculate_metrics(self) -> dict[int, dict]:
        # TP, FP, FN, TN = np.zeros((4, self.num_classes))
        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - TP
        FN = self.confusion_matrix.sum(axis=1) - TP

        with np.errstate(divide="ignore"):
            precisions = np.nan_to_num(TP / (TP + FP))
            recalls = np.nan_to_num(TP / (TP + FN))
            f1s = np.nan_to_num((2 * precisions * recalls) / (precisions + recalls))
            supports = self.confusion_matrix.sum(axis=1)

        metrics = {}
        metrics.update(
            {
                cls_name: {
                    "precision": precisions[k],
                    "recall": recalls[k],
                    "f1": f1s[k],
                    "support": supports[k],
                }
                for k, cls_name in self.class_mapping.items()
            }
        )
        metrics["micro"] = {
            "precision": TP.sum() / (TP.sum() + FP.sum()),
            "recall": TP.sum() / (TP.sum() + FN.sum()),
            "f1": (2 * TP.sum() / (2 * TP.sum() + FP.sum() + FN.sum())),
        }

        metrics["weighted"] = {
            "precision": np.average(precisions, weights=supports),
            "recall": np.average(recalls, weights=supports),
            "f1": np.average(f1s, weights=supports),
        }
        valid_weights = (supports > 0) * 1
        metrics["macro"] = {
            "precision": np.average(precisions, weights=valid_weights),
            "recall": np.average(recalls, weights=valid_weights),
            "f1": np.average(f1s, weights=valid_weights),
        }
        return metrics

    def visualize_confusion_matrix(self, output_path: PathLike | str | None = None):
        classes_name = list(range(self.num_classes))
        if self.class_mapping is not None:
            classes_name = [self.class_mapping[i] for i in classes_name]
        classes_name = [str(cls) for cls in classes_name]
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            self.confusion_matrix,
            xticklabels=classes_name,
            yticklabels=classes_name,
            cmap="YlGnBu",
        )
        plt.xlabel("Ground Truth As")
        plt.ylabel("Predicted As")
        plt.title("confusion matrix")
        if output_path is not None:
            plt.savefig(output_path)

        plt.show()
