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
        self.class_mapping = class_mapping

        if class_mapping is not None:
            assert self.num_classes == len(class_mapping)

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
        else:
            predicted = predicted.view(-1)
            targets = targets.view(-1)
        for true_label, predicted_label in zip(targets, predicted):
            true_label = int(true_label.item())
            predicted_label = int(predicted_label.item())
            self.confusion_matrix[predicted_label][true_label] += 1

    def calculate_metrics(self):
        """return accuracy, recall, f1 metrics for each class"""
        metrics = {}
        TP_count = 0
        FP_count = 0
        FN_count = 0
        precisions = []
        recalls = []
        f1s = []
        supports = []
        for cls in range(self.num_classes):
            TP = self.confusion_matrix[cls, cls]
            FP = self.confusion_matrix[cls, :].sum() - TP
            FN = self.confusion_matrix[:, cls].sum() - TP
            TN = self.confusion_matrix.sum() - (TP + FP + FN)

            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) != 0
                else 0
            )
            support = int(self.confusion_matrix[:, cls].sum())
            metrics[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }

            TP_count += TP
            FP_count += FP
            FN_count += FN
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

        if self.class_mapping is not None:
            metrics = {self.class_mapping[k]: v for k, v in metrics.items()}
        micro_precision = TP_count / (TP_count + FP_count)
        micro_recall = TP_count / (TP_count + FN_count)
        micro_f1 = (2 * micro_precision * micro_recall) / (
            micro_recall + micro_precision
        )
        metrics["micro"] = {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        }

        def weighted(value, weights):
            value, weights = np.array(value), np.array(weights)
            return (value * weights).sum() / weights.sum()

        metrics["weighted"] = {
            "precision": weighted(precisions, supports),
            "recall": weighted(recalls, supports),
            "f1": weighted(f1s, supports),
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
