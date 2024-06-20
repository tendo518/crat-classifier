from collections import defaultdict
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt


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
    def __init__(self, num_classes, ignore_class=None, class_mapping=None):
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.class_mapping = class_mapping

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

        total_correct = 0
        total_samples = 0
        recall_sum = 0
        recall_count = 0

        for cls in range(self.num_classes):
            if cls == self.ignore_class:
                continue

            TP = self.confusion_matrix[cls, cls]
            FP = self.confusion_matrix[cls, :].sum() - TP
            FN = self.confusion_matrix[:, cls].sum() - TP
            TN = self.confusion_matrix.sum() - (TP + FP + FN)

            accuracy = (
                (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
            )
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) != 0
                else 0
            )

            metrics[cls] = {
                "accuracy": accuracy,
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
            total_correct += TP
            total_samples += TP + FP + FN
            recall_sum += recall
            recall_count += 1
        if self.class_mapping is not None:
            metrics = {self.class_mapping[k]: v for k, v in metrics.items()}
        overall_accuracy = total_correct / total_samples if total_samples != 0 else 0
        overall_recall = recall_sum / recall_count if recall_count != 0 else 0
        metrics["overall"] = {"accuracy": overall_accuracy, "recall": overall_recall}
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
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("confusion matrix (density)")
        if output_path is not None:
            plt.savefig(output_path)

        plt.show()
