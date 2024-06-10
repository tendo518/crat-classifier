import torch


class MetricsAccumulator:
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
