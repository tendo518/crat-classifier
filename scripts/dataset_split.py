import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

suscape_classes = [
    "1.1 InLane",
    "1.2 ChangingLaneLeft",
    "1.3 ChangingLaneRight",
    "1.4 ChangingTurnLeft",
    "1.5 ChangingTurnRight",
    "1.6 ChangingStop",
    "1.8 Avoidance",
    "1.9 RidingLane",
    "2.1 StopAndWait",
    "2.2 Starting",
    "2.4 GoStraight",
    "2.5 TurnLeft",
    "2.6 TurnRight",
    "2.7 UTurn",
    # "3.1 DriveIn",
    # "3.2 DriveOut",
    # "4.1 DriveIn",
    # "4.2 DriveOut",
    # "4.3 Driving",
    "9.9 Invalid",
]
suscape_class2id = {k: v for v, k in enumerate(suscape_classes)}


csv_dir = Path("data/suscape-trajs-csv/trajs")
csv_labels_dir = Path("data/suscape-trajs-csv/labels")
output_dir = Path("output/StratifiedSplit")


train_data_dir = output_dir / "train" / "data"
train_label_dir = output_dir / "train" / "label"
val_data_dir = output_dir / "val" / "data"
val_label_dir = output_dir / "val" / "label"


train_data_dir.mkdir(exist_ok=True, parents=True)
train_label_dir.mkdir(exist_ok=True, parents=True)
val_data_dir.mkdir(exist_ok=True, parents=True)
val_label_dir.mkdir(exist_ok=True, parents=True)

val_size = 0.1

all_scene_names = []
stratify_array = []
for label_path in csv_labels_dir.iterdir():
    all_scene_names.append(label_path.stem)
    df = pd.read_csv(label_path)
    cls_ids = df["second_class"].apply(
        lambda x: (
            suscape_class2id["9.9 Invalid"]
            if x not in suscape_class2id.keys()
            else suscape_class2id[x]
        )
    )
    stratify_array.append(cls_ids)


X = np.array(all_scene_names, dtype=str)
y = np.array(stratify_array)

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1)
train_index, val_index = list(msss.split(X, y))[0]

X_train, X_val = X[train_index], X[val_index]
y_train, y_val = y[train_index], y[val_index]

train_unique, train_counts = np.unique(y_train, return_counts=True)
val_unique, val_counts = np.unique(y_val, return_counts=True)

train_percentage = train_counts / np.sum(train_counts)
val_percentage = val_counts / np.sum(val_counts)

print(f"train counter: {dict(zip(train_unique, train_percentage))}")
print(f"val counter: {dict(zip(val_unique, val_percentage))}")

assert np.all(train_unique == val_unique)

for scene in X_train:
    csv_path = csv_dir / f"{scene}.csv"
    csv_label_path = csv_labels_dir / f"{scene}.csv"

    shutil.copy2(csv_path, train_data_dir)
    shutil.copy2(csv_label_path, train_label_dir)

for scene in X_val:
    csv_path = csv_dir / f"{scene}.csv"
    csv_label_path = csv_labels_dir / f"{scene}.csv"

    shutil.copy2(csv_path, val_data_dir)
    shutil.copy2(csv_label_path, val_label_dir)
