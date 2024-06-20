import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data

suscape_hints_classes = ["1.GoStraight", "2. Crossing"]
suscape_hints2id = {k: v for v, k in enumerate(suscape_hints_classes)}
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
]
suscape_class2id = {k: v for v, k in enumerate(suscape_classes)}
suscape_num_valid_classes = len(suscape_class2id)

# NOTE negative label (for non-labeled and not used labels)
suscape_invalid_class = "9.9 Invalid"
suscape_class2id.update({suscape_invalid_class: len(suscape_class2id)})
suscape_id2class = {v: k for k, v in suscape_class2id.items()}
suscape_csv_length = 40

# @dataclass
# class DatasetConfig:


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, input_folder, args):
        self.files = sorted(glob.glob(f"{input_folder}/data/*.csv"))
        self.cls_files = sorted(glob.glob(f"{input_folder}/label/*.csv"))
        assert len(self.files) == len(self.cls_files)

    def __getitem__(self, idx):
        return self.extract_data(self.files[idx], self.cls_files[idx])

    def __len__(self):
        return len(self.files)

    @staticmethod
    def get_displ(trajs, reversed=False):
        if reversed:
            trajs = trajs[:, ::-1, :]

        # Compute x and y displacements
        displ_xy = trajs[:, 1:, :2] - trajs[:, :-1, :2]

        # Compute validity flags for the displacements
        valid = trajs[:, :-1, 2] * trajs[:, 1:, 2]

        # Compute angles of the displacements
        angles = np.arctan2(displ_xy[:, :, 1], displ_xy[:, :, 0])
        angles = np.expand_dims(angles, axis=-1)

        # Initialize the displacement array
        displ = np.zeros((trajs.shape[0], trajs.shape[1] - 1, 4), dtype=np.float32)

        # Update the displacement array
        displ[:, :, :2] = displ_xy
        displ[:, :, 2:3] = angles
        displ[:, :, 3] = valid

        # Set displacements to zero where the validity flag is zero
        displ[displ[:, :, 2] == 0, :2] = 0
        displ[displ[:, :, 2] == 0, 2] = 0.0  # Set angle to 0 where validity flag is 0

        return displ

    def extract_data(self, csv_path, label_path):
        df = pd.read_csv(csv_path)
        df_classes = pd.read_csv(label_path)
        argo_id = int(Path(csv_path).stem)

        city = df["CITY_NAME"].values[0]

        timestamps = np.sort(np.unique(df["TIMESTAMP"]))

        ts2frame = {ts: i for i, ts in enumerate(timestamps)}

        trajs = np.stack((df.X.to_numpy(), df.Y.to_numpy()), axis=-1)

        steps = [ts2frame[x] for x in df["TIMESTAMP"].values]
        steps = np.asarray(steps, np.int64)

        objs2indxs = df.groupby("TRACK_ID").groups
        obj_keys = list(objs2indxs.keys())

        # make sure ego car indexed firstly
        ego_key = "0"
        obj_keys.remove(ego_key)
        obj_keys = [ego_key] + obj_keys

        all_trajs = []
        for obj_key in obj_keys:
            obj_indxs = objs2indxs[obj_key]
            # ignore quickly disappearred
            if len(obj_indxs) < 8:
                continue
            obj_trajs = trajs[obj_indxs]
            obj_frames = steps[obj_indxs]

            # padding
            padded_obj_trajs = np.zeros((40, 3))
            padded_obj_trajs[obj_frames, :2] = obj_trajs
            padded_obj_trajs[obj_frames, 2] = 1.0
            all_trajs.append(padded_obj_trajs)

        all_trajs = np.array(all_trajs, np.float32)
        # res_gt = res_trajs[:, 20:].copy()

        # origin = res_trajs[0, 19, :2].copy()

        # rotation = np.eye(2, dtype=np.float32)
        # theta = 0
        # if self.align_image_with_target_x:
        #     pre = res_trajs[0, 19, :2] - res_trajs[0, 18, :2]
        #     theta = np.arctan2(pre[1], pre[0])
        #     rotation = np.asarray([[np.cos(theta), -np.sin(theta)],
        #                      [np.sin(theta), np.cos(theta)]], np.float32)

        # res_trajs[:, :, :2] = np.dot(res_trajs[:, :, :2] - origin, rotation)
        # all_trajs[np.where(all_trajs[:, :, 2] == 0)] = 0
        displ = self.get_displ(all_trajs.copy())
        center = all_trajs[:, -1, :2]
        classes = df_classes["second_class"].to_numpy()
        class_labels = np.array(
            [
                (
                    suscape_class2id[key]
                    if key in suscape_class2id.keys()
                    else suscape_class2id[suscape_invalid_class]
                )
                for key in classes
            ]
        )
        valid_mask = class_labels != suscape_class2id[suscape_invalid_class]

        hint = np.array(
            [
                (
                    suscape_hints2id[hint_cls]
                    if hint_cls in suscape_hints2id.keys()
                    else 2
                )
                for hint_cls in df_classes["first_class"].to_numpy()
            ]
        )

        return {
            "csv_path": csv_path,
            "label_file": label_path,
            "argo_id": argo_id,
            "city": city,
            "gt": class_labels,
            "displ": displ,
            "traj": all_trajs,
            "hint": hint,
            "centers": center,
            "valid_mask": valid_mask,
        }
