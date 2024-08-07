from collections import Counter
import glob
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn.functional as F

"""
"9.9.9 Invalid"                             17450
"1.1.1 LeadVehicleConstant"                 966
"1.1.2 LeadVehicleCutOut"                   260
"1.1.3 VehicleCutInAhead"                   1008
"1.1.4 LeadVehicleDecelerating"             259
"1.1.5 LeadVehicleStppoed"                  897
"1.1.6 LeadVehicleAccelerating"             1281
"1.1.7  LeadVehicleWrongDriveway"           180
"1.1.8 PedestrianCrossing"                  18
"1.2.1 RidingLane"                          13
"1.2.1 ObstaclesAhead"                      54
"1.2.2 ObstaclesInLeftLane"                 11
"1.3.1 ObstaclesAhead"                      84
"1.3.2 ObstaclesInRightLane"                47
"2.1.1 StopAtRedLight"                      889
"2.1.2 StopAtYellowLight"                   29
"2.1.4 LeadVehicleStppoed"                  117
"2.1.5 PedestrianCrossing"                  94
"2.1.8 VehiclesCrossing"                    32
"2.2.1 ArrowLightChanged"                   41
"2.2.2 RoundLightChanged"                   211
"2.4.1 NoVehiclesAhead"                     1081
"2.4.2 WithLeadVehicle"                     286
"2.4.3 VehiclesCrossing"                    118
"2.5.1 NoVehiclesAhead"                     493
"2.5.2 WithLeadVehicle"                     220
"2.5.3 VehiclesCrossing"                    125
"2.6.1 NoVehiclesAhead"                     551
"2.6.2 WithLeadVehicle"                     56
"2.6.3 VehiclesCrossing"                    89
"2.7.1 NoVehiclesAhead"                     109
"2.7.3 VehiclesCrossing"                    1
"""
suscape_classes_def = {
    "1.GoStraight": {
        "1.1 InLane": [
            "1.1.1 LeadVehicleConstant",
            "1.1.2 LeadVehicleCutOut",
            "1.1.3 VehicleCutInAhead",
            "1.1.4 LeadVehicleDecelerating",
            "1.1.5 LeadVehicleStppoed",
            "1.1.6 LeadVehicleAccelerating",
            "1.1.7  LeadVehicleWrongDriveway",
            "1.1.8 PedestrianCrossing",
        ],
        "1.2 ChangingLaneLeft": [
            "1.2.1 RidingLane",
            "1.2.1 ObstaclesAhead",
            "1.2.2 ObstaclesInLeftLane",
        ],
        "1.3 ChangingLaneRight": [
            "1.3.1 ObstaclesAhead",
            "1.3.2 ObstaclesInRightLane",
        ],
        "1.4 ChangingTurnLeft": [],
        "1.5 ChangingTurnRight": [],
        "1.6 ChangingStop": [],
        "1.8 Avoidance": [],
        "1.9 RidingLane": [],
    },
    "2. Crossing": {
        "2.1 StopAndWait": [
            "2.1.1 StopAtRedLight",
            "2.1.2 StopAtYellowLight",
            "2.1.4 LeadVehicleStppoed",
            "2.1.5 PedestrianCrossing",
            "2.1.8 VehiclesCrossing",
        ],
        "2.2 Starting": [
            "2.2.1 ArrowLightChanged",
            "2.2.2 RoundLightChanged",
        ],
        "2.4 GoStraight": [
            "2.4.1 NoVehiclesAhead",
            "2.4.2 WithLeadVehicle",
            "2.4.3 VehiclesCrossing",
        ],
        "2.5 TurnLeft": [
            "2.5.1 NoVehiclesAhead",
            "2.5.2 WithLeadVehicle",
            "2.5.3 VehiclesCrossing",
        ],
        "2.6 TurnRight": [
            "2.6.1 NoVehiclesAhead",
            "2.6.2 WithLeadVehicle",
            "2.6.3 VehiclesCrossing",
        ],
        "2.7 UTurn": [
            "2.7.1 NoVehiclesAhead",
            "2.7.3 VehiclesCrossing",
        ],
    },
    "3.Bridge": {
        "3.1 DriveIn": [],
        "3.2 DriveOut": [],
    },
    "4.SlipRoad": {
        "4.1 DriveIn": [],
        "4.2 DriveOut": [],
        "4.3 Driving": [],
    },
}

# def get_classes(classes_level: Literal[1, 2, 3, 0]):
#     if classes_level == 0:
#         return

#     else:


# suscape_hint_classes = ["1.GoStraight", "2. Crossing"]
# suscape_label_classes = [
#     "1.1 InLane",
#     "1.2 ChangingLaneLeft",
#     "1.3 ChangingLaneRight",
#     "1.4 ChangingTurnLeft",
#     "1.5 ChangingTurnRight",
#     "1.6 ChangingStop",
#     "1.8 Avoidance",
#     "1.9 RidingLane",
#     "2.1 StopAndWait",
#     "2.2 Starting",
#     "2.4 GoStraight",
#     "2.5 TurnLeft",
#     "2.6 TurnRight",
#     "2.7 UTurn",
#     # "3.1 DriveIn",
#     # "3.2 DriveOut",
#     # "4.1 DriveIn",
#     # "4.2 DriveOut",
#     # "4.3 Driving",
# ]
# suscape_invalid_hint = "9. Invalid"
# suscape_invalid_label = "9.9 Invalid"

suscape_hint_classes = [
    "1.1 InLane",
    "1.2 ChangingLaneLeft",
    "1.3 ChangingLaneRight",
    # "1.4 ChangingTurnLeft",
    # "1.5 ChangingTurnRight",
    # "1.6 ChangingStop",
    # "1.8 Avoidance",
    # "1.9 RidingLane",
    "2.1 StopAndWait",
    "2.2 Starting",
    "2.4 GoStraight",
    "2.5 TurnLeft",
    "2.6 TurnRight",
    "2.7 UTurn",
]
"""
000001
000010
000100
...
"""
suscape_label_classes = [
    "1.1.1 LeadVehicleConstant",
    "1.1.2 LeadVehicleCutOut",
    "1.1.3 VehicleCutInAhead",
    "1.1.4 LeadVehicleDecelerating",
    "1.1.5 LeadVehicleStppoed",
    "1.1.6 LeadVehicleAccelerating",
    "1.1.7  LeadVehicleWrongDriveway",
    # "1.1.8 PedestrianCrossing",
    # "1.2.1 RidingLane",
    # "1.2.1 ObstaclesAhead",
    # "1.2.2 ObstaclesInLeftLane",
    # "1.3.1 ObstaclesAhead",
    # "1.3.2 ObstaclesInRightLane",
    # "2.1.1 StopAtRedLight",
    # "2.1.2 StopAtYellowLight",
    "2.1.4 LeadVehicleStppoed",
    "2.1.5 PedestrianCrossing",
    "2.1.8 VehiclesCrossing",
    # "2.2.1 ArrowLightChanged",
    # "2.2.2 RoundLightChanged",
    "2.4.1 NoVehiclesAhead",
    "2.4.2 WithLeadVehicle",
    "2.4.3 VehiclesCrossing",
    "2.5.1 NoVehiclesAhead",
    "2.5.2 WithLeadVehicle",
    "2.5.3 VehiclesCrossing",
    "2.6.1 NoVehiclesAhead",
    "2.6.2 WithLeadVehicle",
    "2.6.3 VehiclesCrossing",
    "2.7.1 NoVehiclesAhead",
    # "2.7.3 VehiclesCrossing", 
]
suscape_invalid_hint = "9.9 Invalid"
suscape_invalid_label = "9.9.9 Invalid"

# NOTE add negative label (for non-labeled and not used labels)
suscape_hints2id = {
    k: v for v, k in enumerate(suscape_hint_classes + [suscape_invalid_hint])
}
suscape_class2id = {
    k: v for v, k in enumerate(suscape_label_classes + [suscape_invalid_label])
}
suscape_num_valid_classes = len(suscape_label_classes)

suscape_id2class = {v: k for k, v in suscape_class2id.items()}
suscape_csv_length = 40

suscape_obj_type_mapping = {
    "AV": 0,
    "Vehicle": 1,
    "Pedestrian": 2,
    "Bicycle": 3,
    "Barrier": 4,
}
suscape_num_obj_type = len(suscape_obj_type_mapping)


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, input_folder):
        input_folder = Path(input_folder)
        self.trajs_root = Path(input_folder / "data")
        self.label_root = Path(input_folder / "label")

        self.scene_ids = [
            label_file.stem for label_file in self.label_root.glob("*.csv")
        ]

    def __getitem__(self, idx):
        try:
            return self.extract_data(
                self.trajs_root / f"{self.scene_ids[idx]}.csv",
                self.label_root / f"{self.scene_ids[idx]}.csv",
            )
        except Exception as e:
            print(f"read {self.scene_ids[idx]}.csv error: {e}")
            raise e

    def __len__(self):
        return len(self.scene_ids)

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
        scene_id = int(Path(csv_path).stem)

        city = df["CITY_NAME"][0]
        timestamps = np.sort(np.unique(df["TIMESTAMP"].values))
        trajs = np.stack([df["X"].values, df["Y"].values], axis=-1)
        obj_types = df["OBJECT_TYPE"].values

        ts2step = {ts: i for i, ts in enumerate(timestamps)}
        steps = np.array([ts2step[x] for x in df["TIMESTAMP"]], dtype=np.int64)

        # steps = np.asarray(steps, np.int64)

        objs2idx = df.groupby("TRACK_ID").groups
        obj_keys = list(objs2idx.keys())
        # make sure ego car indexed firstly
        ego_key = "ego"
        obj_keys.remove(ego_key)
        obj_keys = [ego_key] + obj_keys

        all_obj_trajs = []
        all_obj_types = []
        for obj_key in obj_keys:
            obj_indxs = objs2idx[obj_key]
            # # ignore quickly disappearred
            # if len(obj_indxs) < 8:
            #     continue
            obj_trajs = trajs[obj_indxs]
            obj_frames = steps[obj_indxs]

            # padding
            padded_obj_trajs = np.zeros((40, 3))
            padded_obj_trajs[obj_frames, :2] = obj_trajs
            padded_obj_trajs[obj_frames, 2] = 1.0
            all_obj_trajs.append(padded_obj_trajs)
            all_obj_types.append(suscape_obj_type_mapping[obj_types[obj_indxs][0]])

        all_obj_trajs = np.array(all_obj_trajs, np.float32)
        all_obj_types = np.array(all_obj_types, np.int64)
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
        displ = self.get_displ(all_obj_trajs.copy())
        center = all_obj_trajs[:, -1, :2]

        df_classes = pd.read_csv(label_path)
        class_labels = np.array(
            [
                (
                    suscape_class2id[key]
                    if key in suscape_class2id.keys()
                    else suscape_class2id[suscape_invalid_label]
                )
                for key in df_classes["third_class"].to_numpy()
            ]
        )
        valid_mask = class_labels != suscape_class2id[suscape_invalid_label]

        hint = np.array(
            [
                (
                    suscape_hints2id[hint_cls]
                    if hint_cls in suscape_hints2id.keys()
                    else suscape_hints2id[suscape_invalid_hint]
                )
                for hint_cls in df_classes["second_class"].to_numpy()
            ]
        )

        hint_onehot = F.one_hot(
            torch.from_numpy(hint), num_classes=len(suscape_hints2id)
        )
        all_obj_types = F.one_hot(
            torch.from_numpy(all_obj_types), num_classes=suscape_num_obj_type
        )

        return {
            "csv_path": csv_path,
            "label_file": label_path,
            "scene_id": scene_id,
            "city": city,
            # gt
            "gt": class_labels,
            "hint": hint_onehot,
            "valid_mask": valid_mask,
            # data
            "displ": displ,  # TODO: deprecated
            "centers": center,  # TODO: deprecated
            "obj_trajs": all_obj_trajs,
            "obj_types": all_obj_types,
            "velocity": None,
        }


if __name__ == "__main__":
    dataset = CSVDataset("data/suscape_trajs_all")
    for data in dataset:
        pass
