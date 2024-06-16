# TODO add yaw (需要修改数据部分)

from __future__ import annotations

from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class Trajectory:
    def __init__(self, milestones: ArrayLike, times: ArrayLike) -> None:
        """
        z
        ^  y
        | /
        |/
        +-------> x
        """
        self.milestones = np.array(milestones)
        self.times = np.array(times)

        assert self.milestones.shape[0] >= 2, "#milestones should >= 2"

        self.times.sort()  # inplace sort

        self.rot_interp = Slerp(self.times, R.from_matrix(self.milestones[:, :3, :3]))
        self.trans_inter = CubicSpline(self.times, self.milestones[:, :3, 3])

    def __len__(self) -> int:
        return self.milestones.__len__()

    @property
    def duration(self):
        return self.times[-1] - self.times[0]

    @property
    def start_time(self):
        return self.times[0]

    @property
    def end_time(self):
        return self.times[-1]

    @property
    def fps(self):
        return len(self) / self.duration

    def eval_trans(self, t, **args):
        return self.trans_inter(t, **args)

    def eval_rots(self, t, **args) -> R:
        return self.rot_interp(t, **args)

    def velocity(self, t) -> NDArray:
        return self.eval_trans(t, nu=1)

    def velocity_angle(self, t) -> NDArray:
        v = self.velocity(t)[:3]
        angles = np.degrees(np.arccos(v / np.linalg.norm(v)))
        return angles

    def accelation(self, t):
        return self.eval_trans(t, nu=2)

    @classmethod
    def from_xy(
        cls,
        xy: Sequence,
        times: Sequence[float],
        rel_time=True,
        rel_pose=True,
    ) -> Self:
        xy_np = np.array(xy, dtype=np.float32)
        times = np.array(times)

        if rel_time:
            start_time = min(times)
            times = [time - start_time for time in times]

        if rel_pose:
            initial_pose_inv = np.linalg.inv(milestones[0])
            # print(R.from_matrix(milestones[0][:3, :3]).as_euler("xyz", degrees=True))
            # TODO 用1-2帧的移动方向当作y轴方向
            milestones = [
                np.matmul(milestone, initial_pose_inv) for milestone in milestones
            ]
            # T_21 = np.matmul(milestones[10], np.linalg.inv(milestones[0]))
            # yaw_21 = R.from_matrix(T_21[:3, :3]).as_euler("xyz", degrees=True)
        return cls(milestones=milestones, times=times)

    # @staticmethod
    # def from_txt(txt: str) -> SE3Trajectory:


if __name__ == "__main__":
    traj = Trajectory.from_xy("data/suscape/scene-000025/")  # cross
    physics, _ = traj.inteploate_physics(10)
    for k, v in physics.items():
        print(f"{k}: {v[0:10]}")
    # traj = SE3Trajectory.from_suscape("data/suscape/scene-000358/")  # turn around
    # traj.visualize("output/test.png")
    # traj = SE3Trajectory.from_suscape("data/suscape/scene-000011/")  # turn right
