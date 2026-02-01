#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from gym_hil.envs.kuka_window_assembly_env import KukaWindowAssemblyEnv
from gym_hil.mujoco_gym_env import GymRenderingSpec

# 自定义机械臂XYZ工作空间范围
# 格式：[[X_min, Y_min, Z_min], [X_max, Y_max, Z_max]]
# 当前设置：X: 0.0-0.9, Y: -0.4-0.0, Z: 0-0.44
_CUSTOM_CARTESIAN_BOUNDS = np.asarray([[0.0, -0.4, 0], [0.9, 0.4, 0.44]])  # 可以根据需要修改

# 自定义窗户（plate）随机采样区域
# 格式：[[X_min, Y_min], [X_max, Y_max]]
# 当前设置：X: 0.3-0.6, Y: -0.15-0.15
# 可以根据需要修改这个区域，例如：
# - 更大的区域：[[0.2, -0.2], [0.7, 0.2]]
# - 更小的区域：[[0.4, -0.1], [0.5, 0.1]]
_CUSTOM_SAMPLING_BOUNDS = np.asarray([[0.4, -0.3], [0.6, 0.3]])  # 可以根据需要修改

# 自定义 ry 旋转范围限制（绕Y轴的旋转角度范围，单位：弧度）
# 格式：[ry_min, ry_max]
# 例如：[0, π/2] 表示允许从 0 度到 90 度的旋转
# 设置为 None 表示不限制
_RY_ROTATION_BOUNDS = [0.0, np.pi / 2]  # 0 到 90 度，可以根据需要修改

# 自定义玻璃初始 orientation 的 rz 随机范围（绕Z轴的旋转角度范围，单位：弧度）
# 格式：[rz_min, rz_max]
# 例如：[-π/12, π/12] 表示允许 -15 度到 +15 度的旋转
# 设置为 None 表示不随机化 rz（使用固定值 0）
# 注意：对于平放在地面的窗户，通常只需要小角度变化（±15度到±30度）来增加任务多样性
_RANDOM_RZ_BOUNDS = [-np.pi / 12, np.pi / 12]  # -15 度到 +15 度，可以根据需要修改


class RandomKukaWindowAssemblyEnv(KukaWindowAssemblyEnv):
    """Environment for a KUKA iiwa14 robot with vacuum gripper inserting a window into a wall slot.
    
    This is a variant of KukaWindowAssemblyEnv with:
    - Random window position (uses custom sampling bounds)
    - No A-frame (uses kuka_window_assembly_scene_simple.xml)
    - Custom cartesian bounds for robot workspace
    - All other behavior is identical to the parent class
    
    The task is to:
        1. Pick up the window from the ground using the vacuum gripper
        2. Insert the window into a rectangular slot in the facade wall
        3. Align the window properly (position and orientation) within the slot
    """
    
    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        xml_path: Path | None = None,
    ):
        # Force random_window_position=False (fixed position) and use simple XML (no A-frame)
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "assets" / "kuka_window_assembly_scene_simple.xml"
        
        # 注意：父类 KukaWindowAssemblyEnv 不接受 cartesian_bounds 参数
        # 所以我们需要在初始化后修改 self._cartesian_bounds
        # Call parent with random_window_position=True (enable random plate position)
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            reward_type=reward_type,
            random_window_position=True,  # Enable random plate position
            xml_path=xml_path,
        )
        
        # 自定义机械臂XYZ工作空间范围（在父类初始化后修改）
        # 格式：[[X_min, Y_min, Z_min], [X_max, Y_max, Z_max]]
        # 例如：X: 0.0-0.9, Y: -0.4-0.0, Z: 0-0.44
        self._cartesian_bounds = _CUSTOM_CARTESIAN_BOUNDS.copy()
        
        # 保存自定义采样区域（用于reset方法）
        self._custom_sampling_bounds = _CUSTOM_SAMPLING_BOUNDS.copy()
        
        # 保存 ry 旋转范围限制
        self._ry_rotation_bounds = _RY_ROTATION_BOUNDS.copy() if _RY_ROTATION_BOUNDS is not None else None
        
        # 保存 rz 随机范围
        self._random_rz_bounds = _RANDOM_RZ_BOUNDS.copy() if _RANDOM_RZ_BOUNDS is not None else None
    
    def _snap_window_to_gripper(self):
        """当gripper开启且玻璃足够近时，自动对齐玻璃到gripper中心
        
        与父类保持一致，考虑玻璃厚度偏移，并增加偏移量以避免穿模
        """
        # 获取gripper位置
        if self._gripper_dock_site_id is not None:
            gripper_pos = self._data.site(self._gripper_dock_site_id).xpos.copy()
            gripper_mat = self._data.site(self._gripper_dock_site_id).xmat.reshape(3, 3).copy()
        else:
            try:
                attachment_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                gripper_pos = self._data.site(attachment_site_id).xpos.copy()
                gripper_mat = self._data.site(attachment_site_id).xmat.reshape(3, 3).copy()
            except:
                return
        
        # 计算目标位置：玻璃表面对齐到gripper中心（考虑玻璃厚度和偏移以避免穿模）
        # 与父类保持一致，但增加偏移量以确保不穿模
        gripper_z_axis = gripper_mat[:, 2]
        # 使用更大的偏移量：window_thickness (0.005) + 0.03 = 0.035m，确保玻璃表面与吸盘有足够间隙
        # 这样可以避免玻璃与gripper穿模
        target_pos = gripper_pos - (self._window_thickness + 0.03) * gripper_z_axis
        target_mat = gripper_mat
        
        # 转换为四元数
        target_quat = R.from_matrix(target_mat).as_quat()
        target_quat_mujoco = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
        
        # 设置玻璃位置和姿态
        window_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "window_joint")
        qpos_adr = self._model.jnt_qposadr[window_joint_id]
        qvel_adr = self._model.jnt_dofadr[window_joint_id]
        
        # 保存旧位置用于碰撞检测
        old_pos = self._data.qpos[qpos_adr:qpos_adr+3].copy()
        old_quat = self._data.qpos[qpos_adr+3:qpos_adr+7].copy()
        
        # 设置新位置
        self._data.qpos[qpos_adr:qpos_adr+3] = target_pos
        self._data.qpos[qpos_adr+3:qpos_adr+7] = target_quat_mujoco
        self._data.qvel[qvel_adr:qvel_adr+6] = 0.0
        
        # 重置body速度和加速度
        window_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "window_body")
        self._data.cvel[window_body_id][:3] = 0.0
        self._data.cvel[window_body_id][3:6] = 0.0
        self._data.cacc[window_body_id][:3] = 0.0
        self._data.cacc[window_body_id][3:6] = 0.0

        # 更新物理状态
        mujoco.mj_forward(self._model, self._data)
        
        # 碰撞检测：检查是否有穿透（借鉴父类的思路，但这里检查gripper和window的碰撞）
        ncon = self._data.ncon
        if ncon > 0:
            max_penetration = 0.0
            for i in range(ncon):
                contact = self._data.contact[i]
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                geom1_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                geom2_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
                
                # 检查是否涉及window_collision和gripper相关的几何体
                is_window = (geom1_name == "window_collision" or geom2_name == "window_collision")
                is_gripper = ("gripper" in str(geom1_name).lower() or "gripper" in str(geom2_name).lower() or
                             "suction" in str(geom1_name).lower() or "suction" in str(geom2_name).lower() or
                             "cup" in str(geom1_name).lower() or "cup" in str(geom2_name).lower())
                
                if is_window and is_gripper:
                    # 穿透深度（负值表示穿透）
                    dist = contact.dist
                    if dist < -0.005:  # 如果穿透超过5mm，认为有问题
                        max_penetration = min(max_penetration, dist)
            
            # 如果检测到严重穿透，回退到旧位置并增加偏移量
            if max_penetration < -0.005:
                # 增加偏移量重试
                target_pos = gripper_pos - (self._window_thickness + 0.04) * gripper_z_axis
                self._data.qpos[qpos_adr:qpos_adr+3] = target_pos
                self._data.qpos[qpos_adr+3:qpos_adr+7] = target_quat_mujoco
                self._data.qvel[qvel_adr:qvel_adr+6] = 0.0
                self._data.cvel[window_body_id][:3] = 0.0
                self._data.cvel[window_body_id][3:6] = 0.0
                self._data.cacc[window_body_id][:3] = 0.0
                self._data.cacc[window_body_id][3:6] = 0.0
                mujoco.mj_forward(self._model, self._data)
        
        # 最终更新物理状态
        mujoco.mj_forward(self._model, self._data)
    
    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重写 reset 方法，使用自定义采样区域"""
        # 调用父类 reset
        obs, info = super().reset(seed=seed, **kwargs)
        
        # 如果启用了随机位置，使用自定义采样区域重新采样
        if self._random_window_position:
            window_xy = self._np_random.uniform(*self._custom_sampling_bounds)
            self._data.jnt("window_joint").qpos[:3] = (*window_xy, self._window_z)
            
            # 随机化 rz（绕Z轴旋转），保持 rx=0, ry=0
            if self._random_rz_bounds is not None:
                # 从 rz 范围中随机采样
                random_rz = self._np_random.uniform(self._random_rz_bounds[0], self._random_rz_bounds[1])
                # 转换为四元数：rx=0, ry=0, rz=random_rz
                random_rot = R.from_euler('xyz', [0.0, 0.0, random_rz], degrees=False)
                random_quat = random_rot.as_quat()  # [x, y, z, w]
                random_quat_mujoco = np.array([random_quat[3], random_quat[0], random_quat[1], random_quat[2]])  # [w, x, y, z]
                self._data.jnt("window_joint").qpos[3:7] = random_quat_mujoco
            else:
                # Reset window orientation to flat (no rotation, lying on ground)
                self._data.jnt("window_joint").qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion
            
            # Reset window velocity
            self._data.jnt("window_joint").qvel[:] = 0.0
            mujoco.mj_forward(self._model, self._data)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """重写 step 方法，限制 rx 和 ry 旋转，并在吸附后保持玻璃与吸盘中心牢牢固定，不能发生任何位移"""
        # 在调用父类之前，限制 rx 和 ry 旋转
        action = action.copy()  # 复制 action 避免修改原始数组
        
        # 直接限制 rx_delta = 0，禁止绕X轴旋转
        # action[3] 是 rx_delta（绕X轴的旋转增量）
        action[3] = 0.0
        
        # 限制 ry 旋转范围
        if self._ry_rotation_bounds is not None:
            # 获取当前姿态
            mujoco.mj_forward(self._model, self._data)
            current_quat = self._data.mocap_quat[0].copy()  # [w, x, y, z]
            current_rot = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])  # [x, y, z, w]
            current_euler = current_rot.as_euler('xyz', degrees=False)
            current_ry = current_euler[1]
            
            # 计算应用 ry_delta 后的新 ry 角度
            ry_delta = action[4]  # action[4] 是 ry_delta（绕Y轴的旋转增量）
            delta_rot_ry = R.from_euler('xyz', [0, ry_delta, 0], degrees=False)
            new_rot_ry = delta_rot_ry * current_rot
            new_euler_ry = new_rot_ry.as_euler('xyz', degrees=False)
            new_ry = new_euler_ry[1]
            
            # 限制 ry 在指定范围内
            ry_min, ry_max = self._ry_rotation_bounds
            if new_ry < ry_min:
                # 计算需要限制的 ry_delta，使 new_ry = ry_min
                target_rot_ry = R.from_euler('xyz', [current_euler[0], ry_min, current_euler[2]], degrees=False)
                rot_delta_ry = target_rot_ry * current_rot.inv()
                delta_euler_ry = rot_delta_ry.as_euler('xyz', degrees=False)
                action[4] = delta_euler_ry[1]  # 限制后的 ry_delta
            elif new_ry > ry_max:
                # 计算需要限制的 ry_delta，使 new_ry = ry_max
                target_rot_ry = R.from_euler('xyz', [current_euler[0], ry_max, current_euler[2]], degrees=False)
                rot_delta_ry = target_rot_ry * current_rot.inv()
                delta_euler_ry = rot_delta_ry.as_euler('xyz', degrees=False)
                action[4] = delta_euler_ry[1]  # 限制后的 ry_delta
        
        # 调用父类 step（这会处理吸附逻辑）
        obs, rew, terminated, truncated, info = super().step(action)
        
        # 如果吸附了，强制同步玻璃位置到吸盘中心，确保牢牢固定，不能晃来晃去
        if self._window_snapped:
            # 获取当前吸盘位置
            if self._gripper_dock_site_id is not None:
                gripper_pos = self._data.site(self._gripper_dock_site_id).xpos.copy()
                gripper_mat = self._data.site(self._gripper_dock_site_id).xmat.reshape(3, 3).copy()
            else:
                try:
                    attachment_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                    gripper_pos = self._data.site(attachment_site_id).xpos.copy()
                    gripper_mat = self._data.site(attachment_site_id).xmat.reshape(3, 3).copy()
                except:
                    return obs, rew, terminated, truncated, info
            
            # 计算目标位置：玻璃表面对齐到gripper中心（与 _snap_window_to_gripper 保持一致）
            gripper_z_axis = gripper_mat[:, 2]
            target_pos = gripper_pos - (self._window_thickness + 0.03) * gripper_z_axis
            target_mat = gripper_mat
            
            # 转换为四元数
            target_quat = R.from_matrix(target_mat).as_quat()
            target_quat_mujoco = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
            
            # 设置玻璃位置和姿态，强制同步到吸盘中心
            window_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "window_joint")
            qpos_adr = self._model.jnt_qposadr[window_joint_id]
            qvel_adr = self._model.jnt_dofadr[window_joint_id]
            
            # 强制同步位置和姿态
            self._data.qpos[qpos_adr:qpos_adr+3] = target_pos
            self._data.qpos[qpos_adr+3:qpos_adr+7] = target_quat_mujoco
            # 强制速度为零，防止晃动
            self._data.qvel[qvel_adr:qvel_adr+6] = 0.0
            
            # 重置body速度和加速度，防止晃动
            window_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "window_body")
            self._data.cvel[window_body_id][:3] = 0.0
            self._data.cvel[window_body_id][3:6] = 0.0
            self._data.cacc[window_body_id][:3] = 0.0
            self._data.cacc[window_body_id][3:6] = 0.0
            
            # 清除约束力和应用力，防止物理引擎产生额外运动
            self._data.qfrc_constraint[qvel_adr:qvel_adr+6] = 0.0
            self._data.qfrc_applied[qvel_adr:qvel_adr+6] = 0.0
            
            # 更新物理状态
            mujoco.mj_forward(self._model, self._data)
        
        return obs, rew, terminated, truncated, info
