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

import json
from pathlib import Path
import numpy as np


def load_controller_config(controller_name: str, config_path: str | None = None) -> dict:
    """
    Load controller configuration from a JSON file.

    Args:
        controller_name: Name of the controller to load.
        config_path: Path to the config file. If None, uses the package's default config.

    Returns:
        Dictionary containing the selected controller's configuration.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "controller_config.json"

    with open(config_path) as f:
        config = json.load(f)

    controller_config = config[controller_name] if controller_name in config else config["default"]

    if controller_name not in config:
        print(f"Controller {controller_name} not found in config. Using default configuration.")

    return controller_config


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def reset(self):
        """Reset the controller."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "no-op"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "success": False,
            "failure": False,
            "intervention": False,
            "rerecord": False,
        }
        self.listener = None

    def start(self):
        """Start the keyboard listener."""
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = True
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = True
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = "success"
                elif key == keyboard.Key.esc:
                    self.key_states["failure"] = True
                    self.episode_end_status = "failure"
                elif key == keyboard.Key.space:
                    self.key_states["intervention"] = not self.key_states["intervention"]
                elif key == keyboard.Key.r:
                    self.key_states["rerecord"] = True
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = False
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Right Ctrl and Left Ctrl: Open and close gripper")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  Space: Start/Stop Intervention")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z

    def should_save(self):
        """Return True if Enter was pressed (save episode)."""
        return self.key_states["success"] or self.key_states["failure"]

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.key_states["intervention"]

    def reset(self):
        """Reset the controller."""
        for key in self.key_states:
            self.key_states[key] = False


class GamepadController(InputController):
    """Generate motion deltas from gamepad input."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, deadzone=0.1, config_path=None):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False
        self.config_path = config_path
        self.controller_config = None

    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        joystick_name = self.joystick.get_name()
        print(f"Initialized gamepad: {joystick_name}")

        # Load controller configuration based on joystick name
        self.controller_config = load_controller_config(joystick_name, self.config_path)

        # Get button mappings from config
        buttons = self.controller_config.get("buttons", {})

        print("Gamepad controls:")
        print(f"  {buttons.get('rb', 'RB')} button: Intervention")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print(f"  {buttons.get('lt', 'LT')} button: Close gripper")
        print(f"  {buttons.get('rt', 'RT')} button: Open gripper")
        print(f"  {buttons.get('b', 'B')}/Circle button: Exit")
        print(f"  {buttons.get('y', 'Y')}/Triangle button: End episode with SUCCESS")
        print(f"  {buttons.get('a', 'A')}/Cross button: End episode with FAILURE")
        print(f"  {buttons.get('x', 'X')}/Square button: Rerecord episode")

    def stop(self):
        """Clean up pygame resources."""
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        import pygame

        # Get button mappings from config
        buttons = self.controller_config.get("buttons", {})
        y_button = buttons.get("y", 3)  # Default to 3 if not found
        a_button = buttons.get("a", 0)  # Default to 0 if not found (Logitech F310)
        x_button = buttons.get("x", 2)  # Default to 2 if not found (Logitech F310)
        lt_button = buttons.get("lt", 6)  # Default to 6 if not found
        rt_button = buttons.get("rt", 7)  # Default to 7 if not found
        rb_button = buttons.get("rb", 5)  # Default to 5 if not found

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == y_button:
                    self.episode_end_status = "success"
                elif event.button == a_button:
                    self.episode_end_status = "failure"
                elif event.button == x_button:
                    self.episode_end_status = "rerecord_episode"
                elif event.button == lt_button:
                    self.close_gripper_command = True
                elif event.button == rt_button:
                    self.open_gripper_command = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [x_button, a_button, y_button]:
                    self.episode_end_status = None
                elif event.button == lt_button:
                    self.close_gripper_command = False
                elif event.button == rt_button:
                    self.open_gripper_command = False

            # Check for RB button for intervention flag
            if self.joystick.get_button(rb_button):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        import pygame

        try:
            # Get axis mappings from config
            axes = self.controller_config.get("axes", {})
            axis_inversion = self.controller_config.get("axis_inversion", {})

            # Get axis indices from config (with defaults if not found)
            left_x_axis = axes.get("left_x", 0)
            left_y_axis = axes.get("left_y", 1)
            right_y_axis = axes.get("right_y", 3)

            # Get axis inversion settings (with defaults if not found)
            invert_left_x = axis_inversion.get("left_x", False)
            invert_left_y = axis_inversion.get("left_y", True)
            invert_right_y = axis_inversion.get("right_y", True)

            # Read joystick axes
            x_input = self.joystick.get_axis(left_x_axis)  # Left/Right
            y_input = self.joystick.get_axis(left_y_axis)  # Up/Down
            z_input = self.joystick.get_axis(right_y_axis)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Apply inversion if configured
            if invert_left_x:
                x_input = -x_input
            if invert_left_y:
                y_input = -y_input
            if invert_right_y:
                z_input = -z_input

            # Calculate deltas
            delta_x = y_input * self.y_step_size  # Forward/backward
            delta_y = x_input * self.x_step_size  # Left/right
            delta_z = z_input * self.z_step_size  # Up/down

            return delta_x, delta_y, delta_z

        except pygame.error:
            print("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Button states
        self.buttons = {}
        self.quit_requested = False
        self.save_requested = False

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        import hid

        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5"]):
                return device

        print("No gamepad found, check the connection and the product string in HID to add your gamepad")
        return None

    def start(self):
        """Connect to the gamepad using HIDAPI."""
        import hid

        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            print(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            print(f"Connected to {manufacturer} {product}")

            print("Gamepad controls (HID mode):")
            print("  Left analog stick: Move in X-Y plane")
            print("  Right analog stick: Move in Z axis (vertical)")
            print("  Button 1/B/Circle: Exit")
            print("  Button 2/A/Cross: End episode with SUCCESS")
            print("  Button 3/X/Square: End episode with FAILURE")

        except OSError as e:
            print(f"Error opening gamepad: {e}")
            print("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        Read and process the latest gamepad data.
        Due to an issue with the HIDAPI, we need to read the read the device several times in order to get a stable reading
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            # Read data from the gamepad
            data = self.device.read(64)
            # Interpret gamepad data - this will vary by controller model
            # These offsets are for the Logitech RumblePad 2
            if data and len(data) >= 8:
                # Normalize joystick values from 0-255 to -1.0-1.0
                self.left_x = (data[1] - 128) / 128.0
                self.left_y = (data[2] - 128) / 128.0
                self.right_x = (data[3] - 128) / 128.0
                self.right_y = (data[4] - 128) / 128.0

                # Apply deadzone
                self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                # Parse button states (byte 5 in the Logitech RumblePad 2)
                buttons = data[5]

                # Check if RB is pressed then the intervention flag should be set
                self.intervention_flag = data[6] in [2, 6, 10, 14]

                # Check if RT is pressed
                self.open_gripper_command = data[6] in [8, 10, 12]

                # Check if LT is pressed
                self.close_gripper_command = data[6] in [4, 6, 12]

                # Check if Y/Triangle button (bit 7) is pressed for saving
                # Check if X/Square button (bit 5) is pressed for failure
                # Check if A/Cross button (bit 4) is pressed for rerecording
                if buttons & 1 << 7:
                    self.episode_end_status = "success"
                elif buttons & 1 << 5:
                    self.episode_end_status = "failure"
                elif buttons & 1 << 4:
                    self.episode_end_status = "rerecord_episode"
                else:
                    self.episode_end_status = None

        except OSError as e:
            print(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_y * self.x_step_size  # Forward/backward
        delta_y = -self.left_x * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z

    def should_quit(self):
        """Return True if quit button was pressed."""
        return self.quit_requested

    def should_save(self):
        """Return True if save button was pressed."""
        return self.save_requested


class MetaQuestController(InputController):
    """
    Adapter that wraps MetaQuestTeleop to work with InputController interface.
    
    Uses the existing MetaQuestTeleop from lerobot.teleoperators.meta_quest for:
    - 6-DoF control (position + rotation)
    - Gripper control via trigger
    - Episode control via keyboard
    
    Controls:
    - Hand movement: X, Y, Z position
    - Hand rotation: Roll, Pitch, Yaw
    - RTr (trigger) 按住: Close gripper
    - RTr (trigger) 松开: Open gripper  
    - A button 按住: Intervention mode (干预模式)
    - Enter: Success
    - Backspace: Failure
    - R: Rerecord
    - ESC: Quit
    """

    def __init__(
        self, 
        x_step_size=0.01, 
        y_step_size=0.01, 
        z_step_size=0.01,
        translation_scale=1.0,   # Increased from 0.2 for larger movements
        rotation_scale=1.0,      # Increased from 0.2 for larger rotations
        deadzone=0.001,
        right_controller=True,
    ):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.deadzone = deadzone
        self.right_controller = right_controller
        
        self._teleop = None  # MetaQuestTeleop instance
        self._robot_init_pose = None
        self._is_initialized = False
        
        # 6-DoF deltas (updated from MetaQuestTeleop.get_action())
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_z = 0.0
        self.delta_roll = 0.0
        self.delta_pitch = 0.0
        self.delta_yaw = 0.0
        
        # Intervention state
        self._intervention_active = False
        
        # Episode control states
        self.key_states = {
            "success": False,
            "failure": False,
            "quit": False,
            "rerecord": False,
        }

    def start(self):
        """Start the Meta Quest controller using MetaQuestTeleop."""
        try:
            from lerobot.teleoperators.meta_quest.teleop_meta_quest import MetaQuestTeleop
            from lerobot.teleoperators.meta_quest.configuration_meta_quest import MetaQuestTeleopConfig
            
            # Create config with our settings
            config = MetaQuestTeleopConfig(
                use_gripper=True,
                right_controller=self.right_controller,
                translation_scale=self.translation_scale,
                rotation_scale=self.rotation_scale,
                deadzone=self.deadzone,
            )
            
            # Create and connect MetaQuestTeleop
            self._teleop = MetaQuestTeleop(config)
            self._teleop.connect()
            
            print("")
            print("=" * 70)
            print("       Meta Quest + HILSerl Teleop Controller Initialized")
            print("=" * 70)
            print("")
            print("【Meta Quest 6-DoF 控制】")
            print("  ✋ 手柄移动 → 控制机械臂位置 (X, Y, Z)")
            print("  🔄 手柄旋转 → 控制机械臂姿态 (Roll, Pitch, Yaw)")
            print("  🎮 RTr (扳机) 按住 → 关闭夹爪 (抓取)")
            print("  🎮 RTr (扳机) 松开 → 打开夹爪 (释放)")
            print("")
            print("【HILSerl 键盘功能键】")
            print("  ⏎  Enter     → 标记 Episode 成功 (SUCCESS) 并结束")
            print("  ⌫  Backspace → 标记 Episode 失败 (FAILURE) 并结束")
            print("  🔄 R         → 丢弃当前录制，重新录制 (RERECORD)")
            print("  🔤 q         → 退出程序 (QUIT)")
            print("")
            print("【缩放参数】")
            print(f"  Position scale: {self.translation_scale} | Rotation scale: {self.rotation_scale}")
            print("=" * 70)
            print("")
            
        except ImportError as e:
            print(f"Failed to import MetaQuestTeleop: {e}")
            print("Make sure lerobot.teleoperators.meta_quest is installed.")
            self._teleop = None
        except Exception as e:
            print(f"Failed to initialize MetaQuestTeleop: {e}")
            import traceback
            traceback.print_exc()
            self._teleop = None

    def set_robot_init_pose(self, robot_init_pose):
        """Set the robot initial pose for delta calculation.
        
        This should be called after environment reset with the robot's EE pose.
        """
        self._robot_init_pose = robot_init_pose.copy() if robot_init_pose is not None else np.eye(4)
        
        if self._teleop is not None:
            self._teleop.set_robot_init_pose(self._robot_init_pose)
            if not self._is_initialized:
                print(f"[MetaQuest] Robot init pose set. Ready for control.")
            self._is_initialized = True

    def stop(self):
        """Stop the Meta Quest controller."""
        if self._teleop is not None:
            try:
                self._teleop.disconnect()
            except Exception:
                pass
            self._teleop = None

    def update(self):
        """Update controller state from MetaQuestTeleop."""
        if self._teleop is None:
            return
        
        # Auto-initialize with default pose if not set
        if not self._is_initialized:
            default_pose = np.eye(4)
            default_pose[:3, 3] = [0, 0, 0]  # Origin
            self.set_robot_init_pose(default_pose)
        
        try:
            # Get action from MetaQuestTeleop
            action = self._teleop.get_action()
            
            # Extract deltas
            self.delta_x = action.get("delta_x", 0.0)
            self.delta_y = action.get("delta_y", 0.0)
            self.delta_z = action.get("delta_z", 0.0)
            self.delta_roll = action.get("delta_roll", 0.0)
            self.delta_pitch = action.get("delta_pitch", 0.0)
            self.delta_yaw = action.get("delta_yaw", 0.0)
            
            # Get gripper state (0=close, 1=stay, 2=open)
            gripper_action = action.get("gripper", 1)
            self.close_gripper_command = (gripper_action == 0)
            self.open_gripper_command = (gripper_action == 2)
            
            # Get teleop events for intervention and episode control
            events = self._teleop.get_teleop_events()
            self._intervention_active = events.get("is_intervention", False)
            
            # Handle episode end events from keyboard
            if events.get("success", False):
                self.key_states["success"] = True
                self.episode_end_status = "success"
            if events.get("terminate_episode", False) and not events.get("success", False):
                self.key_states["failure"] = True
                self.episode_end_status = "failure"
            if events.get("rerecord_episode", False):
                self.key_states["rerecord"] = True
                self.episode_end_status = "rerecord_episode"
            
            # Debug print removed - now using ACTION print in get_gamepad_action()
            
        except Exception as e:
            print(f"\nError reading MetaQuestTeleop state: {e}")

    def get_deltas(self):
        """Get the current 3-DoF position deltas."""
        return self.delta_x, self.delta_y, self.delta_z

    def get_6dof_deltas(self):
        """Get all 6-DoF deltas (position + rotation)."""
        return {
            "delta_x": self.delta_x,
            "delta_y": self.delta_y,
            "delta_z": self.delta_z,
            "delta_roll": self.delta_roll,
            "delta_pitch": self.delta_pitch,
            "delta_yaw": self.delta_yaw,
        }

    def should_intervene(self):
        """Return True when intervention is active (always True for Meta Quest)."""
        # For Meta Quest, we consider any valid controller connection as intervention
        # Unlike gamepad where you explicitly press a button
        return self._teleop is not None and self._is_initialized

    def reset(self):
        """Reset the controller state."""
        self._robot_init_pose = None
        self._is_initialized = False
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_z = 0.0
        self.delta_roll = 0.0
        self.delta_pitch = 0.0
        self.delta_yaw = 0.0
        
        for key in self.key_states:
            self.key_states[key] = False
        
        self.episode_end_status = None
