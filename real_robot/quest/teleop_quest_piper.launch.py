"""Meta Quest teleop of one Piper through AgileX's QuestArmTeleop stack.

Upstream's teleop_single_piper.launch.py with our rig's settings:
  - can_port: our persistent names (can_piper1 = close arm, can_piper2 = far);
    upstream renames the adapters to can0 / can_left / can_right.
  - fast_mode false: IK targets go through move_j (position mode, firmware
    speed cap). Upstream uses move_js (MIT mode, no smoothing), which
    pyAgxArm itself rates "EXTREMELY HIGH" risk.
  - speed_percent: firmware speed cap (our configs use 30).
  - IK weights: see the overrides below (gripper joint locked, w_reg 0,
    w_pos 20, w_smooth 0.3).
  - ik_guard.py sits between the IK and the driver: per-step rate limit from
    configs/piper_<arm>.json, stop on IK jumps or large lag (upstream sends
    the IK solution straight to the arm).
  - check:=true starts only the arm driver with auto_enable false: it reads
    joints and TCP pose and ignores commands, so nothing moves.

Press and release A (right controller) to engage: the arm then follows the
controller's motion; B stops, and the arm holds. The driver does not disable the arm on exit: it keeps holding the
last target with motors on, so power down with `make park`.

  ros2 launch real_robot/quest/teleop_quest_piper.launch.py arm:=1 speed_percent:=60
See docs/robot.md ("Quest teleop") and `make quest-teleop`.
"""
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CAN_PORTS = {"1": "can_piper1", "2": "can_piper2"}
TCP_OFFSET = "[0.0, 0.0, 0.13, 0.0, 0.0, 0.0]"  # gripper fingertips, as upstream


def _nodes(context):
    arm = LaunchConfiguration("arm").perform(context)
    if arm not in CAN_PORTS:
        raise RuntimeError(f"arm must be 1 (close) or 2 (far), got {arm!r}")
    check = LaunchConfiguration("check").perform(context) == "true"
    rviz = LaunchConfiguration("rviz").perform(context) == "true"
    reader_dir = get_package_share_directory("oculus_reader")
    ctrl_dir = get_package_share_directory("agx_arm_ctrl")

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            ctrl_dir, "launch",
            "start_single_agx_arm_rviz.launch.py" if rviz else "start_single_agx_arm.launch.py")),
        launch_arguments={
            "can_port": CAN_PORTS[arm],
            "arm_type": "piper",
            "auto_enable": "false" if check else "true",
            "fast_mode": "false",
            "speed_percent": "0" if check else LaunchConfiguration("speed_percent").perform(context),
            "effector_type": "agx_gripper",
            "tcp_offset": TCP_OFFSET,
            "control": "false",
        }.items(),
    )
    if check:
        return [driver]

    # Upstream's IK node with latest-pose-only input and no feedback resync
    # while engaged (ik_node.py explains the 340 ms it cost).
    ik = Node(
        executable=sys.executable, arguments=[os.path.join(HERE, "ik_node.py")],
        name="arm_ik_pose_node", output="screen",
        parameters=[os.path.join(reader_dir, "config", "arm_ik_pose_node.piper.yaml"), {
            # The current URDF adds a prismatic "gripper" joint the upstream
            # config doesn't lock (the IK would solve for 7 joints).
            "locked_joints": ["gripper_base_joint", "gripper", "gripper_joint1", "gripper_joint2"],
            # Upstream regularizes toward q = 0; our bases rest at -93 deg, so it
            # dragged the base ~12 deg off during a 3 cm move (offline test).
            "w_reg": 0.0,
            # Upstream's piper yaml has w_pos 2 = w_ori 2: 1 cm of position error
            # then costs ~100x less than 6 deg of orientation, and offline a 20 cm
            # sideways sweep missed the hand by 53 mm median, 172 max. 20 / 0.3
            # (with quest_delta's heading following the base): 4.6 mm, 9 max,
            # joint steps <= 9 deg (2026-10-15).
            "w_pos": 20.0,
            "w_smooth": 0.3,
            "pin_joint_status_topic": "/ik/joint_states",
        }],
    )
    guard = Node(
        executable=sys.executable, arguments=[os.path.join(HERE, "ik_guard.py")],
        name="ik_guard", output="screen",
        parameters=[{"config": os.path.join(REPO, "real_robot", "configs", f"piper_{arm}.json"),
                     "smooth_hz": float(LaunchConfiguration("smooth_hz").perform(context)),
                     "floor_z": float(LaunchConfiguration("floor_mm").perform(context)) / 1000.0}],
    )
    # Controller frame -> arm end-effector frame; tune on the Quest side.
    pub_pose = Node(
        package="oculus_reader", executable="pub_pose.py", name="pub_pose_node", output="screen",
        arguments=["--ros-args", "-p",
                   f"ros_to_arm_rpy:={LaunchConfiguration('ros_to_arm_rpy').perform(context)}"],
    )
    # Buttons/trigger -> IK target and gripper; replaces upstream's
    # pub_delta_pose (A re-anchors at the arm's pose and clears guard pauses).
    pub_delta = Node(
        executable=sys.executable, arguments=[os.path.join(HERE, "quest_delta.py")],
        name="quest_delta", output="screen",
        parameters=[{"mode": LaunchConfiguration("mode").perform(context),
                     "forward_yaw_deg": float(LaunchConfiguration("forward_yaw_deg").perform(context)),
                     "scale": float(LaunchConfiguration("scale").perform(context))}],
    )
    nodes = [driver, ik, guard, pub_pose, pub_delta]
    trace = LaunchConfiguration("trace").perform(context)
    if trace:  # passive latency recorder (subscribes only); see trace_report.py
        nodes.append(Node(executable=sys.executable, arguments=[os.path.join(HERE, "trace.py")],
                          name="quest_trace", output="screen",
                          parameters=[{"out": os.path.abspath(trace)}]))
    if rviz:
        nodes.append(Node(
            package="rviz2", executable="rviz2", name="rviz2", output="screen",
            arguments=["-d", os.path.join(reader_dir, "config", "oculus_reader.rviz")]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("arm", default_value="1", description="1 = close arm, 2 = far arm"),
        DeclareLaunchArgument("speed_percent", default_value="60",
                              description="firmware speed cap, percent"),
        DeclareLaunchArgument("smooth_hz", default_value="3.0",
                              description="ik_guard filter cutoff, Hz (lower: smoother, more lag)"),
        DeclareLaunchArgument("ros_to_arm_rpy", default_value="[0.0, 1.5708, 0.0]"),
        DeclareLaunchArgument("rviz", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("mode", default_value="room", choices=["room", "controller"],
                              description="room: 1:1 in the room (calibrate forward with the right "
                                          "grip); controller: upstream's controller-frame motion"),
        DeclareLaunchArgument("forward_yaw_deg", default_value="-93.3",
                              description="room mode: base-frame direction of your 'forward'"),
        DeclareLaunchArgument("floor_mm", default_value="0",
                              description="ik_guard: lowest the gripper/wrist may go, mm above z = 0 "
                                          "of the arm's base frame"),
        DeclareLaunchArgument("scale", default_value="1.0",
                              description="room mode: gripper motion per hand motion"),
        DeclareLaunchArgument("trace", default_value="",
                              description="record latency data to this .npz (trace_report.py)"),
        DeclareLaunchArgument("check", default_value="false", choices=["true", "false"],
                              description="driver only, arm not enabled: read-only"),
        OpaqueFunction(function=_nodes),
    ])
