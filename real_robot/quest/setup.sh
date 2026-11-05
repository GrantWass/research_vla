#!/usr/bin/env bash
# Robot-side install for Meta Quest teleop (AgileX QuestArmTeleop), no sudo:
#   ~/miniforge3            conda; env "vt" = ROS 2 Humble (RoboStack) + pinocchio + casadi
#   ~/QuestArmTeleop        colcon workspace: oculus_reader + agx_arm_ros + agx_arm_urdf
#   ~/src_ext/pyAgxArm      AgileX SDK used by the agx_arm_ros driver (pip-installed into vt)
#   ~/opt/platform-tools    Google's adb build (upstream uses apt's android-tools-adb)
# Upstream's README uses apt ROS + conda python 3.10 + pinocchio 3.2; this was
# tested with the versions pinned below (2026-10-08). Safe to rerun.
# Run: make quest-setup. See docs/robot.md ("Quest teleop").
set -euo pipefail

QAT_COMMIT=145b803        # agilexrobotics/QuestArmTeleop (ros2)
AGX_ROS_COMMIT=e4ccec1    # agilexrobotics/agx_arm_ros (ros2)
URDF_COMMIT=453a4a1       # agilexrobotics/agx_arm_urdf (flattened)
PYAGXARM_COMMIT=841a625   # agilexrobotics/pyAgxArm (master)

clone() {  # url dir branch commit
    if [ ! -d "$2/.git" ]; then git clone -b "$3" "$1" "$2"; fi
    git -C "$2" fetch -q origin "$3"
    git -C "$2" checkout -q "$4"
}

if [ ! -x ~/miniforge3/bin/conda ]; then
    mkdir -p ~/Downloads/miniforge_dl
    curl -fsSL -o ~/Downloads/miniforge_dl/Miniforge3-Linux-x86_64.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash ~/Downloads/miniforge_dl/Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
fi
source ~/miniforge3/etc/profile.d/conda.sh
if ! conda env list | grep -q '^vt '; then
    mamba create -y -n vt -c robostack-staging -c conda-forge python=3.14 \
        ros-humble-ros-base ros-humble-rviz2 ros-humble-rviz-default-plugins \
        ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
        ros-humble-joint-state-publisher-gui ros-humble-xacro ros-humble-tf2-ros \
        ros-humble-launch-ros ros-humble-ament-cmake ros-humble-ament-cmake-python \
        ros-humble-rosidl-default-generators ros-humble-std-srvs ros-humble-topic-tools \
        colcon-common-extensions compilers cmake pkg-config make ninja pip \
        pinocchio=4.1.0 casadi=3.7.2 scipy numpy pyyaml python-can
fi
conda activate vt

clone https://github.com/agilexrobotics/QuestArmTeleop.git ~/QuestArmTeleop ros2 $QAT_COMMIT
clone https://github.com/agilexrobotics/agx_arm_ros.git ~/QuestArmTeleop/src/agx_arm_ros ros2 $AGX_ROS_COMMIT
clone https://github.com/agilexrobotics/agx_arm_urdf.git \
    ~/QuestArmTeleop/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf flattened $URDF_COMMIT
clone https://github.com/agilexrobotics/pyAgxArm.git ~/src_ext/pyAgxArm master $PYAGXARM_COMMIT

python -m pip install -q ~/src_ext/pyAgxArm pure-python-adb meshcat
(cd ~/QuestArmTeleop && colcon build)

if [ ! -x ~/opt/platform-tools/adb ]; then
    mkdir -p ~/opt/adb_dl
    curl -fsSL -o ~/opt/adb_dl/platform-tools.zip \
        https://dl.google.com/android/repository/platform-tools-latest-linux.zip
    python -c "import zipfile; zipfile.ZipFile('$HOME/opt/adb_dl/platform-tools.zip').extractall('$HOME/opt')"
    chmod +x ~/opt/platform-tools/adb
fi
echo "Quest teleop installed. Read-only arm check: make quest-check ARM=1"
