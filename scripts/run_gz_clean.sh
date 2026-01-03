#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD_FILE="world.sdf"
if [[ $# -gt 0 ]]; then
  WORLD_FILE="$1"
fi

# Clean environment to avoid snap core libc
exec env -i \
  HOME="$HOME" \
  DISPLAY="${DISPLAY:-}" \
  XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
  PATH="/opt/ros/jazzy/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  LD_LIBRARY_PATH="/opt/ros/jazzy/lib:/opt/ros/jazzy/opt/gz_sim_vendor/lib:/opt/ros/jazzy/opt/gz_gui_vendor/lib:/opt/ros/jazzy/opt/gz_rendering_vendor/lib:/opt/ros/jazzy/opt/gz_transport_vendor/lib:/opt/ros/jazzy/opt/gz_plugin_vendor/lib:/opt/ros/jazzy/opt/gz_common_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu" \
  ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  bash -lc "cd '$ROOT_DIR' && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch bots my_launch_file.py world:=$WORLD_FILE"
