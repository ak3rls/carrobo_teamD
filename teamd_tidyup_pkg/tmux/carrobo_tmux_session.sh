#!/usr/bin/env bash

# Start the four terminals used for the Car-Robo Isaac Sim workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ISAAC_DIR="${CARROBO_ISAAC_DIR:-/home/hma/carrobo-isaac}"
SESSION_NAME="${CARROBO_TMUX_SESSION:-carrobo}"
COMPE_SEED="${COMPE_SEED:-1}"
ATTACH_SESSION=true

usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  -n, --name NAME      tmux session name (default: carrobo)"
  echo "  -s, --seed 1..4      Isaac competition seed (default: 1)"
  echo "      --no-attach      Create the session without attaching"
  echo "  -h, --help           Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n | --name)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      SESSION_NAME="$2"
      shift 2
      ;;
    -s | --seed)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      COMPE_SEED="$2"
      shift 2
      ;;
    --no-attach)
      ATTACH_SESSION=false
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is not installed." >&2
  exit 1
}

[[ "$SESSION_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "Session name may contain only letters, numbers, '_' and '-'." >&2
  exit 2
}

[[ "$COMPE_SEED" =~ ^[1-4]$ ]] || {
  echo "Competition seed must be one of 1, 2, 3 or 4." >&2
  exit 2
}

[[ -f "$WORKSPACE_DIR/0_shell.sh" ]] || {
  echo "ROS workspace launcher not found: $WORKSPACE_DIR/0_shell.sh" >&2
  exit 1
}

[[ -f "$WORKSPACE_DIR/5e_isaac_mode.sh" ]] || {
  echo "Isaac mode setup not found: $WORKSPACE_DIR/5e_isaac_mode.sh" >&2
  exit 1
}

[[ -f "$ISAAC_DIR/Makefile" ]] || {
  echo "Car-Robo Isaac directory not found: $ISAAC_DIR" >&2
  exit 1
}

attach_or_switch() {
  if [[ "$ATTACH_SESSION" != true ]]; then
    return
  fi

  if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$SESSION_NAME"
  else
    tmux attach-session -t "$SESSION_NAME"
  fi
}

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists."
  attach_or_switch
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -n main -c "$ISAAC_DIR"

isaac_pane="$(
  tmux display-message -p -t "$SESSION_NAME:main.0" '#{pane_id}'
)"
bringup_pane="$(
  tmux split-window -h -P -F '#{pane_id}' -t "$isaac_pane" \
    -c "$WORKSPACE_DIR"
)"
navigation_pane="$(
  tmux split-window -v -P -F '#{pane_id}' -t "$isaac_pane" \
    -c "$WORKSPACE_DIR"
)"
tidyup_pane="$(
  tmux split-window -v -P -F '#{pane_id}' -t "$bringup_pane" \
    -c "$WORKSPACE_DIR"
)"

tmux select-layout -t "$SESSION_NAME:main" tiled >/dev/null
tmux select-pane -t "$isaac_pane" -T 'Isaac Sim'
tmux select-pane -t "$bringup_pane" -T 'HSR bringup'
tmux select-pane -t "$navigation_pane" -T 'Navigation'
tmux select-pane -t "$tidyup_pane" -T 'Tidyup (ready)'

queue_ros_environment() {
  local pane_id="$1"

  tmux send-keys -t "$pane_id" 'bash ./0_shell.sh' C-m
  tmux send-keys -t "$pane_id" '. /entrypoint.sh' C-m
  tmux send-keys -t "$pane_id" '. 5e_isaac_mode.sh' C-m
  tmux send-keys -t "$pane_id" 'is' C-m
}

tmux send-keys -t "$isaac_pane" \
  "make up localhost compe_seed=$COMPE_SEED" C-m

queue_ros_environment "$bringup_pane"
tmux send-keys -t "$bringup_pane" \
  'ros2 launch hma_hsr_utils2 bringup.launch.py simulator:=true simulator_mode:=isaac' \
  C-m

queue_ros_environment "$navigation_pane"
tmux send-keys -t "$navigation_pane" \
  'ros2 launch carrobo_slam navigation.launch.py map_name:=carrobo'

queue_ros_environment "$tidyup_pane"
# Prepare the command but leave it unexecuted for the operator.
tmux send-keys -t "$tidyup_pane" \
  'ros2 launch teamd_tidyup_pkg tidyup.launch.py'

tmux select-pane -t "$tidyup_pane"

echo "Created tmux session '$SESSION_NAME' (competition seed $COMPE_SEED)."
echo "The tidyup command is prepared but has not been executed."
attach_or_switch
