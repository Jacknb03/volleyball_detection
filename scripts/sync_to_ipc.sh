#!/bin/bash
# 从开发机同步代码 + ONNX 模型到工控机，并在远端执行 deploy_ipc.sh
#
# 用法（在开发机 volleyball_detection 根目录）:
#   bash scripts/sync_to_ipc.sh USER@172.19.33.210
#   bash scripts/sync_to_ipc.sh USER@172.19.33.210 --deploy        # 同步后远端自动 deploy
#   bash scripts/sync_to_ipc.sh USER@172.19.33.210 --deploy --pull # 远端 git pull 后 deploy（远端也是 git 仓时）
#
# 环境变量:
#   IPC_DEST   远端目录，默认 ~/volleyball_detection
#   RSYNC_EXCLUDE  额外 rsync --exclude
#
set -eo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,12p' "$0"
  exit 1
fi

IPC_HOST="$1"
shift

DO_DEPLOY=false
DO_PULL=false
for arg in "$@"; do
  case "$arg" in
    --deploy) DO_DEPLOY=true ;;
    --pull) DO_PULL=true ;;
    *)
      echo "未知参数: $arg"
      exit 1
      ;;
  esac
done

WS="$(cd "$(dirname "$0")/.." && pwd)"
IPC_DEST="${IPC_DEST:-~/volleyball_detection}"

# 展开远端 ~
REMOTE_PATH="${IPC_DEST/#\~/$HOME}"
# rsync 远端路径用 SSH 语法
RSYNC_DEST="${IPC_HOST}:${IPC_DEST}/"

log() { echo ""; echo "=== $* ==="; }

log "同步 $WS → $RSYNC_DEST"

rsync -avz --delete \
  --exclude '.git/' \
  --exclude 'build/' \
  --exclude 'install/' \
  --exclude 'log/' \
  --exclude '__pycache__/' \
  --exclude '.cursor/' \
  --exclude '.vscode/' \
  --exclude 'universal_controllers_v2-main/' \
  --exclude 'debug_images/' \
  "${RSYNC_EXCLUDE:-}" \
  "$WS/" "$RSYNC_DEST"

log "同步 ONNX 模型"
for f in best_416.onnx best.onnx; do
  src="$WS/src/station_detector_cpp/model/$f"
  if [[ -f "$src" ]]; then
    rsync -avz "$src" "${IPC_HOST}:${IPC_DEST}/src/station_detector_cpp/model/"
  else
    echo "SKIP (本地无): $f"
  fi
done

if $DO_DEPLOY; then
  log "远端 deploy_ipc.sh"
  DEPLOY_CMD="cd ${IPC_DEST} && bash scripts/deploy_ipc.sh"
  if $DO_PULL; then
    DEPLOY_CMD="cd ${IPC_DEST} && bash scripts/deploy_ipc.sh --pull"
  fi
  ssh -t "$IPC_HOST" "$DEPLOY_CMD"
else
  echo ""
  echo "同步完成。远端手动执行:"
  echo "  ssh $IPC_HOST"
  echo "  cd ${IPC_DEST} && bash scripts/deploy_ipc.sh"
  echo ""
  echo "或本机一条:"
  echo "  bash scripts/sync_to_ipc.sh $IPC_HOST --deploy"
fi
