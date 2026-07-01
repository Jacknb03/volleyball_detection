#!/bin/bash
# 从 GitHub Release 下载 ONNX（缺文件时 deploy / start 可自动调用）
#
# 用法:
#   bash scripts/download_models.sh              # 按 YOLO_INPUT_SIZE 下缺的那个
#   bash scripts/download_models.sh --all        # best_416 + best.onnx
#   bash scripts/download_models.sh --416
#   bash scripts/download_models.sh --640
#
set -eo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$WS/src/station_detector_cpp/model"
mkdir -p "$MODEL_DIR"

MODELS_CONF="${MODELS_CONF:-$WS/config/models.conf}"
PIPELINE_CONF="${PIPELINE_CONF:-$WS/config/pipeline.conf}"
if [[ -f "$MODELS_CONF" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$MODELS_CONF"
  set -u 2>/dev/null || true
fi
if [[ -f "$PIPELINE_CONF" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$PIPELINE_CONF"
  set -u 2>/dev/null || true
fi

MODEL_REPO="${MODEL_REPO:-Jacknb03/volleyball_detection}"
MODEL_RELEASE_TAG="${MODEL_RELEASE_TAG:-v0.1-models}"
MODEL_416_FILE="${MODEL_416_FILE:-best_416.onnx}"
MODEL_640_FILE="${MODEL_640_FILE:-best.onnx}"

download_one() {
  local filename="$1"
  local dest="$MODEL_DIR/$filename"
  if [[ -f "$dest" ]]; then
    echo "已有: $dest ($(du -h "$dest" | cut -f1))"
    return 0
  fi
  local url="https://github.com/${MODEL_REPO}/releases/download/${MODEL_RELEASE_TAG}/${filename}"
  echo "下载: $url"
  if command -v curl >/dev/null 2>&1; then
    if curl -fsSL --connect-timeout 15 --retry 2 -o "$dest" "$url"; then
      echo "OK: $dest ($(du -h "$dest" | cut -f1))"
      return 0
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget -q -O "$dest" "$url"; then
      echo "OK: $dest"
      return 0
    fi
  else
    echo "ERROR: 需要 curl 或 wget"
    return 1
  fi
  rm -f "$dest"
  echo "ERROR: 下载失败。请确认 Release 已发布: ${MODEL_RELEASE_TAG} / ${filename}"
  echo "  或: bash scripts/sync_to_ipc.sh USER@IPC --deploy"
  return 1
}

DO_416=false
DO_640=false
for arg in "$@"; do
  case "$arg" in
    --all) DO_416=true; DO_640=true ;;
    --416) DO_416=true ;;
    --640) DO_640=true ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
  esac
done

if ! $DO_416 && ! $DO_640; then
  if [[ "${YOLO_INPUT_SIZE:-416}" == "416" ]]; then
    DO_416=true
  else
    DO_640=true
  fi
fi

ok=true
$DO_416 && download_one "$MODEL_416_FILE" || ok=false
$DO_640 && download_one "$MODEL_640_FILE" || ok=false
$ok
