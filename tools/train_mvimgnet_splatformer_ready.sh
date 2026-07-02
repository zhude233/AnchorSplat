#!/usr/bin/env bash

set -u

ROOT="data/mvimgnet_splatformer_ready"
GPU="0"
SCENE_LIMIT=""
SCENE_LIST=""
COLMAP_SCRIPT="${COLMAP_SCRIPT:-tools/colmap.py}"
COLMAP_PYTHON="${COLMAP_PYTHON:-python}"
COLMAP_ENV_BIN="${COLMAP_ENV_BIN:-}"
REWRITE_BINS_SCRIPT="${REWRITE_BINS_SCRIPT:-$(dirname "${BASH_SOURCE[0]}")/rewrite_colmap_bins_legacy.py}"
COLMAP_ARGS_TEMPLATE=""
GSPLAT_TRAINER="${GSPLAT_TRAINER:-python third_party/gsplat/examples/simple_trainer.py default}"
GSPLAT_ARGS_TEMPLATE='--data_dir "{scene256}" --result_dir "{scene256}/gaussian_splatting"'
LOG_FILE=""
DRY_RUN=0
SKIP_REWRITE_BINS=0

usage() {
  cat <<'EOF'
Usage: bash tools/train_mvimgnet_splatformer_ready.sh [options]

Options:
  --target-root PATH        Converted MVImgNet root.
  --root PATH               Alias for --target-root.
  --gpu ID                  CUDA_VISIBLE_DEVICES value. Default: 0.
  --scene-limit N           Maximum number of scenes to process.
  --scene-list PATH         Text file with scene names or paths, one per line.
  --colmap-script PATH      colmap.py path. Default: tools/colmap.py.
  --colmap-python PATH      Python executable for COLMAP preprocessing. Default: python.
  --colmap-env-bin PATH     Optional bin directory for the COLMAP env.
  --rewrite-bins-script PATH
                            Script that rewrites COLMAP text models to legacy .bin files.
  --colmap-args TEMPLATE    Args for colmap step. Tokens: {scene}, {scene256}, {root}.
                            Default: uses render_set/colmap.py required arguments.
  --gsplat-trainer CMD      gsplat trainer command.
                            Default: python third_party/gsplat/examples/simple_trainer.py default.
  --gsplat-args TEMPLATE    Extra gsplat args before required flags.
                            Default: --data_dir "{scene256}" --result_dir "{scene256}/gaussian_splatting"
  --log-file PATH           Log file. Default: <target-root>/train_mvimgnet_splatformer_ready.log
  --dry-run                 Print commands without running them.
  --skip-rewrite-bins       Do not rewrite sparse/0/*.bin from colmap_work text outputs.
  -h, --help                Show this help.

The gsplat command always appends:
  --disable_viewer --steps_scaler 0.5 --data_factor 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-root|--root)
      ROOT="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --scene-limit|--limit)
      SCENE_LIMIT="$2"
      shift 2
      ;;
    --scene-list)
      SCENE_LIST="$2"
      shift 2
      ;;
    --colmap-script)
      COLMAP_SCRIPT="$2"
      shift 2
      ;;
    --colmap-python)
      COLMAP_PYTHON="$2"
      shift 2
      ;;
    --colmap-env-bin)
      COLMAP_ENV_BIN="$2"
      shift 2
      ;;
    --rewrite-bins-script)
      REWRITE_BINS_SCRIPT="$2"
      shift 2
      ;;
    --colmap-args)
      COLMAP_ARGS_TEMPLATE="$2"
      shift 2
      ;;
    --gsplat-trainer)
      GSPLAT_TRAINER="$2"
      shift 2
      ;;
    --gsplat-args)
      GSPLAT_ARGS_TEMPLATE="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-rewrite-bins)
      SKIP_REWRITE_BINS=1
      shift
      ;;
    -h|--help)
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

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="${ROOT%/}/train_mvimgnet_splatformer_ready.log"
fi

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

render_template() {
  local template="$1"
  local scene="$2"
  local scene256="$3"
  local rendered="$template"
  rendered="${rendered//\{scene256\}/$scene256}"
  rendered="${rendered//\{scene\}/$scene}"
  rendered="${rendered//\{root\}/$ROOT}"
  printf '%s' "$rendered"
}

run_command() {
  local command="$1"
  log "RUN: CUDA_VISIBLE_DEVICES=${GPU} QT_QPA_PLATFORM=offscreen ${command}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU" QT_QPA_PLATFORM=offscreen bash -lc "$command" >> "$LOG_FILE" 2>&1
}

with_optional_bin_path() {
  local command="$1"
  if [[ -n "$COLMAP_ENV_BIN" ]]; then
    printf 'PATH="%s:$PATH" %s' "$COLMAP_ENV_BIN" "$command"
  else
    printf '%s' "$command"
  fi
}

has_colmap_bins() {
  local scene256="$1"
  [[ -s "$scene256/sparse/0/cameras.bin" && -s "$scene256/sparse/0/images.bin" && -f "$scene256/sparse/0/points3D.bin" ]]
}

has_required_text_inputs() {
  local scene256="$1"
  [[ -d "$scene256/images" && -s "$scene256/cameras.txt" && -s "$scene256/images.txt" ]]
}

rewrite_legacy_bins() {
  local scene256="$1"
  if [[ "$SKIP_REWRITE_BINS" -eq 1 ]]; then
    log "Skipping legacy COLMAP bin rewrite"
    return 0
  fi
  local text_model_dir="$scene256/colmap_work/triangulated/sparse/0"
  if [[ ! -s "$text_model_dir/cameras.txt" || ! -s "$text_model_dir/images.txt" || ! -s "$text_model_dir/points3D.txt" ]]; then
    return 0
  fi
  run_command "\"$COLMAP_PYTHON\" \"$REWRITE_BINS_SCRIPT\" --text-model-dir \"$text_model_dir\" --output-dir \"$scene256/sparse/0\""
}

resolve_scene_path() {
  local entry="$1"
  if [[ "$entry" = /* ]]; then
    printf '%s' "$entry"
  elif [[ -d "$ROOT/$entry" ]]; then
    printf '%s' "$ROOT/$entry"
  elif [[ "$entry" == */* ]]; then
    printf '%s' "$ROOT/${entry//\//__}"
  else
    printf '%s' "$ROOT/$entry"
  fi
}

if [[ ! -d "$ROOT" ]]; then
  log "ERROR: target root does not exist: $ROOT"
  exit 1
fi

SCENES=()
if [[ -n "$SCENE_LIST" ]]; then
  if [[ ! -f "$SCENE_LIST" ]]; then
    log "ERROR: scene list does not exist: $SCENE_LIST"
    exit 1
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    SCENES+=("$(resolve_scene_path "$line")")
  done < "$SCENE_LIST"
else
  for scene_dir in "$ROOT"/*; do
    [[ -d "$scene_dir" ]] && SCENES+=("$scene_dir")
  done
fi

if [[ -n "$SCENE_LIMIT" ]]; then
  SCENES=("${SCENES[@]:0:$SCENE_LIMIT}")
fi

log "Target root: $ROOT"
log "Scenes queued: ${#SCENES[@]}"
log "Log file: $LOG_FILE"
if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Mode: dry-run (commands are printed but not executed)"
fi

PROCESSED=0
SKIPPED=0
FAILED=0
FAILURES=()

for scene_dir in "${SCENES[@]}"; do
  scene_name="$(basename "$scene_dir")"
  scene256="$scene_dir/256"
  ckpt="$scene256/gaussian_splatting/ckpts/ckpt_14999_rank0.pt"

  log "Scene: $scene_name"

  if ! has_required_text_inputs "$scene256"; then
    log "SKIP: missing required 256/images, 256/cameras.txt, or 256/images.txt"
    SKIPPED=$((SKIPPED + 1))
    FAILURES+=("$scene_name: missing required 256 inputs")
    continue
  fi

  if has_colmap_bins "$scene256"; then
    log "COLMAP bins already exist; skipping colmap step"
    rewrite_legacy_bins "$scene256"
  else
    if [[ "$DRY_RUN" -eq 0 ]]; then
      rm -rf "$scene256/colmap_work"
    fi
    if [[ -n "$COLMAP_ARGS_TEMPLATE" ]]; then
      colmap_args="$(render_template "$COLMAP_ARGS_TEMPLATE" "$scene_dir" "$scene256")"
      colmap_cmd="$(with_optional_bin_path "\"$COLMAP_PYTHON\" \"$COLMAP_SCRIPT\" $colmap_args")"
    else
      colmap_cmd="$(with_optional_bin_path "\"$COLMAP_PYTHON\" \"$COLMAP_SCRIPT\" --colmap_dir \"$scene256/colmap_work\" --train_images_dir \"$scene256/images\" --set_dir \"$scene256\" --colmap_gpu \"$GPU\" --image_dir \"$scene256/images\"")"
    fi
    if ! run_command "$colmap_cmd"; then
      log "FAIL: colmap step failed"
      FAILED=$((FAILED + 1))
      FAILURES+=("$scene_name: colmap step failed")
      continue
    fi
    if [[ "$DRY_RUN" -eq 0 ]] && ! has_colmap_bins "$scene256"; then
      log "FAIL: colmap step did not produce 256/sparse/0/*.bin"
      FAILED=$((FAILED + 1))
      FAILURES+=("$scene_name: missing COLMAP bin outputs")
      continue
    fi
    rewrite_legacy_bins "$scene256"
  fi

  if [[ -s "$ckpt" ]]; then
    log "Checkpoint already exists: $ckpt"
    PROCESSED=$((PROCESSED + 1))
    continue
  fi

  gsplat_args="$(render_template "$GSPLAT_ARGS_TEMPLATE" "$scene_dir" "$scene256")"
  gsplat_cmd="$GSPLAT_TRAINER $gsplat_args --disable_viewer --steps_scaler 0.5 --data_factor 1"
  if ! run_command "$gsplat_cmd"; then
    log "FAIL: gsplat training failed"
    FAILED=$((FAILED + 1))
    FAILURES+=("$scene_name: gsplat training failed")
    continue
  fi

  if [[ "$DRY_RUN" -eq 0 ]] && [[ ! -s "$ckpt" ]]; then
    log "FAIL: missing checkpoint after training: $ckpt"
    FAILED=$((FAILED + 1))
    FAILURES+=("$scene_name: missing ckpt_14999_rank0.pt")
    continue
  fi

  PROCESSED=$((PROCESSED + 1))
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN DONE: $scene_name"
  else
    log "DONE: $scene_name"
  fi
done

log "Summary: processed=${PROCESSED} skipped=${SKIPPED} failed=${FAILED}"
if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  log "Failure list:"
  for failure in "${FAILURES[@]}"; do
    log "  - $failure"
  done
fi

if [[ "$FAILED" -gt 0 || "$SKIPPED" -gt 0 ]]; then
  exit 1
fi
