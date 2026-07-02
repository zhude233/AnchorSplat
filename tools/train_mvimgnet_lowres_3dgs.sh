#!/usr/bin/env bash

set -u

ROOT="data/mvimgnet_raw_lowres_3dgs"
TRAIN_PY="third_party/SuperGaussian/third_parties/gaussian-splatting/train.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="0"
SCENE_LIMIT=""
SCENE_LIST=""
OUTPUT_DIRNAME="raw_lowres_3dgs_train"
POINT_COUNT="131072"
ITERATIONS=""
EXTRA_ARGS=()
LOG_FILE=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash tools/train_mvimgnet_lowres_3dgs.sh [options] [-- extra train.py args]

Options:
  --root PATH              Root produced by prepare_mvimgnet_lowres_3dgs.py.
  --scene PATH_OR_NAME     Train one scene path/name. Can be repeated.
  --scene-list PATH        Text file with scene names or paths, one per line.
  --scene-limit N          Limit scenes from --root or --scene-list.
  --train-py PATH          SuperGaussian fork gaussian-splatting/train.py.
  --python PATH            Python executable. Default: $PYTHON_BIN or python.
  --gpu ID                 CUDA_VISIBLE_DEVICES value. Default: 0.
  --output-dirname NAME    Per-scene output folder. Default: raw_lowres_3dgs_train.
  --num-of-gaussians N     Passed to train.py. Default: 131072.
  --iterations N           Optional short/long run override for train.py.
  --log-file PATH          Log file. Default: <root>/train_mvimgnet_lowres_3dgs.log.
  --dry-run                Print commands without running them.
  -h, --help               Show this help.

The wrapper always calls the SuperGaussian fork train.py with:
  --use_low_res_as_gt --no_gt_eval 1 --num_of_gaussians <N> -r 1

It intentionally bypasses main_supergaussian.py, VSR, 256 fitting, and
SplatFormer gsplat simple_trainer.
EOF
}

SCENES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --scene)
      SCENES+=("$2")
      shift 2
      ;;
    --scene-list)
      SCENE_LIST="$2"
      shift 2
      ;;
    --scene-limit|--limit)
      SCENE_LIMIT="$2"
      shift 2
      ;;
    --train-py)
      TRAIN_PY="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --output-dirname)
      OUTPUT_DIRNAME="$2"
      shift 2
      ;;
    --num-of-gaussians|--point-count)
      POINT_COUNT="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
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
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
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
  LOG_FILE="${ROOT%/}/train_mvimgnet_lowres_3dgs.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

resolve_scene_path() {
  local entry="$1"
  if [[ "$entry" = /* ]]; then
    printf '%s' "$entry"
  else
    printf '%s' "${ROOT%/}/$entry"
  fi
}

run_command() {
  local command="$1"
  log "RUN: CUDA_VISIBLE_DEVICES=${GPU} ${command}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU" bash -lc "$command" >> "$LOG_FILE" 2>&1
}

expected_final_iteration() {
  local iterations="${ITERATIONS:-30000}"
  local warmup_iterations="0"
  local idx=0
  while [[ "$idx" -lt "${#EXTRA_ARGS[@]}" ]]; do
    case "${EXTRA_ARGS[$idx]}" in
      --iterations)
        idx=$((idx + 1))
        [[ "$idx" -lt "${#EXTRA_ARGS[@]}" ]] && iterations="${EXTRA_ARGS[$idx]}"
        ;;
      --iterations=*)
        iterations="${EXTRA_ARGS[$idx]#--iterations=}"
        ;;
      --warmup_iterations)
        idx=$((idx + 1))
        [[ "$idx" -lt "${#EXTRA_ARGS[@]}" ]] && warmup_iterations="${EXTRA_ARGS[$idx]}"
        ;;
      --warmup_iterations=*)
        warmup_iterations="${EXTRA_ARGS[$idx]#--warmup_iterations=}"
        ;;
    esac
    idx=$((idx + 1))
  done
  printf '%s' "$((iterations + warmup_iterations))"
}

if [[ ! -f "$TRAIN_PY" ]]; then
  log "ERROR: train.py not found: $TRAIN_PY"
  exit 1
fi
if [[ ! -d "$ROOT" && "${#SCENES[@]}" -eq 0 ]]; then
  log "ERROR: root does not exist: $ROOT"
  exit 1
fi

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
fi

if [[ "${#SCENES[@]}" -eq 0 ]]; then
  for scene_dir in "${ROOT%/}"/*; do
    [[ -d "$scene_dir" ]] && SCENES+=("$scene_dir")
  done
else
  for idx in "${!SCENES[@]}"; do
    SCENES[$idx]="$(resolve_scene_path "${SCENES[$idx]}")"
  done
fi

if [[ -n "$SCENE_LIMIT" ]]; then
  SCENES=("${SCENES[@]:0:$SCENE_LIMIT}")
fi

log "Root: $ROOT"
log "Scenes queued: ${#SCENES[@]}"
log "train.py: $TRAIN_PY"
log "Log file: $LOG_FILE"
[[ "$DRY_RUN" -eq 1 ]] && log "Mode: dry-run"

PROCESSED=0
FAILED=0
SKIPPED=0
FAILURES=()

for scene_dir in "${SCENES[@]}"; do
  scene_name="$(basename "$scene_dir")"
  resolution_low="$scene_dir/resolution_low"
  output_dir="$scene_dir/$OUTPUT_DIRNAME"
  final_iteration="$(expected_final_iteration)"
  final_ply="$output_dir/point_cloud/iteration_${final_iteration}/point_cloud.ply"

  log "Scene: $scene_name"
  if [[ ! -s "$resolution_low/transforms.json" || ! -d "$resolution_low/images" ]]; then
    log "SKIP: missing resolution_low/transforms.json or images"
    SKIPPED=$((SKIPPED + 1))
    FAILURES+=("$scene_name: missing prepared MVImgNet low-res 3DGS inputs")
    continue
  fi
  if [[ ! -s "$resolution_low/surface_pcd_${POINT_COUNT}_seed_0.ply" ]]; then
    log "SKIP: missing surface_pcd_${POINT_COUNT}_seed_0.ply"
    SKIPPED=$((SKIPPED + 1))
    FAILURES+=("$scene_name: missing initialization PLY")
    continue
  fi

  command="\"$PYTHON_BIN\" \"$TRAIN_PY\" -s \"$scene_dir\" --exp_name \"$output_dir\" --use_low_res_as_gt --no_gt_eval 1 --num_of_gaussians \"$POINT_COUNT\" -r 1"
  if [[ -n "$ITERATIONS" ]]; then
    command="$command --iterations \"$ITERATIONS\""
  fi
  for arg in "${EXTRA_ARGS[@]}"; do
    command="$command $(printf '%q' "$arg")"
  done

  if ! run_command "$command"; then
    log "FAIL: fork training failed"
    FAILED=$((FAILED + 1))
    FAILURES+=("$scene_name: training failed")
    continue
  fi

  if [[ "$DRY_RUN" -eq 0 && ! -s "$final_ply" ]]; then
    log "FAIL: expected trained PLY missing: $final_ply"
    FAILED=$((FAILED + 1))
    FAILURES+=("$scene_name: missing trained PLY")
    continue
  fi

  PROCESSED=$((PROCESSED + 1))
  [[ "$DRY_RUN" -eq 1 ]] && log "DRY-RUN DONE: $scene_name" || log "DONE: $scene_name"
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
