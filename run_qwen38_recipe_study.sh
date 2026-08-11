#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required." >&2
  exit 1
fi

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
STUDY_DIR=${STUDY_DIR:-qwen38-fsod-runs/final-recipe-study}
DREIDEL=RF100VL/rf20-vl-fsod/the-dreidel-project
ORION=RF100VL/rf20-vl-fsod/orionproducts
COMMON=(
  --concurrency 256
  --requests-per-minute 570
  --tokens-per-minute 900000
  --timeout-seconds 180
  --max-completion-tokens 8192
  --max-retries 3
  --temperature 0
)

mkdir -p "$STUDY_DIR"

run_resumable() {
  local attempt status
  status=1
  for attempt in 1 2 3 4 5 6; do
    "$@" && return 0
    status=$?
    echo "Invocation exited ${status}; resumable pass ${attempt}/6." >&2
    sleep 10
  done
  return "$status"
}

wait_for_file() {
  local path=$1
  while [[ ! -f "$path" ]]; do
    sleep 30
  done
}

# Do not contend with or alter the already-running Orion screen.
wait_for_file qwen38-fsod-runs/orion-box-count-ablation-v1/_SUCCESS.json

# The pre-existing extension session starts when Orion completes.  If that
# session exits without combining its artifacts, resume it here.
while tmux has-session -t qwen-exemplar-extension 2>/dev/null; do
  sleep 30
done
if [[ ! -f qwen38-fsod-runs/dreidel-exemplar-only-box-combined-v1/_SUCCESS.json ]]; then
  run_resumable bash run_qwen38_exemplar_only_extension.sh || exit $?
fi

# Establish the residual fixed-test noise floor before making any close-call
# selection. Each dataset gets five identical complete repeats with all known
# sampling controls fixed.
run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$DREIDEL" \
  --conditions qwen38-fsod-configs/noise-floor-multi-names.json \
  --output-dir "$STUDY_DIR/dreidel-noise-floor" \
  "${COMMON[@]}" || exit $?

run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$ORION" \
  --conditions qwen38-fsod-configs/noise-floor-multi-names.json \
  --output-dir "$STUDY_DIR/orion-noise-floor" \
  "${COMMON[@]}" || exit $?

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python analyze_qwen38_noise_floor.py \
  --run "dreidel=$STUDY_DIR/dreidel-noise-floor" \
  --run "orion=$STUDY_DIR/orion-noise-floor" \
  --output "$STUDY_DIR/noise_floor.json" || exit $?

# Complete the only missing full factorial: anonymous multi-class concepts.
run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$DREIDEL" \
  --conditions qwen38-fsod-configs/anonymous-multi-screen.json \
  --output-dir qwen38-fsod-runs/dreidel-anonymous-multi-screen-v1 \
  "${COMMON[@]}" || exit $?

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python prepare_qwen38_recipe_study.py prepare-screen \
  --named-summary qwen38-fsod-runs/dreidel-box-count-ablation-v1/comparison_summary.json \
  --anonymous-single-summary qwen38-fsod-runs/dreidel-exemplar-only-box-combined-v1/comparison_summary.json \
  --anonymous-multi-summary qwen38-fsod-runs/dreidel-anonymous-multi-screen-v1/comparison_summary.json \
  --noise-floor "$STUDY_DIR/noise_floor.json" \
  --dreidel-annotations "$DREIDEL/test/_annotations.coco.json" \
  --orion-annotations "$ORION/test/_annotations.coco.json" \
  --output-dir "$STUDY_DIR" || exit $?

representation=$(jq -r '.best_box.representation' "$STUDY_DIR/screen_selection.json")
box_count=$(jq -r '.best_box.box_count' "$STUDY_DIR/screen_selection.json")

run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python generate_qwen38_self_names.py \
  --dataset-dir "$DREIDEL" \
  --output-dir "$STUDY_DIR/dreidel-self-names" \
  --representation "$representation" \
  --box-count "$box_count" || exit $?

mapfile -t dreidel_ids < <(jq -r '.dreidel[]' "$STUDY_DIR/subset_image_ids.json")
run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$DREIDEL" \
  --conditions "$STUDY_DIR/self_name_screen_conditions.json" \
  --self-names-json "$STUDY_DIR/dreidel-self-names/self_names.json" \
  --output-dir "$STUDY_DIR/dreidel-self-name-screen" \
  --image-ids "${dreidel_ids[@]}" \
  "${COMMON[@]}" || exit $?

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python prepare_qwen38_recipe_study.py prepare-finalists \
  --self-name-screen-summary "$STUDY_DIR/dreidel-self-name-screen/comparison_summary.json" \
  --output-dir "$STUDY_DIR" || exit $?

# A small temperature-zero reasoning gate answers the remaining reasoning
# question without paying for full low-reasoning runs.
run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$DREIDEL" \
  --conditions "$STUDY_DIR/reasoning_gate_conditions.json" \
  --self-names-json "$STUDY_DIR/dreidel-self-names/self_names.json" \
  --output-dir "$STUDY_DIR/dreidel-reasoning-gate" \
  --image-ids "${dreidel_ids[@]}" \
  "${COMMON[@]}" || exit $?

# Generate the label-free names independently from Orion's train split so no
# test data or Dreidel vocabulary crosses datasets.
run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python generate_qwen38_self_names.py \
  --dataset-dir "$ORION" \
  --output-dir "$STUDY_DIR/orion-self-names" \
  --representation "$representation" \
  --box-count "$box_count" \
  --allow-shared-reference-images || exit $?

# These are the only full temperature-zero reruns: the shortlisted recipes.
run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$DREIDEL" \
  --conditions "$STUDY_DIR/finalist_conditions.json" \
  --self-names-json "$STUDY_DIR/dreidel-self-names/self_names.json" \
  --output-dir "$STUDY_DIR/dreidel-finalists" \
  "${COMMON[@]}" || exit $?

run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$ORION" \
  --conditions "$STUDY_DIR/finalist_conditions.json" \
  --self-names-json "$STUDY_DIR/orion-self-names/self_names.json" \
  --output-dir "$STUDY_DIR/orion-finalists" \
  --allow-shared-reference-images \
  "${COMMON[@]}" || exit $?

echo "All Qwen3.8 recipe-study inference stages completed successfully."
