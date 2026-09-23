#!/usr/bin/env bash
# Rebuilds the Jevstral model files from public checkpoints and the adapter in this repository.
#
#   tools/build_models.sh [OUT_DIR] [QUANTS] [--clean]     e.g. tools/build_models.sh ~/jevstral "q8_0 q5_1 q4_0"
#
# Produces, in OUT_DIR:
#   Jevstral-1.0-3B-<Q>.gguf               Shieldstral 1.0 3B + adapters/jevstral-v1 folded in: the verdict
#                                          decider (JevstralDecider, `jev decide`)
#   Ministral-3-3B-Reasoning-<Q>.gguf      Mistral's reasoning checkpoint, unchanged: reason-then-decide
#                                          (ReasoningDecider, `jev reason`)
#   Ministral-3-3B-Reasoning.SYSTEM_PROMPT.txt   its trained system prompt, which `jev reason --system` needs
#   SHA256SUMS
#
# Dependencies: bash, sha256sum, df, python3 (3.10+) with numpy, safetensors and gguf
#   (pip install numpy safetensors gguf), and network access to huggingface.co (both repositories are public;
#   HF_TOKEN is used if set). The .NET 10 SDK is optional: with it, a one-decision smoke test runs at the end.
# No PyTorch and no GPU. RAM: ~6 GB (the conversion widens one tensor at a time).
#
# Everything heavy happens in a Linux-side working directory, JEVSTRAL_WORK (default ~/.cache/jevstral-build):
# 2 x 7.2 GiB of bf16 checkpoints, and the GGUFs are built and checksummed there (about 3.4 / 2.4 / 1.8 GiB per model
# at Q8_0 / Q5_1 / Q4_0). Only the finished files are copied to OUT_DIR and verified against SHA256SUMS there. On WSL
# this keeps the large sequential writes off /mnt/<drive>, whose 9p mounts can return short writes or EIO under them.
# --clean removes the working directory at the end.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$REPO/build/models}"
QUANTS="${2:-q8_0 q5_1 q4_0}"
WORK="${JEVSTRAL_WORK:-$HOME/.cache/jevstral-build}"
STAGE="$WORK/stage"
mkdir -p "$OUT" "$WORK" "$STAGE"

# Checkpoints an earlier run of this script downloaded under OUT_DIR/.work are reused where they are.
dl_dir() {
  local name="$1"
  if [[ -z "${JEVSTRAL_WORK:-}" && -s "$OUT/.work/$name/consolidated.safetensors" ]]; then
    echo "$OUT/.work/$name"
  else
    echo "$WORK/$name"
  fi
}
SHIELD="$(dl_dir shieldstral)"
REASON="$(dl_dir ministral3b-reasoning)"

free_gib() { df -Pk "$1" | awk 'NR==2 {printf "%d", $4 / 1048576}'; }
need=0
for q in $QUANTS; do
  case "$q" in q8_0) need=$((need + 7));; q5_1) need=$((need + 5));; *) need=$((need + 4));; esac
done
[[ -s "$SHIELD/consolidated.safetensors" ]] || need=$((need + 8))
[[ -s "$REASON/consolidated.safetensors" ]] || need=$((need + 8))
if (( $(free_gib "$WORK") < need )); then
  echo "error: $WORK has $(free_gib "$WORK") GiB free; this build needs about $need GiB. Set JEVSTRAL_WORK elsewhere." >&2
  exit 2
fi

echo "== base checkpoints (Mistral format; resumable, size-checked)"
python3 "$REPO/tools/download_shieldstral.py" "$SHIELD"
python3 "$REPO/tools/download_shieldstral.py" "$REASON" --repo mistralai/Ministral-3-3B-Reasoning-2512

for q in $QUANTS; do
  Q="$(echo "$q" | tr '[:lower:]' '[:upper:]')"
  echo "== Jevstral-1.0-3B-$Q.gguf (Shieldstral + adapters/jevstral-v1)"
  python3 "$REPO/tools/decision/merge_lora_to_gguf.py" "$SHIELD" "$REPO/adapters/jevstral-v1" \
      --outtype "$q" --outfile "$STAGE/Jevstral-1.0-3B-$Q.gguf"
  echo "== Ministral-3-3B-Reasoning-$Q.gguf"
  python3 "$REPO/tools/convert_shieldstral_to_gguf.py" "$REASON" \
      --outtype "$q" --outfile "$STAGE/Ministral-3-3B-Reasoning-$Q.gguf"
done
cp "$REASON/SYSTEM_PROMPT.txt" "$STAGE/Ministral-3-3B-Reasoning.SYSTEM_PROMPT.txt"

first_q="$(echo "$QUANTS" | awk '{print toupper($1)}')"
if command -v dotnet >/dev/null 2>&1; then
  echo "== smoke test: one decision through the C# runtime"
  printf '%s\n' '{"id":"smoke","state":"Please cancel my order #4471, I do not need it anymore.","question":{"type":"choice","instructions":"Which intent does the message express?","criteria":{"track_order":"Wants to know where an order is","cancel_order":"Wants to cancel an order","billing_question":"Asks about a charge"}},"labels":["track_order","cancel_order","billing_question"],"expected":"cancel_order"}' > "$WORK/smoke.jsonl"
  dotnet run --project "$REPO/src/Jevstral.Cli" -c Release -- decide "$STAGE/Jevstral-1.0-3B-$first_q.gguf" --tasks "$WORK/smoke.jsonl" 2>&1 | tail -1
else
  echo "== no dotnet on PATH: skipping the smoke test"
fi

(cd "$STAGE" && sha256sum ./*.gguf ./*.txt > SHA256SUMS && cat SHA256SUMS)

if [[ "$(cd "$OUT" && pwd -P)" != "$(cd "$STAGE" && pwd -P)" ]]; then
  echo "== copying to $OUT"
  for f in "$STAGE"/*.gguf "$STAGE"/*.txt "$STAGE/SHA256SUMS"; do
    cp "$f" "$OUT/$(basename "$f").partial"
    mv -f "$OUT/$(basename "$f").partial" "$OUT/$(basename "$f")"
  done
  (cd "$OUT" && sha256sum -c SHA256SUMS) || {
    echo "error: the copies in $OUT do not match their checksums; the verified originals are in $STAGE" >&2
    exit 1
  }
fi
if [[ "${3:-}" == "--clean" ]]; then rm -rf "$WORK"; fi
echo "done: $OUT"
