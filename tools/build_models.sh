#!/usr/bin/env bash
# Rebuilds the Jevstral model files from public checkpoints and the adapter in this repository.
#
#   tools/build_models.sh [OUT_DIR] [QUANTS]      e.g. tools/build_models.sh ~/jevstral "q8_0 q5_1 q4_0"
#
# Produces, in OUT_DIR:
#   Jevstral-1.0-3B-<Q>.gguf               Shieldstral 1.0 3B + adapters/jevstral-v1 folded in: the verdict
#                                          decider (JevstralDecider, `jev decide`)
#   Ministral-3-3B-Reasoning-<Q>.gguf      Mistral's reasoning checkpoint, unchanged: reason-then-decide
#                                          (ReasoningDecider, `jev reason`)
#   Ministral-3-3B-Reasoning.SYSTEM_PROMPT.txt   its trained system prompt, which `jev reason --system` needs
#   SHA256SUMS
#
# Dependencies: bash, sha256sum, python3 (3.10+) with numpy, safetensors and gguf
#   (pip install numpy safetensors gguf), and network access to huggingface.co (both repositories are public;
#   HF_TOKEN is used if set). The .NET 10 SDK is optional: with it, a one-decision smoke test runs at the end.
# No PyTorch and no GPU. Disk: 2 x 7.2 GiB for the bf16 checkpoints (in OUT_DIR/.work, removed with --clean), plus
# about 3.4 / 2.4 / 1.8 GiB per model at Q8_0 / Q5_1 / Q4_0. RAM: ~6 GB (the conversion widens one tensor at a time,
# the 131072 x 3072 embedding table being the largest).
#
# On WSL, a Windows drive (/mnt/c, /mnt/d, ...) works but is slow for the 15 GiB of downloads. Set
# JEVSTRAL_WORK to a Linux path (e.g. ~/.cache/jevstral-build) to keep the scratch files there, and point
# OUT_DIR wherever the results should land.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$REPO/build/models}"
QUANTS="${2:-q8_0 q5_1 q4_0}"
WORK="${JEVSTRAL_WORK:-$OUT/.work}"
mkdir -p "$OUT" "$WORK"

echo "== base checkpoints (Mistral format; resumable, size-checked)"
python3 "$REPO/tools/download_shieldstral.py" "$WORK/shieldstral"
python3 "$REPO/tools/download_shieldstral.py" "$WORK/ministral3b-reasoning" --repo mistralai/Ministral-3-3B-Reasoning-2512

for q in $QUANTS; do
  Q="$(echo "$q" | tr '[:lower:]' '[:upper:]')"
  echo "== Jevstral-1.0-3B-$Q.gguf (Shieldstral + adapters/jevstral-v1)"
  python3 "$REPO/tools/decision/merge_lora_to_gguf.py" "$WORK/shieldstral" "$REPO/adapters/jevstral-v1" \
      --outtype "$q" --outfile "$OUT/Jevstral-1.0-3B-$Q.gguf"
  echo "== Ministral-3-3B-Reasoning-$Q.gguf"
  python3 "$REPO/tools/convert_shieldstral_to_gguf.py" "$WORK/ministral3b-reasoning" \
      --outtype "$q" --outfile "$OUT/Ministral-3-3B-Reasoning-$Q.gguf"
done
cp "$WORK/ministral3b-reasoning/SYSTEM_PROMPT.txt" "$OUT/Ministral-3-3B-Reasoning.SYSTEM_PROMPT.txt"

first_q="$(echo "$QUANTS" | awk '{print toupper($1)}')"
if command -v dotnet >/dev/null 2>&1; then
echo "== smoke test: one decision through the C# runtime"
printf '%s\n' '{"id":"smoke","state":"Please cancel my order #4471, I do not need it anymore.","question":{"type":"choice","instructions":"Which intent does the message express?","criteria":{"track_order":"Wants to know where an order is","cancel_order":"Wants to cancel an order","billing_question":"Asks about a charge"}},"labels":["track_order","cancel_order","billing_question"],"expected":"cancel_order"}' > "$WORK/smoke.jsonl"
dotnet run --project "$REPO/src/Jevstral.Cli" -c Release -- decide "$OUT/Jevstral-1.0-3B-$first_q.gguf" --tasks "$WORK/smoke.jsonl" 2>&1 | tail -1
else
echo "== no dotnet on PATH: skipping the smoke test"
fi

(cd "$OUT" && sha256sum ./*.gguf ./*.txt > SHA256SUMS && cat SHA256SUMS)
if [[ "${3:-}" == "--clean" ]]; then rm -rf "$WORK"; fi
echo "done: $OUT"
