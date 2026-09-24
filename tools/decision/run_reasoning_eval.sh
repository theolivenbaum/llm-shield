#!/usr/bin/env bash
# The whole reason-then-decide evaluation, end to end, on one machine:
#
#   tools/decision/run_reasoning_eval.sh [OUT_DIR]
#
#   1. reasoning on the public JevBench tiers (easy, standard, hard) at MAX_THINK
#   2. reasoning on held-out synthetic items (calibration only; never JevBench)
#   3. the verdict decider (adapters/jevstral-v1) on the same synthetic items and on JevBench
#   4. fit_ensemble: T and w fitted on the synthetic items, then applied to JevBench
#   5. budget_study: accuracy at every lower thinking budget, from the saved reasoning
#   6. speed_reason (torch) and `jev reason` (C#): tokens/s and latency
#
# Every step appends to its JSONL and skips ids already there, so an interrupted run picks up
# where it stopped: run the script again. Steps whose output is complete are skipped.
#
# Inputs (environment):
#   MODELS      dir holding shieldstral/ and ministral3b-reasoning/ (bf16, tools/download_shieldstral.py)
#               and, for step 6, Ministral-3-3B-Reasoning-Q8_0.gguf (`jev download reasoning --to $MODELS`)
#   JEVBENCH    a checkout of theolivenbaum/jevbench (eval_jevbench.py reads it)
#   THREADS     torch threads (default: all cores)
#   MAX_THINK   thinking budget (default 4096; the budget study reports every budget below it)
#   KV_TOKENS   rows x (prompt + budget) held at once. The bf16 cache is ~106 KB/token; the default
#               30000 is ~3.2 GB. On a big machine raise it with BATCH so more rows decode per step.
#   BATCH       most rows decoded together (default 16)
#   SYN_N       synthetic calibration items (default 150)
#
# Dependencies: python3 with torch (CPU is fine; bf16 with AMX is what it was tuned on), numpy,
# safetensors, pyarrow, regex; the .NET 10 SDK for step 6's C# half. RAM: ~8 GB for the model plus
# the KV cache.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$REPO/build/eval}"
MODELS="${MODELS:-$HOME/models}"
export JEVBENCH="${JEVBENCH:-$HOME/jevbench}"
THREADS="${THREADS:-$(nproc)}"
MAX_THINK="${MAX_THINK:-4096}"
KV_TOKENS="${KV_TOKENS:-30000}"
BATCH="${BATCH:-16}"
SYN_N="${SYN_N:-150}"
REASON_DIR="$MODELS/ministral3b-reasoning"
SHIELD_DIR="$MODELS/shieldstral"
ADAPTER="$REPO/adapters/jevstral-v1"
export MALLOC_ARENA_MAX=2
mkdir -p "$OUT"
cd "$HERE"

for d in "$REASON_DIR" "$SHIELD_DIR" "$JEVBENCH/datasets/public"; do
  [[ -d "$d" ]] || { echo "error: $d is missing (see the header of this script)" >&2; exit 2; }
done

lines() { [[ -f "$1" ]] && wc -l < "$1" || echo 0; }
step() { echo; echo "== $*"; }

step "synthetic calibration set ($SYN_N items, seed 99; disjoint from the training seed)"
[[ -s "$OUT/syn_val.jsonl" ]] || python3 gen_synthetic.py --n $((SYN_N * 2)) --seed 99 --out "$OUT/syn_val.full.jsonl"
[[ -s "$OUT/syn_val.jsonl" ]] || head -n "$SYN_N" "$OUT/syn_val.full.jsonl" > "$OUT/syn_val.jsonl"

R=(--model-dir "$REASON_DIR" --threads "$THREADS" --max-think "$MAX_THINK" --kv-tokens "$KV_TOKENS" --batch "$BATCH")
step "1. reasoning, JevBench public tiers"
python3 reason.py "${R[@]}" --tiers easy,original,hard --out "$OUT/reason_jev.jsonl" 2>&1 | tee -a "$OUT/reason_jev.log"
step "2. reasoning, synthetic calibration items"
python3 reason.py "${R[@]}" --syn "$OUT/syn_val.jsonl" --out "$OUT/reason_syn.jsonl" 2>&1 | tee -a "$OUT/reason_syn.log"

V=(--model-dir "$SHIELD_DIR" --threads "$THREADS" --lora "$ADAPTER" --query full)
step "3. verdict decider, synthetic items and JevBench"
python3 eval_td.py "${V[@]}" --data "syn:$OUT/syn_val.jsonl" --out "$OUT/verdict_syn.jsonl" 2>&1 | tee -a "$OUT/verdict_syn.log"
python3 eval_jevbench.py "${V[@]}" --out "$OUT/verdict_jev.jsonl" 2>&1 | tee -a "$OUT/verdict_jev.log"

step "4. ensemble: fitted on synthetic items, applied to JevBench"
python3 fit_ensemble.py --reason "$OUT/reason_syn.jsonl" --verdict "$OUT/verdict_syn.jsonl" --out "$OUT/ensemble.json"
python3 fit_ensemble.py --apply "$OUT/ensemble.json" --reason "$OUT/reason_jev.jsonl" --verdict "$OUT/verdict_jev.jsonl" \
    --out "$OUT/ensemble_jev.jsonl" 2>&1 | tee "$OUT/ensemble_jev.log"

step "5. thinking budget against accuracy"
budgets="0,256,512,1024,1536,2048"
(( MAX_THINK > 2048 )) && budgets="$budgets,$MAX_THINK"
python3 budget_study.py --model-dir "$REASON_DIR" --threads "$THREADS" --runs "$OUT/reason_jev.jsonl" \
    --budgets "$budgets" --out "$OUT/budget_study.json" 2>&1 | tee "$OUT/budget_study.log"

step "6. speed"
python3 speed_reason.py --model-dir "$REASON_DIR" --threads "$THREADS" --rows 1,4,8,$BATCH --out "$OUT/speed_torch.json" \
    2>&1 | tee "$OUT/speed_torch.log"
GGUF="$MODELS/Ministral-3-3B-Reasoning-Q8_0.gguf"
if command -v dotnet >/dev/null && [[ -f "$GGUF" ]]; then
  python3 - "$JEVBENCH/datasets/public/original.jsonl" "$OUT/speed_tasks.jsonl" <<'PY'
import sys
lines = [l for l in open(sys.argv[1]) if l.strip()][:12]
open(sys.argv[2], "w").writelines(lines)
PY
  dotnet build "$REPO/src/Jevstral.Cli" -c Release -v q -nologo >/dev/null
  JEV="$REPO/src/Jevstral.Cli/bin/Release/net10.0/jev"
  for rows in 1 12; do
    limit=$(( rows == 1 ? 1 : 12 ))
    echo "-- jev reason, $rows row(s), 256 thinking tokens"
    "$JEV" reason "$GGUF" --tasks "$OUT/speed_tasks.jsonl" --limit "$limit" --rows "$rows" --max-think 256 \
        --out "$OUT/speed_cs_r$rows.jsonl" 2>&1 | tee "$OUT/speed_cs_r$rows.log"
  done
else
  echo "(skipping the C# half: needs dotnet and $GGUF)"
fi

step "summary"
python3 eval_jevbench.py --report "$OUT/reason_jev.jsonl" | tail -8
echo "results in $OUT: reason_jev.jsonl, ensemble_jev.log, budget_study.log, speed_*.log"
