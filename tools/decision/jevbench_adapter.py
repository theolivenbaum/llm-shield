"""JevBench adapter for Jevstral typed decisions, in the shape of jevbench's `laya_local` adapter.

Two backends, one prompt:
  csharp  the product: `jev decide <gguf> --serve`, a persistent subprocess running the
          .NET runtime (CPU, no native dependency), JSONL in and out
  torch   the research path: tools/decision/engine.py on bf16 weights, optionally with a LoRA

Both read Shieldstral's own yes/no verdict once per option and softmax the log-odds. The
probabilities are native (never verbalized). For noul, JevBench's no/yes map onto the
false/true criteria.

Usage, from this directory (the results and raw dirs must be outside the jevbench repo):
  python3 jevbench_adapter.py --backend csharp --model ~/models/Shieldstral-1.0-3B-Q8_0.gguf \\
      --run-dir ~/runs/jev-cs [--temps temps.json] [--threads 4]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JEVBENCH = Path(os.environ.get("JEVBENCH", "/home/user/jevbench"))
sys.path.insert(0, str(JEVBENCH))
sys.path.insert(0, str(HERE))

from jevbench.adapters.base import DecisionResult  # noqa: E402

CLI = HERE.parents[1] / "src/Jevstral.Cli/bin/Release/net10.0/jev.dll"


class JevstralAdapter:
    name = "jevstral_local"
    cost_basis = "local_cpu_no_provider_tariff"

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None, price_input_per_m=None,
                 price_output_per_m=None, backend="csharp", temps=None, lora=None, threads=4, revision=None):
        self.path = endpoint
        self.model = model or "mistralai/Shieldstral-1.0-3B"
        self.backend = backend
        self.temps = temps
        self.lora = lora
        self.threads = threads
        self.revision = revision
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self._proc = None
        self._decider = None

    def reserve_estimate(self, task):
        return 0.0

    def load(self):
        if self.backend == "csharp" and self._proc is None:
            cmd = ["dotnet", str(CLI), "decide", self.path, "--serve", "--threads", str(self.threads)]
            if self.temps:
                cmd += ["--temps", self.temps]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
        elif self.backend == "torch" and self._decider is None:
            import torch
            from engine import Decider
            from lora_io import load_lora
            from jevstral_torch import Shieldstral

            torch.set_num_threads(self.threads)
            state, cfg = load_lora(self.lora) if self.lora else (None, None)
            m = Shieldstral(self.path, lora=cfg).eval()
            if state:
                m.load_state_dict(state, strict=False)
            temps = json.loads(Path(self.temps).read_text()) if self.temps else None
            self._decider = Decider(m, temperature=temps)

    def _decide(self, task):
        if self.backend == "csharp":
            req = {"id": task.id, "state": task.state, "question": task.question, "labels": task.labels}
            self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            out = json.loads(self._proc.stdout.readline())
            return out["probs"], out["margins"], out.get("prefix_tokens", 0) + out.get("suffix_tokens", 0)
        labels = ["no", "yes"] if task.question["type"] == "noul" else task.labels
        res = self._decider.decide(task.question, labels, task.state)
        return res["probs"], res["margins"], None

    def run(self, task) -> DecisionResult:
        res = DecisionResult(adapter=self.name, ok=False, probs_source="native", model=self.model)
        res.request_body = {"state": task.state, "question": task.question}
        try:
            self.load()
        except Exception as e:  # noqa: BLE001 - a failed load is a failed attempt
            res.error = f"load failed: {type(e).__name__}: {str(e)[:250]}"
            return res
        t0 = time.perf_counter()
        try:
            probs, margins, tokens = self._decide(task)
        except Exception as e:  # noqa: BLE001
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        res.latency_s = time.perf_counter() - t0
        if task.question["type"] == "noul":
            p = float(probs.get("yes", probs.get("true")))
            res.probs = {"no": 1.0 - p, "yes": p}
        else:
            res.probs = {k: float(probs[k]) for k in task.labels}
        res.ok = True
        res.usage = {"input_tokens": tokens, "output_tokens": 0} if tokens else {}
        res.raw = {"margins": margins, "runtime": {"backend": self.backend, "threads": self.threads,
                                                   "lora": self.lora, "temps": self.temps,
                                                   "probability_origin": "native-softmax-over-verdict-log-odds"}}
        return res


def main():
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.summarize import public_export, summarize
    from jevbench.tasks import load_jsonl

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="csharp", choices=["csharp", "torch"])
    ap.add_argument("--model", required=True, help="GGUF for csharp, model dir for torch")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tiers", default="easy,original,hard")
    ap.add_argument("--temps")
    ap.add_argument("--lora")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    run = Path(a.run_dir).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    tasks = []
    for t in a.tiers.split(","):
        tasks += load_jsonl(str(JEVBENCH / "datasets/public" / f"{t}.jsonl"))
    if a.limit:
        tasks = tasks[:a.limit]
    adapter = JevstralAdapter(endpoint=a.model, backend=a.backend, temps=a.temps, lora=a.lora,
                                 threads=a.threads, price_input_per_m=0, price_output_per_m=0)
    ledger = Ledger(str(run / "ledger.jsonl"))
    runner = Runner(adapter, ledger, run / "raw", default_reserve_usd=0.0)
    records = runner.run_all(tasks, results_path=run / "results.jsonl")
    summary = summarize(tasks, records)
    (run / "summary.json").write_text(json.dumps(public_export(summary, tasks, records), indent=1, default=str))
    print(json.dumps(summary.get("headline", summary), indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
