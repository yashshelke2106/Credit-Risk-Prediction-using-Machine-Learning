"""One command that regenerates every number in README.md and MODEL_CARD.md.

    python run_eval.py

Runs, in order: the leakage audit, the WoE scorecard, the full evaluation
(which rewrites the README results block and the model artifacts), then
verify_readme.py, which fails if any hand-written number has drifted.
"""
import subprocess
import sys
import time

STEPS = ["leakage_audit.py", "build_scorecard.py", "evaluate.py", "verify_readme.py"]

t_all = time.perf_counter()
for step in STEPS:
    print(f"\n{'=' * 70}\n>>> python {step}\n{'=' * 70}", flush=True)
    t = time.perf_counter()
    rc = subprocess.call([sys.executable, step])
    print(f"<<< {step} finished in {time.perf_counter() - t:.0f}s (exit {rc})", flush=True)
    if rc != 0:
        sys.exit(f"FAILED at {step}")
print(f"\nAll steps passed in {(time.perf_counter() - t_all) / 60:.1f} min.")
