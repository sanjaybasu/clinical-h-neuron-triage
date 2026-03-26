#!/usr/bin/env python3
"""Modal-based orchestrator: monitors Phase 7 completion and auto-launches
Phases 8, 9, 10, and random controls.  Runs entirely on Modal (CPU) so the
local laptop can sleep.

Deploy once:  modal deploy orchestrator.py
Trigger:      /opt/anaconda3/bin/python3 -c "
    import modal
    fn = modal.Function.from_name('h-neuron-orchestrator', 'orchestrate')
    fn.spawn()
"
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("h-neuron-orchestrator")
results_volume = modal.Volume.from_name("h-neuron-results-v3", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("modal")

POLL_INTERVAL = 300  # 5 minutes


def _file_exists(vol_path: str) -> bool:
    """Check if a file exists on the results volume."""
    return Path(vol_path).exists()


@app.function(
    image=image,
    timeout=86400,  # 24 hours (max allowed; re-spawn if needed)
    volumes={"/results": results_volume},
)
def orchestrate():
    """Poll the volume for completion markers and spawn downstream phases."""
    import modal as _modal

    pipeline_app = "h-neuron-triage-v4"
    orchestrator_app = "h-neuron-orchestrator"
    spawned = set()
    phase7_last_cases = 0
    phase7_stall_count = 0
    STALL_THRESHOLD = 6  # 6 polls × 5 min = 30 min with no progress → relaunch
    SELF_RENEW_AFTER = 82800  # 23 hours — renew before 24h timeout
    start_time = time.time()

    print("=" * 60)
    print("ORCHESTRATOR STARTED")
    print("Monitoring volume for phase completion markers...")
    print("=" * 60)

    while True:
        # Self-renewal: spawn a fresh orchestrator before timeout
        elapsed = time.time() - start_time
        if elapsed > SELF_RENEW_AFTER:
            print(f">>> Orchestrator approaching 24h timeout ({elapsed/3600:.1f}h). Spawning successor...")
            fn_self = _modal.Function.from_name(orchestrator_app, "orchestrate")
            fn_self.spawn()
            print(">>> Successor spawned. This instance exiting.")
            return
        # Refresh volume to see latest files
        results_volume.reload()

        phase7_done = _file_exists("/results/output/medical_h_neurons.json")
        phase8_done = _file_exists("/results/output/two_by_two_results.json")
        phase9_done = _file_exists("/results/output/vulnerability_results.json")
        phase10_done = _file_exists("/results/output/finetuned_logit_lens.json")
        random_done = all(
            _file_exists(f"/results/output/random_set_{i}.json")
            for i in range(100)
        )

        # Check partial progress for Phase 7
        partial_path = Path("/results/output/medical_probe_responses_partial.json")
        n_cases = 0
        if partial_path.exists():
            try:
                data = json.loads(partial_path.read_text())
                n_cases = len(data)
            except Exception:
                n_cases = 0

        status = (
            f"Phase 7: {'DONE' if phase7_done else f'running ({n_cases}/1280 cases)'} | "
            f"Phase 8: {'DONE' if phase8_done else ('pending' if 'phase8' not in spawned else 'running')} | "
            f"Phase 9: {'DONE' if phase9_done else ('pending' if 'phase9' not in spawned else 'running')} | "
            f"Phase 10: {'DONE' if phase10_done else ('pending' if 'phase10' not in spawned else 'running')} | "
            f"Random: {'DONE' if random_done else ('pending' if 'random' not in spawned else 'running')}"
        )
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {status}")

        # Auto-relaunch Phase 7 if stalled (timeout/preemption)
        if not phase7_done and n_cases > 0:
            if n_cases == phase7_last_cases:
                phase7_stall_count += 1
                if phase7_stall_count >= STALL_THRESHOLD:
                    print(f">>> Phase 7 stalled at {n_cases}/1280 for {STALL_THRESHOLD * POLL_INTERVAL // 60} min. Relaunching...")
                    fn7 = _modal.Function.from_name(pipeline_app, "run_medical_h_neurons")
                    fn7.spawn()
                    phase7_stall_count = 0
            else:
                phase7_stall_count = 0
            phase7_last_cases = n_cases

        # Phase 7 done → spawn Phase 8 (A100) + Phase 10 (L4)
        if phase7_done and "phase8" not in spawned and not phase8_done:
            print(">>> Phase 7 complete! Spawning Phase 8 (two_by_two, A100-80GB)...")
            fn8 = _modal.Function.from_name(pipeline_app, "run_two_by_two")
            fn8.spawn()
            spawned.add("phase8")

        if phase7_done and "phase10" not in spawned and not phase10_done:
            print(">>> Phase 7 complete! Spawning Phase 10 (finetuned_logit_lens, L4)...")
            fn10 = _modal.Function.from_name(pipeline_app, "run_finetuned_logit_lens")
            fn10.spawn()
            spawned.add("phase10")

        # Phase 8 done → spawn Phase 9 (CPU)
        if phase8_done and "phase9" not in spawned and not phase9_done:
            print(">>> Phase 8 complete! Spawning Phase 9 (vulnerability_analysis, CPU)...")
            fn9 = _modal.Function.from_name(pipeline_app, "run_vulnerability_analysis")
            fn9.spawn()
            spawned.add("phase9")

        # Phase 10 done → spawn random controls (L4, freed from Phase 10)
        if phase10_done and "random" not in spawned and not random_done:
            print(">>> Phase 10 complete! Spawning random controls (L4)...")
            fnr = _modal.Function.from_name(pipeline_app, "run_all_random_sets")
            fnr.spawn()
            spawned.add("random")

        # All done?
        if phase7_done and phase8_done and phase9_done and phase10_done:
            print("=" * 60)
            print("ALL PHASES COMPLETE!")
            print("=" * 60)
            return

        time.sleep(POLL_INTERVAL)
