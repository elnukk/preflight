---
name: preflight
description: Estimate and profile an experiment (script or sbatch file) before launching it.
argument-hint: "[script-or-sbatch] [launch command]"
disable-model-invocation: true
---

Run a Preflight check on: $ARGUMENTS

Follow the `compute-efficiency` skill throughout.

1. **Locate the files.** Resolve each path against the current working directory. If one is not there, search the
   project for it (Glob `**/<name>`). If several match, list them and ask which one; if none match, ask for the path.
   If no file was given, look for sbatch files and training scripts in the current directory and ask which to check.
2. **Load the study.** Find the project root (nearest folder with `.preflight/`, else the script's folder) and read
   `.preflight/study.yaml`. If it is missing, ask the researcher for a study name, the outputs that must not change,
   and a budget, then call `init_study` with their answers (it writes the file; do not edit it by hand).
3. **Read the code.** Read the script and any sbatch file. Work out the full-run dataset size, epochs and GPU count
   from the code and launch arguments.
4. **Profile.** Call `capabilities`, then `profile_run` with a subset big enough for at least 10 progress steps and
   10–60 s of runtime (`max_seconds` 120). If the script ignores `PREFLIGHT_SUBSET`, use its own subset or debug flag
   in `cmd`. If it prints no `[preflight]` progress lines, say so and suggest the two-line patch that adds them;
   do not apply it without asking.
5. **Estimate.** Call `estimate_job` with the `profile_id` and the numbers from step 3.
6. **Diagnose.** Call `diagnose` with the `profile_id`, `estimate_id`, `script_path` and `sbatch_path`, plus `include_history` when
   Slurm is available.
7. **Report** to the researcher:
   - a table of runtime, GPU-hours, $, kWh and kg CO2 (low / point / high) with the basis and assumptions;
   - every warning, saying plainly when telemetry is simulated;
   - findings grouped as failure risks, safe fixes, and fixes that may change results, each with its evidence and
     a proposed diff where one applies.
8. **Do not launch** the full job or apply any change. Ask the researcher what to do next.
