#!/usr/bin/env python3
"""What does a plain "git clone + manually staged data/" actually contain?

Run inside the clone. Compares the clone against the working checkout for the
files a training run and its diagnostics need, and reports what .gitignore hides.
"""
import re
import subprocess
import sys
from pathlib import Path

CLONE = Path(sys.argv[1]).resolve()

tracked = set(
    subprocess.run(["git", "-C", str(CLONE), "ls-files"], capture_output=True, text=True, check=True)
    .stdout.split()
)
print("tracked files in clone:", len(tracked))

# Files a training run needs, plus the defaults the diagnostic scripts point at.
RUN_CRITICAL = [
    "CLAUDE.md",
    "memory.md",
    "requirements-cms.txt",
    "requirements.txt",
    "configs_joint/cms_Joint_runH_kneeKernel.yaml",
    "configs_joint/cms_Joint_runH_D3b.yaml",
    "configs_joint/cms_Joint_runH_A2frozen.yaml",
    "scripts_joint/run_joint.py",
    "scripts_joint/joint_trainer.py",
    "scripts_joint/joint_model.py",
    "scripts_joint/joint_data.py",
    "scripts_joint/kernel_knee_fit.py",
    "scripts_joint/kernel_joint_calibration.py",
    "scripts_joint/upsilon_map_spread_audit.py",
    "scripts_joint/s2c_peak_fit.py",
    "scripts_joint/preflight.sh",
    "scripts_sota/cylindrical_flow.py",
    "scripts/cms_data.py",
    "tests/test_kernel_knee.py",
    "docs/project_tree.md",
    "paper/FRAMING.md",
]
print("\n=== presence in the clone ===")
missing = []
for rel in RUN_CRITICAL:
    present = rel in tracked
    print(f"  {'OK     ' if present else 'MISSING'}  {rel}")
    if not present:
        missing.append(rel)

# Defaults inside the diagnostic scripts that point at gitignored artifacts.
print("\n=== diagnostic-script defaults: do they exist in a clone? ===")
DEFAULTS = {
    "scripts_joint/kernel_knee_fit.py": [
        "outputs/cms_Joint/Run_H_cycleNoNoise/best_RunHcycleNoNoise_stage2_stochastic_core.pt",
        "outputs/cms_Joint/Run_H_cycleNoNoise/kernel_fit_linear/kernel_spec_fitted.json",
    ],
    "scripts_joint/kernel_joint_calibration.py": [
        "outputs/cms_Joint/Run_H_cycleNoNoise/best_RunHcycleNoNoise_stage2_stochastic_core.pt",
        "outputs/cms_Joint/Run_H_cycleNoNoise/kernel_fit_linear/kernel_spec_fitted.json",
    ],
    "scripts_joint/upsilon_map_spread_audit.py": [
        "outputs/cms_Joint/Run_H_D3b/best_model.pt",
        "data/upsilon_prior_continuumReweighted.hdf5",
    ],
    "scripts_joint/s2c_peak_fit.py": [
        "outputs/cms_Joint/Run_H_cycleNoNoise/best_RunHcycleNoNoise_stage2_stochastic_core.pt",
    ],
}
for script, defaults in DEFAULTS.items():
    src = (CLONE / script).read_text(encoding="utf-8") if (CLONE / script).exists() else ""
    for target in defaults:
        shipped = target in tracked
        staged = (CLONE / target).exists() or target.startswith("data/")
        verdict = "in git" if shipped else ("not in git; must be staged" if target.startswith("data/") else "NOT IN GIT")
        print(f"  {script.split('/')[-1]:34s} -> {target}")
        print(f"       {verdict}")

# What .gitignore hides, by category.
print("\n=== gitignored content that exists in the working tree but NOT in the clone ===")
import os
groups = {}
root = Path(r"C:/Users/AhrixMarin/Desktop/otus")
for base, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in {".git"}]
    for name in files:
        p = Path(base) / name
        rel = p.relative_to(root).as_posix()
        if rel in tracked:
            continue
        if rel.startswith("data/") or rel.startswith("data\\"):
            groups.setdefault("data/ (staged manually by design)", []).append(rel)
        elif rel.startswith("outputs/"):
            groups.setdefault("outputs/ (gitignored: *.json, *.npz, *.pt, *.hdf5)", []).append(rel)
        elif rel.startswith("docs/") or rel.startswith("deliverables/"):
            groups.setdefault("docs|deliverables (gitignored by directory rule)", []).append(rel)
        elif name.endswith(".md"):
            groups.setdefault("*.md outside docs/ (gitignored by extension)", []).append(rel)
        else:
            groups.setdefault("other", []).append(rel)
for key in sorted(groups):
    items = groups[key]
    print(f"  {key}: {len(items)}")
    for item in sorted(items)[:12]:
        print(f"       {item}")
    if len(items) > 12:
        print(f"       ... and {len(items)-12} more")
print("\nMISSING-COUNT", len(missing))
