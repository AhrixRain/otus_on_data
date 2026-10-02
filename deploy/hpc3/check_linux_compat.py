#!/usr/bin/env python3
"""Verify that this checkout will actually run on the Linux cluster.

Run it *on the cluster*, immediately after the first ``scp``/``rsync`` and
inside the ``otus-cms`` environment, before spending a GPU-hour:

    srun -c 4 -p free --mem=8G --time=00:30:00 --pty /bin/bash -i
    module purge && module load miniconda3/23.5.2
    source ~/.mycondainit-23.5.2
    conda activate /pub/$USER/otus-conda/envs/otus-cms
    cd /pub/$USER/otus
    python deploy/hpc3/check_linux_compat.py

It is also safe to run on the Windows workstation; the platform-specific
checks then simply report what they see instead of asserting Linux.

Every check prints PASS / WARN / FAIL. The exit status is non-zero if and only
if something FAILed, so it can gate a batch script:

    python deploy/hpc3/check_linux_compat.py || exit 1
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# The CMS open data file this pipeline reads, and the CERN record it comes from.
# Size and adler32 are from https://opendata.cern.ch/api/records/12341.
ROOT_NAME = "Run2012BC_DoubleMuParked_Muons.root"
ROOT_BYTES = 2244449133
ROOT_ADLER32 = "1fa61aca"

# Imported by the joint training path. Missing any of these is fatal for a run.
REQUIRED_IMPORTS = [
    "numpy", "scipy", "pandas", "matplotlib", "seaborn",
    "h5py", "uproot", "awkward", "vector", "hist", "boost_histogram",
    "tqdm", "torch",
]

# Needed by the training path. Missing these is fatal too.
REQUIRED_PATHS = [
    "scripts_joint/run_joint.py",
    "scripts_joint/joint_trainer.py",
    "scripts_joint/joint_data.py",
    "scripts_joint/identity_baseline.py",
    "configs_joint/cms_Joint_runH.yaml",
    "configs_joint/cms_Joint_runE.yaml",
    "tests",
    "requirements-cms.txt",
]

RESULTS: list[tuple[str, str]] = []


def record(status: str, message: str) -> None:
    RESULTS.append((status, message))
    print(f"[{status:4}] {message}")


def check_platform() -> None:
    print("\n== platform ==")
    record("INFO", f"hostname      : {platform.node()}")
    record("INFO", f"os.name       : {os.name}  ({sys.platform})")
    record("INFO", f"platform      : {platform.platform()}")
    record("INFO", f"python        : {sys.version.split()[0]}  ({sys.executable})")
    record("INFO", f"cwd           : {os.getcwd()}")

    home = Path.home().resolve()
    cwd = Path.cwd().resolve()
    if os.name != "nt" and (cwd == home or home in cwd.parents):
        record(
            "WARN",
            f"the checkout is under $HOME ({home}). HPC3 policy: "
            "'Do not run Slurm jobs in your $HOME. Instead, use your DFS storage "
            "/pub/UCInetID' -- and $HOME has a 50 GB quota shared with snapshots.",
        )
    elif os.name != "nt":
        record("PASS", f"checkout is outside $HOME ({cwd})")

    if os.name != "nt":
        total = avail = None
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024 / 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024 / 1024
        except OSError:
            pass
        if total:
            record("INFO", f"memory        : {avail:.1f} GB available of {total:.1f} GB")
        record("INFO", f"cpus          : {os.cpu_count()}")
        if os.environ.get("SLURM_JOB_ID"):
            record("INFO", f"slurm job     : {os.environ['SLURM_JOB_ID']} "
                           f"on {os.environ.get('SLURMD_NODENAME', '?')} "
                           f"({os.environ.get('SLURM_JOB_PARTITION', '?')})")
        else:
            record("WARN", "SLURM_JOB_ID is unset -- this is a login node or an "
                           "interactive shell, not a batch job")
        if os.environ.get("DISPLAY"):
            record("INFO", f"DISPLAY       : {os.environ['DISPLAY']} (X forwarding active)")


def check_line_endings() -> None:
    print("\n== line endings (a scp/rsync copy does not normalise them) ==")
    try:
        sys.path.insert(0, str(HERE))
        from to_lf import analyse, iter_text_files  # type: ignore
    except ImportError as exc:  # pragma: no cover
        record("WARN", f"could not import to_lf.py ({exc}); skipping")
        return

    offenders = []
    scanned = 0
    for path in iter_text_files(REPO_ROOT, include_artifacts=False):
        scanned += 1
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\r" not in data:
            continue
        crlf, lone = analyse(data)
        if crlf or lone:
            offenders.append((path.relative_to(REPO_ROOT).as_posix(), crlf, lone))

    if offenders:
        record("FAIL", f"{len(offenders)} of {scanned} scanned text files still contain CRLF")
        for rel, crlf, lone in offenders[:10]:
            record("INFO", f"          {rel} ({crlf} CRLF, {lone} lone CR)")
        if len(offenders) > 10:
            record("INFO", f"          ... and {len(offenders) - 10} more")
        record("INFO", "fix with: python deploy/hpc3/to_lf.py --apply")
    else:
        record("PASS", f"all {scanned} scanned text files use LF endings")


def check_imports() -> list[dict]:
    print("\n== python packages (requirements-cms.txt) ==")
    versions = []
    missing = []
    for name in REQUIRED_IMPORTS:
        try:
            module = __import__(name)
        except Exception as exc:  # noqa: BLE001 - report whatever went wrong
            missing.append(name)
            record("FAIL", f"{name:<16} import failed: {type(exc).__name__}: {exc}")
            continue
        version = getattr(module, "__version__", "?")
        versions.append({"name": name, "version": str(version)})
        record("PASS", f"{name:<16} {version}")
    if missing:
        record("INFO", "install with: pip install --extra-index-url "
                       "https://download.pytorch.org/whl/cu126 -r requirements-cms.txt")
    return versions


def _arch_supports(arch_list: list[str], capability: tuple[int, int]) -> str | None:
    """Which entry of ``torch.cuda.get_arch_list()`` covers this device?

    CUDA guarantees *minor-version* binary compatibility: a cubin built for
    ``sm_86`` runs on any ``sm_8x`` with x >= 6, which is why an sm_86-only
    PyTorch build trains happily on an sm_89 Ada card. So the device is covered
    if some listed arch has the same major version and a minor version no
    greater than the device's. Cross-major fallback (sm_90 cubin on sm_120) is
    not possible.
    """
    major, minor = capability
    best: tuple[int, str] | None = None
    for entry in arch_list:
        digits = entry.split("_", 1)[-1].rstrip("a")  # sm_90a -> 90
        if not digits.isdigit() or len(digits) < 2:
            continue
        entry_major, entry_minor = int(digits[:-1]), int(digits[-1])
        if entry_major != major or entry_minor > minor:
            continue
        if best is None or entry_minor > best[0]:
            best = (entry_minor, entry)
    return best[1] if best else None


def check_torch_cuda() -> dict:
    print("\n== torch / CUDA ==")
    info: dict = {}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        record("FAIL", f"torch is not importable: {exc}")
        return info

    record("INFO", f"torch         : {torch.__version__} (cuda {torch.version.cuda})")
    arch_list = []
    try:
        arch_list = list(torch.cuda.get_arch_list())
        record("INFO", f"built for     : {', '.join(arch_list) or '(none)'}")
    except Exception as exc:  # noqa: BLE001
        record("WARN", f"could not read torch.cuda.get_arch_list(): {exc}")
    info["torch"] = torch.__version__
    info["arch_list"] = arch_list

    available = False
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        record("WARN", f"torch.cuda.is_available() raised: {exc}")

    if not available:
        status = "WARN" if os.name == "nt" else "FAIL"
        record(status, "torch.cuda.is_available() is False -- no usable GPU on this node")
        record("INFO", "the training path needs one; submit with --gres=gpu:1 "
                       "(free-gpu is fine for smoke tests)")
        return info

    record("PASS", "torch.cuda.is_available() is True")
    try:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        sm = f"sm_{capability[0]}{capability[1]}"
        record("INFO", f"device        : {props.name} ({props.total_memory / 1024**3:.1f} GB, {sm})")
        info["device"] = props.name
        info["capability"] = sm
        # This is the V100 question: the pinned torch 2.12+cu126 still ships
        # sm_70 kernels, but a future torch or a gpu32 Blackwell node may not.
        if not arch_list:
            record("WARN", f"cannot confirm that the build covers {sm}")
        else:
            covering = _arch_supports(arch_list, capability)
            if covering:
                note = "" if covering == sm else f" via minor-version compatibility from {covering}"
                record("PASS", f"{sm} is covered by the torch build{note}")
            else:
                record("FAIL", f"the installed torch has no kernels for {sm}; it was built "
                               f"for {', '.join(arch_list)}. Pick a different GPU type "
                               f"(--gres=gpu:A30:1 / gpu:L40S:1 / gpu:A100:1) or change the pin.")
        free, total = torch.cuda.mem_get_info(index)
        record("INFO", f"free VRAM     : {free / 1024**3:.1f} GB of {total / 1024**3:.1f} GB "
                       f"(Run H peaked at 3.2 GB reserved)")
    except Exception as exc:  # noqa: BLE001
        record("WARN", f"could not query the device: {exc}")
    return info


def check_repo() -> None:
    print("\n== checkout sanity ==")
    missing = [rel for rel in REQUIRED_PATHS if not (REPO_ROOT / rel).exists()]
    if missing:
        for rel in missing:
            record("FAIL", f"missing: {rel}")
        record("INFO", "if you copied only part of the tree, re-run the push "
                       "(deploy/hpc3/push_code.ps1) or check the exclude list")
    else:
        record("PASS", f"all {len(REQUIRED_PATHS)} required paths are present")

    git_dir = REPO_ROOT / ".git"
    if git_dir.is_dir():
        try:
            head = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT,
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            record("INFO", f"git HEAD      : {head}"
                           + ("  (working tree has local changes)" if dirty else "  (clean)"))
        except Exception as exc:  # noqa: BLE001
            record("WARN", f"git present but unusable: {exc}")
    else:
        record("INFO", "no .git directory -- this is a plain copy, so provenance "
                       "and manual git on the cluster are unavailable here")


def check_data(data_root: Path, verify_checksum: bool) -> None:
    print(f"\n== data ({data_root}) ==")
    if not data_root.is_dir():
        record("FAIL", f"{data_root} does not exist; run deploy/hpc3/stage_data.sh")
        return

    root_file = data_root / ROOT_NAME
    if not root_file.is_file():
        record("FAIL", f"{ROOT_NAME} is missing -> run deploy/hpc3/stage_data.sh")
    else:
        size = root_file.stat().st_size
        if size == ROOT_BYTES:
            record("PASS", f"{ROOT_NAME} present, {size} bytes (expected {ROOT_BYTES})")
        else:
            record("FAIL", f"{ROOT_NAME} is {size} bytes, expected {ROOT_BYTES} -- "
                           f"delete it and re-run stage_data.sh")
        if verify_checksum:
            import zlib
            value = 1
            with root_file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    value = zlib.adler32(chunk, value)
            actual = format(value & 0xFFFFFFFF, "08x")
            if actual == ROOT_ADLER32:
                record("PASS", f"adler32 {actual} matches the CERN record 12341")
            else:
                record("FAIL", f"adler32 {actual} != {ROOT_ADLER32} (CERN record 12341)")

    priors = [
        "legacy/cms_jpsi_mumu_mg5_8tev_mixed_ptj5.hdf5",
        "cms_dymumu_mg5py8_ckkwl_8tev_inclusive_0j1j_fiducial_70_110_1M.hdf5",
    ]
    missing = [rel for rel in priors if not (data_root / rel).is_file()]
    if missing:
        for rel in missing:
            record("WARN", f"prior missing: data/{rel}")
        record("INFO", "priors are generated, not downloadable; copy them from the "
                       "workstation (see the runbook section 4.3)")
    else:
        record("PASS", "both priors the joint muon path reads are present")

    free = shutil.disk_usage(data_root)
    record("INFO", f"filesystem    : {free.free / 1024**3:.1f} GB free of "
                   f"{free.total / 1024**3:.1f} GB at {data_root}")


def run_tests() -> None:
    print("\n== unittest suite (this takes a few minutes) ==")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=REPO_ROOT,
    )
    if proc.returncode == 0:
        record("PASS", "unittest discover exited 0")
    else:
        record("FAIL", f"unittest discover exited {proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--no-checksum", action="store_true",
                        help="skip the 2.1 GB adler32 pass over the CMS file")
    parser.add_argument("--run-tests", action="store_true",
                        help="also run the unittest suite")
    args = parser.parse_args()

    print(f"OTUS Linux compatibility check on {platform.node()}")
    print(f"checkout: {REPO_ROOT}")

    check_platform()
    check_line_endings()
    versions = check_imports()
    torch_info = check_torch_cuda()
    check_repo()
    check_data(args.data_root.expanduser().resolve(), verify_checksum=not args.no_checksum)
    if args.run_tests:
        run_tests()

    fails = [m for s, m in RESULTS if s == "FAIL"]
    warns = [m for s, m in RESULTS if s == "WARN"]
    print("\n" + "=" * 68)
    print(f"summary: {len(fails)} FAIL, {len(warns)} WARN, "
          f"{len([1 for s, _ in RESULTS if s == 'PASS'])} PASS")
    if fails:
        print("\nfailures:")
        for message in fails:
            print(f"  - {message}")
    print("=" * 68)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
